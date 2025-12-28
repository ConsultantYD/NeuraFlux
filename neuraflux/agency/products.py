"""Optimization products (reward and feature definitions).

Products define the learning objective for an agent by:

- computing per-timestep rewards from an augmented dataframe, and
- optionally adding additional features/metrics columns used by training or reporting.

The simulation and agent pipelines expect:

- :meth:`Product.calculate_rewards` to return a numpy array (shape: ``(n_rows, n_rewards)``),
  and
- :meth:`Product.get_reward_names` to list the corresponding reward column names.

Most products operate on dataframes indexed by a timezone-aware or naive ``DatetimeIndex``.
"""

import datetime as dt
from abc import ABCMeta, abstractmethod
from enum import Enum, unique

import numpy as np
import pandas as pd

from neuraflux.global_variables import (
    DONE_KEY,
    ENERGY_KEY,
    REWARD_KEY,
    TARIFF_KEY,
)


class Product(metaclass=ABCMeta):
    """Base class for optimization products."""

    def __init__(self, period: int = 24):
        """Initialize a product with its episode period.

        Args:
            period: Period length in hours used to mark episode boundaries in
                :meth:`calculate_dones`. Must be ``<= 24``.
        """
        if period > 24:
            raise ValueError("Product period cannot exceed 24h")
        self.period = period

    def __str__(self):
        return self.client_facing_name()

    @abstractmethod
    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        """Compute rewards for each row of the provided dataframe."""
        raise NotImplementedError

    def calculate_total_price(self, df: pd.DataFrame) -> np.ndarray:
        """Return the total price signal used for reporting (defaults to tariff cost)."""
        return df[[TARIFF_KEY]].values

    @abstractmethod
    def client_facing_name(self) -> str:
        """Return a human-readable product name."""
        raise NotImplementedError

    # NOTE: override this method if the product has multiple rewards
    def get_reward_names(self) -> list[str]:
        """Return the reward column names produced by :meth:`calculate_rewards`."""
        return [REWARD_KEY]

    def calculate_dones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a boolean ``done`` column marking episode boundaries.

        Episodes are defined by :attr:`period` (in hours) and evaluated on minute-of-day.
        The current implementation marks ``done`` 10 minutes before the end of each period
        to align with 5-minute timestep data and next-state reward conventions.

        Args:
            df: Input dataframe indexed by :class:`pandas.DatetimeIndex`.

        Returns:
            A copy of ``df`` with a boolean ``DONE_KEY`` column.

        Raises:
            ValueError: If ``df`` is not indexed by a :class:`pandas.DatetimeIndex`.
        """
        df = df.copy()

        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Calculate the minute of the day for each row
        minutes_of_day = df.index.hour * 60 + df.index.minute

        # Mark the episode boundary 10 minutes before the end of a cycle.
        cycle_minutes = self.period * 60
        df[DONE_KEY] = (((minutes_of_day + 10) % cycle_minutes) == 0).astype(bool)
        return df

    # For products offering additional features in learning process
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return original df with no features, by default
        return df


class SimpleTariffOptimizationProduct(Product):
    """Minimize tariff cost (reward = -tariff)."""

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        return -df[[TARIFF_KEY]].values

    def client_facing_name(self) -> str:
        return "Tariff Optimization"


class HVACTariffAndComfortProduct(Product):
    """Minimize tariff cost while penalizing comfort violations for HVAC control."""

    def __init__(self, period: int = 4):
        super().__init__(period)

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()

        # Energy / cost: use tariff as the primary signal. Tariff magnitudes vary across
        # datasets/unit conventions, so a fixed scaling factor keeps gradients well-scaled
        # without changing the underlying tariff math.
        tariff_weight = 10000.0
        reward = -df[TARIFF_KEY].astype(float).values * tariff_weight

        # Comfort
        # Reward is associated with the (state, action) at time t, but comfort
        # is observed on the next state after applying the control. We therefore
        # evaluate discomfort using the next-timestep temperatures and setpoints.
        temp_cols = [col for col in df.columns if "temperature_" in col]
        sp_cool_now = df["cool_setpoint"].astype(float).values
        sp_heat_now = df["heat_setpoint"].astype(float).values
        sp_cool_next = np.roll(sp_cool_now, -1)
        sp_heat_next = np.roll(sp_heat_now, -1)
        sp_cool_next[-1] = sp_cool_now[-1]
        sp_heat_next[-1] = sp_heat_now[-1]
        for col in temp_cols:
            temps_now = df[col].astype(float).values
            temps_next = np.roll(temps_now, -1)
            temps_next[-1] = temps_now[-1]
            # Calculate discomfort as the absolute deviation from
            # cooling setpoint if warmer, or heating setpoint if colder
            # 0 otherwise
            comfort_weight = 10.0
            reward += np.where(
                temps_next > sp_cool_next,
                -np.square(temps_next - sp_cool_next) * comfort_weight,
                np.where(
                    temps_next < sp_heat_next,
                    -np.square(sp_heat_next - temps_next) * comfort_weight,
                    0,
                ),
            )
        df[REWARD_KEY] = reward
        return df[[REWARD_KEY]].values

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comfort-related reporting features.

        Adds:
            - ``discomfort_c``: sum of degrees outside the heating/cooling deadband across zones.
            - ``comfort_violation``: boolean indicator that any zone violates the deadband.
        """
        df = df.copy()

        temp_cols = [col for col in df.columns if "temperature_" in col]
        if not temp_cols or "cool_setpoint" not in df.columns or "heat_setpoint" not in df.columns:
            return df

        sp_cool = df["cool_setpoint"].astype(float).values
        sp_heat = df["heat_setpoint"].astype(float).values

        discomfort = np.zeros(df.shape[0], dtype=float)
        for col in temp_cols:
            temps = df[col].astype(float).values
            too_hot = np.maximum(temps - sp_cool, 0.0)
            too_cold = np.maximum(sp_heat - temps, 0.0)
            discomfort += too_hot + too_cold

        df["discomfort_c"] = discomfort
        df["comfort_violation"] = discomfort > 0.0
        return df

    def client_facing_name(self):
        return "HVAC Energy and Comfort Optimization"


class DemandResponseProduct(Product):
    """DR product combining an energy signal and tariff cost."""

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df[REWARD_KEY] = -df[ENERGY_KEY].values  # type: ignore
        df.loc[
            (df.index.hour < 15) | (df.index.hour >= 19), REWARD_KEY  # type: ignore # noqa: E501
        ] = 0

        # Subtract the tariff cost from the reward
        df[REWARD_KEY] = df[REWARD_KEY] - df[TARIFF_KEY].values  # type: ignore
        return df[[REWARD_KEY]].values

    def get_reward_names(self) -> list[str]:
        return [REWARD_KEY]

    def calculate_total_price(self, df: pd.DataFrame) -> np.ndarray:
        return -1 * self.calculate_rewards(df)

    def client_facing_name(self) -> str:
        return "Demand Response"


class PureDemandResponseProduct(Product):
    """DR product with a pure energy-based reward during event hours."""

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df[REWARD_KEY] = -df[ENERGY_KEY].values  # type: ignore
        df.loc[
            (df.index.hour < 15) | (df.index.hour >= 19), REWARD_KEY  # type: ignore # noqa: E501
        ] = 0
        return df[[REWARD_KEY]].values

    def get_reward_names(self) -> list[str]:
        return [REWARD_KEY]

    def calculate_total_price(self, df: pd.DataFrame) -> np.ndarray:
        return -1 * self.calculate_rewards(df)

    def client_facing_name(self) -> str:
        return "Pure Demand Response"


class CAISODynamicPricingProduct(Product):
    """Market-price product using CAISO dynamic prices."""

    def __init__(
        self,
        price_file_path: str = "datasets/dynamic_pricing_2023.csv",
        period: int = 24,
    ):
        # Reading the dynamic prices from the provided CSV file
        self.dynamic_prices_df = pd.read_csv(
            price_file_path, index_col=0, parse_dates=True
        )
        # Set the index to be utc
        self.dynamic_prices_df.index = self.dynamic_prices_df.index.tz_localize("UTC")

        # Run default class init
        super().__init__(period)

    def get_market_price(self, time: dt.datetime) -> float:
        # Find the closest time index in the DataFrame to the given time
        closest_time = self.dynamic_prices_df.index.get_loc(time)
        return self.dynamic_prices_df.iloc[closest_time]["price"]

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        df = df.copy()
        prices = [
            self.get_market_price(time=row.name) * -row[ENERGY_KEY]
            for index, row in df.iterrows()
        ]
        df[REWARD_KEY] = prices
        return df[[REWARD_KEY]].values

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy the DataFrame to avoid modifying the original one
        df = df.copy()

        # Iterate over the DataFrame rows
        for index, _ in df.iterrows():
            # Calculate the timestamps for the next 12 intervals (5 minutes each)
            next_times = [index + dt.timedelta(minutes=5 * i) for i in range(1, 6 + 1)]

            # Retrieve price for current timestamp
            if index in self.dynamic_prices_df.index:
                df.loc[index, "market_price_t"] = self.get_market_price(index)
            else:
                raise ValueError(
                    f"Missing CAISO price data in Product for timestamp {index}."
                )
            # Retrieve prices for these timestamps
            for i, time in enumerate(next_times):
                # Check if the time exists in dynamic_prices_df, if not, handle appropriately
                if time in self.dynamic_prices_df.index:
                    price = self.get_market_price(time)
                else:
                    raise ValueError(
                        f"Missing CAISO price data in Product for timestamp {time}."
                    )

                # Add the price to the DataFrame
                df.loc[index, f"market_price_t+{i + 1}"] = price

        return df

    def calculate_total_price(self, df: pd.DataFrame) -> np.ndarray:
        return -1 * self.calculate_rewards(df)

    def client_facing_name(self) -> str:
        return "Dynamic Pricing (CAISO)"


class ERCOTMarketProduct(Product):
    """Market-price product using ERCOT real-time prices."""

    def __init__(
        self,
        data_filepath: str = "datasets/ERCOT_HB_BUSAVG_2023_2024_5min_interpolated.parquet",
        period: int = 24,
    ):
        # Reading the dynamic prices from the parquet file
        self.rt_df = pd.read_parquet(data_filepath)
        self.col = list(self.rt_df.columns)[0]

        # Run default class init
        super().__init__(period)

    def get_market_price(self, time: dt.datetime) -> float:
        # Find the closest time index in the DataFrame to the given time
        closest_time = self.rt_df.index.get_loc(time)
        return self.rt_df.iloc[closest_time][self.col]

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        df = df.copy()
        pnl = np.array(
            [
                self.get_market_price(time=row.name) * -row[ENERGY_KEY]
                for index, row in df.iterrows()
            ],
            dtype=float,
        )
        df[REWARD_KEY] = pnl / 1000.0
        return df[[REWARD_KEY]].values

    def calculate_total_price(self, df: pd.DataFrame) -> np.ndarray:
        return -1 * self.calculate_rewards(df)

    def client_facing_name(self) -> str:
        return "Arbitrage (ERCOT Market)"


class HOEPMarketProduct(Product):
    """Market-price product using Ontario HOEP prices."""

    def __init__(
        self,
        price_file_path: str = "datasets/hoep_interpolated_2023.csv",
        period: int = 24,
    ):
        # Reading the dynamic prices from the provided CSV file
        self.dynamic_prices_df = pd.read_csv(
            price_file_path, index_col=0, parse_dates=True
        )
        # Set the index to be utc
        self.dynamic_prices_df.index = self.dynamic_prices_df.index.tz_localize("UTC")

        # Run default class init
        super().__init__(period)

    def get_market_price(self, time: dt.datetime) -> float:
        # Find the closest time index in the DataFrame to the given time
        closest_time = self.dynamic_prices_df.index.get_loc(time)
        return self.dynamic_prices_df.iloc[closest_time]["HOEP"]

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        df = df.copy()
        prices = [
            self.get_market_price(time=row.name) * -row[ENERGY_KEY]
            for index, row in df.iterrows()
        ]
        df[REWARD_KEY] = prices
        return df[[REWARD_KEY]].values

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy the DataFrame to avoid modifying the original one
        df = df.copy()

        # Iterate over the DataFrame rows
        for index, _ in df.iterrows():
            # Calculate the timestamps for the next 12 intervals (5 minutes each)
            previous_times = [
                index - dt.timedelta(minutes=5 * i) for i in range(1, 6 + 1)
            ]

            # NOTE: Normally not available in real-world scenarios
            # Retrieve price for current timestamp
            # if index in self.dynamic_prices_df.index:
            #    df.loc[index, "market_price_t"] = self.get_market_price(index)
            # else:
            #    raise ValueError(
            #        f"Missing HOEP price data in Product for timestamp {index}."
            #    )
            # Retrieve prices for these timestamps
            for i, time in enumerate(previous_times):
                # Check if the time exists in dynamic_prices_df, if not, handle appropriately
                if time in self.dynamic_prices_df.index:
                    price = self.get_market_price(time)
                else:
                    raise ValueError(
                        f"Missing CAISO price data in Product for timestamp {time}."
                    )

                # Add the price to the DataFrame
                df.loc[index, f"market_price_t-{i + 1}"] = price

        return df

    def calculate_total_price(self, df: pd.DataFrame) -> np.ndarray:
        return -1 * self.calculate_rewards(df)

    def client_facing_name(self) -> str:
        return "Arbitrage (HOEP Market)"


class HVACBuildingProduct(Product):
    """Legacy building HVAC product with multiple reward components."""

    def __init__(self, period: int = 2):
        self.period = period

    def add_features(self, df):
        df = df.copy()
        if "control_1" in df.columns:
            df[["hvac_1", "hvac_2", "hvac_3"]] = (
                df[["control_1", "control_2", "control_3"]] - 2
            )
        return df

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()

        # Calculate comfort reward as the absolute deviation from
        # cooling setpoint if warmer, or heating setpoint if colder
        # 0 otherwise
        discomforts = []
        for zone in range(1, 4):
            discomfort = np.zeros(df.shape[0])

            filter1 = (df[f"temperature_{zone}"] > df["cool_setpoint"]).values
            discomfort[filter1] = (
                df["cool_setpoint"] - df[f"temperature_{zone}"]
            ).values[filter1]

            filter2 = (df[f"temperature_{zone}"] < df["heat_setpoint"]).values
            discomfort[filter2] = (
                df[f"temperature_{zone}"] - df["heat_setpoint"]
            ).values[filter2]
            discomforts.append(discomfort)

        df[REWARD_KEY + "_DISCOMFORT"] = (
            discomforts[0] + discomforts[1] + discomforts[2]
        )

        # Create a filter where all control keys are equal to 2, and where discomfort is 0
        if "control_1" in df.columns:
            filter = (
                (df["control_1"] == 2)
                & (df["control_2"] == 2)
                & (df["control_3"] == 2)
                & (df[REWARD_KEY + "_DISCOMFORT"] == 0)
            )

            df.loc[filter, REWARD_KEY + "_DISCOMFORT"] = 1

        # Calculate energy cost reward (M Rate HydroQuebec)
        df[REWARD_KEY + "_COST"] = df[ENERGY_KEY].values * -0.005567

        # Calculate energy reward as the energy consumption
        df[REWARD_KEY + "_ENERGY"] = df[ENERGY_KEY].values * -1

        return df[self.get_reward_names()].values

    def get_reward_names(self) -> list[str]:
        reward_names = [
            REWARD_KEY + "_DISCOMFORT",
            REWARD_KEY + "_COST",
            REWARD_KEY + "_ENERGY",
        ]
        return reward_names

    def client_facing_name(self) -> str:
        return "Building HVAC Optimization"


class EvDrTouGhgProduct(Product):
    """Example EV product combining tariff cost, GHG, and demand-response events."""

    def __init__(self, period: int = 2):
        self.period = period
        self.duck_curve_values = [
            0.2,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.55,
            0.7,
            0.55,
            0.5,
            0.2,
            0.1,
            0.0,
            0.1,
            0.2,
            0.4,
            0.5,
            0.7,
            0.9,
            1.0,
            0.9,
            0.75,
            0.5,
            0.3,
        ]

    def get_emission_rate(self, time: dt.datetime) -> float:
        hour = time.hour
        seed = time.day

        rng = np.random.RandomState(seed)  # noqa: NPY002 (legacy RNG; kept for reproducibility)

        # Get the base value from the hardcoded array
        base_value = self.duck_curve_values[hour]

        # Add random variation to the value, e.g., +/- 5%
        variation = rng.uniform(-0.1, 0.1)
        final_value = np.clip(base_value + variation, 0, 1)

        return final_value

    def is_dr_event(self, time: dt.datetime) -> bool:
        """Return whether the given time falls within a DR event window.

        Args:
            time: Current time.

        Returns:
            ``True`` if the time is within the (randomized) DR event hour.
        """
        # Use the day of the year as the seed
        seed = time.day
        rng = np.random.RandomState(seed)  # noqa: NPY002 (legacy RNG; kept for reproducibility)

        # Randomly select an hour for the DR event between 5 PM and 8 PM
        dr_event_hour = rng.choice([17, 18, 19])  # 5 PM, 6 PM, or 7 PM

        # Check if the current hour matches the DR event hour
        return time.hour == dr_event_hour

    def calculate_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        df = df.copy()
        emissions = [
            self.get_emission_rate(time=row.name) * -row[ENERGY_KEY]
            for index, row in df.iterrows()
        ]
        cost = [
            min((row[TARIFF_KEY] * -row[ENERGY_KEY]), 0)
            + (self.is_dr_event(row.name) * -row[ENERGY_KEY])
            for index, row in df.iterrows()
        ]
        df[REWARD_KEY + "_COST"] = cost
        df[REWARD_KEY + "_GHG"] = emissions
        return df[self.get_reward_names()].values

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy the DataFrame to avoid modifying the original one
        df = df.copy()

        # Add event column
        dr_event = [self.is_dr_event(time=row.name) for index, row in df.iterrows()]
        df["dr_event"] = dr_event

        # Add GHG emission rate column
        emissions = [
            self.get_emission_rate(time=row.name) for index, row in df.iterrows()
        ]
        df["emission_rate"] = emissions

        return df

    def get_reward_names(self) -> list[str]:
        reward_names = [REWARD_KEY + "_COST", REWARD_KEY + "_GHG"]
        return reward_names

    def calculate_total_price(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def client_facing_name(self) -> str:
        return "Tariff, GHG, and DR Optimization"


@unique
class AvailableProductsEnum(Enum):
    """Registry of available product implementations."""

    DEMAND_RESPONSE = DemandResponseProduct
    PURE_DEMAND_RESPONSE = PureDemandResponseProduct
    DYNAMIC_PRICING = CAISODynamicPricingProduct
    EV_DR = EvDrTouGhgProduct
    ERCOT_ARBITRAGE = ERCOTMarketProduct
    HVAC_BUILDING = HVACBuildingProduct
    HOEP_MARKET = HOEPMarketProduct
    SIMPLE_TARIFF_OPT = SimpleTariffOptimizationProduct
    HVAC_TARIFF_COMFORT = HVACTariffAndComfortProduct

    @classmethod
    def list_products(cls):
        products_list = list(
            map(lambda product: product.value().client_facing_name(), cls)
        )
        products_list.sort()
        return products_list

    @classmethod
    def get_product_class(cls, product_name: str):
        for product in cls:
            if (
                product.value().client_facing_name() == product_name
                or product.name.lower() == product_name.lower()
            ):
                return product.value
        raise ValueError(f"No product found with name: {product_name}")

    @classmethod
    def from_string(cls, product_name: str, **kwargs):
        return cls.get_product_class(product_name)(**kwargs)
