import datetime as dt
from enum import Enum, unique
from typing import Any

import pandas as pd

from neuraflux.global_variables import ENERGY_KEY, TARIFF_KEY
from neuraflux.schemas.base import BaseSchema


# TODO: Create subclasse(s) for specific rates
class Tariff(BaseSchema):
    utility: str = ""  # Utility company that provides the tariff
    tariff_class: str = "General"  # Class of the tariff
    currency: str = "CAD"  # Currency in which the tariff is expressed
    base_rate: float = 0.0  # Base rate for the tariff
    charge_period: str = "Monthly"  # Period for which the charges are calculated
    description: str = ""  # Description of the tariff
    # Indicates whether the tariff includes time-of-use rates
    has_time_of_use_rates: bool = False
    # Indicates whether the tariff includes demand charges
    has_demand_charges: bool = False
    # Indicates whether the tariff includes tiered rates
    has_tiered_rates: bool = False
    # Indicates whether the tariff includes contracted rates
    has_contracted_rates: bool = False
    # Indicates if there are specific conditions for tariff applicability
    has_tariff_applicability: bool = False
    # Indicates if there are specific conditions for rate applicability
    has_rate_applicability: bool = False
    # Indicates whether the tariff supports net metering, allowing customers to sell excess energy back to the grid
    has_net_metering: bool = False
    charge_type_consumption: bool = False
    charge_type_demand: bool = False
    charge_type_fixed: bool = False
    charge_type_minimum: bool = False
    charge_type_maximum: bool = False
    charge_type_quantity: bool = False
    kw_range: tuple[float | None, float | None] = (None, None)
    kwh_range: tuple[float | None, float | None] = (None, None)

    def __str__(self):
        return self.client_facing_name()

    def calculate_price(
        self,
        time: dt.datetime | None = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        raise NotImplementedError

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def client_facing_name(self) -> str:
        raise NotImplementedError


class NoTariff(Tariff):
    utility: str = "None"  # Utility company that provides the tariff
    tariff_class: str = "None"  # Class of the tariff
    currency: str = ""  # Currency in which the tariff is expressed
    base_rate: float = 0.0  # Base rate for the tariff
    charge_period: str = "None"  # Period for which the charges are calculated
    def calculate_price(
        self,
        time: dt.datetime | None = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        return 0

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        df[TARIFF_KEY] = 0
        return df

    def client_facing_name(self) -> str:
        return "No Tariff"


class FlatRateTariff(Tariff):
    base_rate: float = 0.1
    has_net_metering: bool = True
    charge_type_consumption: bool = True

    def calculate_price(
        self,
        time: dt.datetime = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        if energy is not None:
            return self.base_rate * energy
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        df[TARIFF_KEY] = df[ENERGY_KEY].values * self.base_rate
        return df

    def client_facing_name(self) -> str:
        return "Flat Rate"


class HydroQuebecDTariff(Tariff):
    utility: str = "Hydro Quebec"
    tariff_class: str = "Residential"
    base_rate: float = 0.06509
    has_tiered_rates: bool = True
    has_rate_applicability: bool = True
    has_net_metering: bool = True
    charge_type_consumption: bool = True
    charge_type_fixed: bool = True
    charge_type_quantity: bool = True
    patrimonial_value: float = 0.06509
    highest_value: float = 0.10041
    cuttoff_energy_value: float = 40.0

    def calculate_price(
        self,
        time: dt.datetime | None = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        if energy is not None:
            period_energy = (
                0
                if "period_energy" not in other_info.keys()
                else other_info["period_energy"]
            )
            low_consumption_energy = min(self.cuttoff_energy_value, period_energy)
            high_consumption_energy = max(energy - self.cuttoff_energy_value, 0)
            price = float(
                low_consumption_energy * self.patrimonial_value
                + high_consumption_energy * self.highest_value
            )
            return price
        return None

    def client_facing_name(self) -> str:
        return "HydroQuebec Rate D"

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class HydroQuebecMTariff(Tariff):
    utility: str = "Hydro Quebec"
    base_rate: float = 0.005567
    has_tiered_rates: bool = True
    has_rate_applicability: bool = True
    has_net_metering: bool = True
    has_demand_charges: bool = True
    has_rate_applicability: bool = True
    charge_type_consumption: bool = True
    charge_type_demand: bool = True
    charge_type_fixed: bool = True
    charge_type_quantity: bool = True
    energy_price_kwh: float = 0.005567
    power_price_kw: float = 16.139
    cuttoff_energy_value: int = 210000

    def calculate_price(
        self,
        time: dt.datetime | None = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        raise NotImplementedError
        if energy is not None:
            return self.energy_price_kwh * energy
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def client_facing_name(self) -> str:
        return "HydroQuebec Rate M"


class OntarioTOUTariff(Tariff):
    utility: str = "Toronto Hydro"
    base_rate: float = 0.074
    has_contracted_rates: bool = True
    has_time_of_use_rates: bool = True
    has_net_metering: bool = True
    has_rate_applicability: bool = True
    charge_type_consumption: bool = True
    charge_type_fixed: bool = True
    charge_type_quantity: bool = True
    kw_range: tuple[float | None, float | None] = (0., 50.)
    on_peak: float = 0.151
    mid_peak: float = 0.102
    off_peak: float = 0.074
    on_peak_range: tuple[int, int] = (11, 17)
    mid_peak_ranges: tuple[tuple[int, int], tuple[int, int]] = (
        (7, 11),
        (17, 19),
    )

    def calculate_price(
        self,
        time: dt.datetime | None = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        if time is not None and energy is not None:
            hour = time.hour

            # Apply Ontario's TOU based on 24h
            if self.on_peak_range[1] > hour >= self.on_peak_range[0]:
                price = self.on_peak * energy / 1000  #
            if (
                self.mid_peak_ranges[0][1] > hour >= self.mid_peak_ranges[0][0]
                or self.mid_peak_ranges[1][1] > hour >= self.mid_peak_ranges[1][0]
            ):
                price = self.mid_peak * energy / 1000
            price = self.off_peak * energy / 1000

            # TODO: Add this as a parameter of the Tariff
            # Not paid injecting energy back on the grid
            # return max(price, 0)
            return price
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for index, row in df.iterrows():
            df.loc[index, TARIFF_KEY] = self.calculate_price(
                time=row.name, energy=row[ENERGY_KEY]
            )
        return df

    def client_facing_name(self) -> str:
        return "General, TOU < 50 kW"
        #return "Ontario TOU"


class DynamicPricingTariff(Tariff):
    def load_data(self, price_file_path: str) -> None:
        self.dynamic_prices_df = pd.read_csv(
            price_file_path, index_col=0, parse_dates=True
        )

    def calculate_price(
        self,
        time: dt.datetime | None = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        if energy is not None:
            # Load data just the first time
            if not hasattr(self, "dynamic_prices_df"):
                self.load_data("datasets/dynamic_prices_2023.csv")

            # Find the closest time index in the DataFrame to the given time
            closest_time = self.dynamic_prices_df.index.get_loc(time)
            energy_price_kwh = self.dynamic_prices_df.iloc[closest_time]["price"]
            return energy_price_kwh * energy
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        prices = [
            self.calculate_price(time=row.name, energy=row["energy"])
            for index, row in df.iterrows()
        ]
        return pd.DataFrame(prices, index=df.index, columns=[TARIFF_KEY])

    def client_facing_name(self) -> str:
        return "Dynamic Pricing"


class HOEPMarketTariff(Tariff):
    def __init__(
        self,
        hoep_file_path: str = "datasets/hoep_interpolated_2023.csv",
    ):
        # Reading the dynamic prices from the provided CSV file
        self.dynamic_prices_df = pd.read_csv(
            hoep_file_path, index_col=0, parse_dates=True
        )
        self.dynamic_prices_df.index = self.dynamic_prices_df.index - dt.timedelta(
            hours=1
        )

    def calculate_price(
        self,
        time: dt.datetime | None = None,
        power: float | None = None,
        energy: float | None = None,
        other_info: dict[str, Any] = {},
    ) -> float | None:
        if energy is not None:
            # Find the closest time index in the DataFrame to the given time
            closest_time = self.dynamic_prices_df.index.get_loc(time)
            energy_price_kwh = self.dynamic_prices_df.iloc[closest_time]["HOEP"]
            return energy_price_kwh * energy
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        prices = [
            self.calculate_price(time=row.name, energy=row[ENERGY_KEY])
            for index, row in df.iterrows()
        ]
        df[TARIFF_KEY] = prices
        return df

    def client_facing_name(self) -> str:
        return "HOEP Real-Time Market"


@unique
class AvailableTariffsEnum(Enum):
    NO_TARIFF = NoTariff
    FLAT_RATE = FlatRateTariff
    HYDRO_QUEBEC_D = HydroQuebecDTariff
    ONTARIO_GEN_TOU = OntarioTOUTariff
    HYDRO_QUEBEC_M = HydroQuebecMTariff
    DYNAMIC_PRICING = DynamicPricingTariff
    #HOEP_MARKET = HOEPMarketTariff

    @classmethod
    def list_tariffs(cls):
        tariffs_list = list(
            map(lambda tariff: tariff.value().client_facing_name(), cls)
        )
        tariffs_list.sort()
        return tariffs_list

    @classmethod
    def get_tariff_class(cls, tariff_name: str):
        for tariff in cls:
            if (
                tariff.value().client_facing_name() == tariff_name
                or tariff.name.lower() == tariff_name.lower()
            ):
                return tariff.value
        raise ValueError(f"No tariff found with name: {tariff_name}")

    @classmethod
    def from_string(cls, tariff_name: str, **kwargs) -> Tariff:
        return cls.get_tariff_class(tariff_name)(**kwargs)
