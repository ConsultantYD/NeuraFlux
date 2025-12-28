import datetime as dt
import json
import logging
import os
from copy import copy
from typing import Any

import dill
import numpy as np
import pandas as pd

from neuraflux.agency.control_module import ControlModule
from neuraflux.agency.data_module import DataModule
from neuraflux.agency.dqn import DDQNPREstimator
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.replay_buffer import ReplayBuffer
from neuraflux.agency.utils_control import (
    convert_data_to_experience,
    convert_data_to_state,
    get_full_state_signals_from_rl_config,
)
from neuraflux.agency.utils_data import (
    add_product_data_to_df,
    add_tariff_data_to_df,
    add_vm_data_to_df,
    collect_signals_from_asset,
    cron_matches,
    get_active_config_based_on_duration,
    get_x_columns,
    push_df_as_partitionned_parquet,
    read_parquet_table,
    tf_all_cyclic,
)
from neuraflux.agency.utils_policies import fixed_policy, hvac_policy, q_policy, random_policy
from neuraflux.agency.utils_registries import (
    get_entities_in_registry,
    load_dqn_estimator_from_registry,
    load_replay_buffer_from_registry,
    push_dqn_estimator_to_registry,
    push_replay_buffer_to_registry,
)
from neuraflux.agency.utils_rl_training import simple_training_loop
from neuraflux.agency.utils_trajectories import Trajectory
from neuraflux.global_variables import (
    CONTROL_KEY,
    DONE_KEY,
    DT_FILE_STR_FORMAT,
    EPSILON_KEY,
    INFERENCE_BACKEND_KEY,
    LOG_ENTITY_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_SIM_T_KEY,
    MS_AGENT_CONTROL_DATA_KEY,
    MS_AGENT_REAL_TRAINING_KEY,
    MS_AGENT_SIM_DATA_KEY,
    MS_AGENT_SIM_TRAINING_KEY,
    MS_ASSET_SIGNAL_DATA_KEY,
    MODEL_NAME_KEY,
    MODEL_TRAINING_TYPE_KEY,
    POLICY_KEY,
    TIMESTAMP_KEY,
    CONTROL_SELECTION_ENABLED_KEY,
    OAT_KEY,
)
from neuraflux.local_typing import AgentInMemoryStorageType, AssetType, UidType
from neuraflux.schemas.agency import (
    AgentConfig,
    ControlSelectionConfig,
    RealLearningConfig,
    SimLearningConfig,
)
from neuraflux.schemas.control import DiscreteControl
from neuraflux.time_ref import TimeInfo


class Agent:
    """
    Represents an agent in the simulation.
    The agent is responsible for controlling an asset and interacting with the environment.
    """

    def __init__(
        self,
        uid: UidType | None = None,
        config: AgentConfig | None = None,
        directory: str = "",
        data_module: DataModule | None = None,
        control_module: ControlModule | None = None,
        # prediction_module: PredictionModule | None = None,
        time_info: TimeInfo | None = None,
    ) -> None:
        """
        Initialize the agent with the given configuration and modules.
        Args:
            uid (UidType): Unique identifier for the agent.
            config (AgentConfig | None): Configuration for the agent. Defaults to None.
            data_module (DataModule | None): Data module for data retrieval and storage. Defaults to None.
            control_module (ControlModule | None): Control module for controlling the asset. Defaults to None.
            prediction_module (PredictionModule | None): Prediction module for forecasting. Defaults to None.
            time_info (TimeInfo | None): Time information for the simulation. Defaults to None.
        """
        # Return directly if uid is None (used for loading from file)
        if uid is None:
            return

        # Update internal attributes
        self.uid = uid
        self.config = config
        self.directory = directory
        self.data_dir = os.path.join(directory, "data")
        self.sim_data_dir = os.path.join(directory, "sim_data")
        self.buffer_registry_dir = os.path.join(directory, "buffer_registry")
        self.dqn_registry_dir = os.path.join(directory, "dqn_registry")

        # Time reference
        if time_info is not None:
            self.update_time_info(time_info)

        # Modules
        self.data_module = data_module
        self.control_module = control_module
        # self.prediction_module = prediction_module

        # Store agent initialization time information
        self.initial_time_info = time_info
        self.elapsed_time = 0

        # Initialize agent's asset and shadow asset as None
        self.asset = None
        self.shadow_asset = None

        # Agent's local data store
        self._memory_storage: AgentInMemoryStorageType = {
            MS_AGENT_CONTROL_DATA_KEY: [],
            MS_AGENT_SIM_DATA_KEY: [],
            MS_AGENT_REAL_TRAINING_KEY: [],
            MS_AGENT_SIM_TRAINING_KEY: [],
            MS_ASSET_SIGNAL_DATA_KEY: [],
        }

        # Create shortcuts, for convenience
        self._generate_shortcuts()

        self.epsilon = 1.0
        self.control_ready = False
        self.oat: float | None = None
        self._last_control_provenance: dict[str, Any] = {}

        # Ensure a stable training summary artifact exists for downstream consumers.
        try:
            os.makedirs(self.directory, exist_ok=True)
            if not os.path.exists(self._get_training_summary_path()):
                self._save_training_summary(
                    {"n_real_trainings": 0, "n_sim_trainings": 0}
                )
        except Exception:
            pass

    def __call__(self, *args, **kwargs):
        """
        Utility, call the agent to run its main function.
        This method is a wrapper around the run method.
        """
        return self.run(*args, **kwargs)

    def get_control(
        self, policy, policy_kwargs: dict, df: pd.DataFrame | None = None
    ) -> list[int]:
        """
        Get the control for the agent based on the control selection configuration.
        Args:
            policy (str): The policy to use for control selection.
            policy_kwargs (dict): Additional arguments for the policy.
            df (pd.DataFrame | None): The data to use for control selection. Defaults to None.
        Returns:
            list[DiscreteControl]: The control for the agent.
        """
        action_size = self.config.control.rl_config.action_size
        n_controllers = self.config.control.n_controllers
        self._last_control_provenance = {}
        self._last_effective_policy = policy

        if policy == "q_policy":
            epsilon = float(policy_kwargs.get("epsilon", 0.0))
            use_lite_inference = bool(policy_kwargs.get("use_lite_inference", False))
            q_factors, provenance = self.get_q_factors_for_action(
                df=df, use_lite_inference=use_lite_inference
            )
            self._last_control_provenance = provenance
            controls = q_policy(q_values=q_factors, epsilon=epsilon)

        elif policy == "random_policy":
            controls = random_policy(action_size=action_size, n_controllers=n_controllers)

        elif policy == "fixed_policy":
            if "action" in policy_kwargs:
                action = policy_kwargs["action"]
            elif "actions" in policy_kwargs:
                action = policy_kwargs["actions"]
            else:
                raise ValueError(
                    "fixed_policy requires policy_kwargs['action'] (int) "
                    "or policy_kwargs['actions'] (list[int])."
                )
            controls = fixed_policy(action=action, n_controllers=n_controllers)
            for a in controls:
                if a < 0 or a >= action_size:
                    raise ValueError(
                        f"fixed_policy action {a} out of bounds for action_size={action_size}."
                    )

        elif policy == "hvac_policy":
            if df is None:
                df = self.get_rl_data_for_timestamp()

            epsilon = float(policy_kwargs.get("epsilon", 0.0))
            comfort_constraint = bool(policy_kwargs.get("comfort_constraint", True))
            use_lite_inference = bool(policy_kwargs.get("use_lite_inference", False))
            # If no trained estimator exists yet, fall back to the default controller
            # computed from the provided dataframe (works for both real control and
            # simulated trajectories, without relying on the live asset's internal state).
            _, q_estimator_meta = self.get_q_estimator(self.dqn_registry_dir)
            if q_estimator_meta.get("name") is None:
                self._last_control_provenance = {
                    MODEL_NAME_KEY: None,
                    MODEL_TRAINING_TYPE_KEY: None,
                    INFERENCE_BACKEND_KEY: None,
                }
                self._last_effective_policy = "auto_control"

                if df.empty:
                    raise ValueError(
                        "hvac_policy fallback requires a non-empty dataframe of observations."
                    )

                decision_row = df.iloc[-1]
                state_cols = get_x_columns(self.config)
                temperatures = (
                    pd.to_numeric(decision_row[state_cols], errors="coerce")
                    .astype(float)
                    .to_numpy()
                )
                heat_sp = float(decision_row["heat_setpoint"])
                cool_sp = float(decision_row["cool_setpoint"])

                hvac_cols = [f"hvac_{i+1}" for i in range(n_controllers)]
                if all(c in df.columns for c in hvac_cols):
                    hvac_states = (
                        pd.to_numeric(decision_row[hvac_cols], errors="coerce")
                        .fillna(0.0)
                        .astype(float)
                        .to_numpy()
                    )
                else:
                    hvac_states = np.zeros(n_controllers, dtype=float)

                controls = []
                for hvac_state, temp in zip(hvac_states, temperatures):
                    if temp > cool_sp:
                        # Keep stage 2 on if it was already on, or if gap is large
                        controls.append(0 if hvac_state == -2 or temp - cool_sp > 1 else 1)
                    elif temp < heat_sp:
                        # Keep stage 2 on if it was already on, or if gap is large
                        controls.append(4 if hvac_state == 2 or heat_sp - temp > 1 else 3)
                    else:
                        controls.append(2)
                return [int(c) for c in controls]

            q_factors, provenance = self.get_q_factors_for_action(
                df=df, use_lite_inference=use_lite_inference
            )
            self._last_control_provenance = provenance

            state_cols = get_x_columns(self.config)
            temp = df.loc[df.index[-1], state_cols].values
            sp_vec = df.loc[df.index[-1], ["heat_setpoint", "cool_setpoint"]].values
            sp = (sp_vec[0], sp_vec[1])
            controls = hvac_policy(
                temperatures=temp,
                setpoints=sp,
                q_values=q_factors,
                epsilon=epsilon,
                comfort_constraint=comfort_constraint,
            )

        else:
            raise ValueError(f"Unknown policy {policy}. Please check the configuration.")

        return controls

    def get_state_prediction(
        self,
        prev_t: dt.datetime,
        df: pd.DataFrame,
        asset_type: str | None = None,
    ) -> dict[str, int | float]:
        """
        Get the prediction of state variables for the asset at a specific time step.
        Args:
            prev_t (dt.datetime): The time step BEFORE the one to get the state prediction for.
            df (pd.DataFrame | None): The data to use for state prediction. Defaults to None.
            asset_type (str | None): The type of asset. Defaults to None.
        Returns:
            dict[str, int | float]: The prediction of state variables at the timestep.
        """

        # Use associated asset to derive type if None
        if asset_type is None:
            if self.asset is not None:
                asset_type = self.asset.__class__.NAME
            else:
                raise ValueError(
                    "Asset type is None and no asset is assigned to the agent."
                )

        # Retrieve controls
        n_controls = int(getattr(getattr(self.config, "control", None), "n_controllers", 1))
        control_cols = [CONTROL_KEY + f"_{i+1}" for i in range(n_controls)]
        missing_control_cols = [c for c in control_cols if c not in df.columns]
        if missing_control_cols:
            raise ValueError(
                "Missing control columns required for state prediction "
                f"(missing={missing_control_cols}, available={list(df.columns)})."
            )
        control_frame = df.loc[df.index == prev_t, control_cols]
        if control_frame.empty:
            raise ValueError(
                f"No control row found at prev_t={prev_t} for state prediction."
            )
        controls = control_frame.values.reshape(n_controls)
        # -------------------------------------------------
        # INFER STATE
        # -------------------------------------------------
        state_cols = get_x_columns(self.config)
        n_state_cols = len(state_cols)

        # Commercial Building
        if asset_type == "commercial building":
            previous_state = (
                df.loc[df.index == prev_t, state_cols]
                .values.reshape(n_state_cols)
                .astype(float)
            )

            # Approximate the Building RC model directly using the asset's matrices.
            if self.asset is None or not hasattr(self.asset, "Uinv"):
                raise ValueError(
                    "Commercial building state prediction requires an initialized Building asset."
                )

            if OAT_KEY in df.columns:
                oat = float(df.loc[df.index == prev_t, [OAT_KEY]].values.reshape(-1)[0])
            else:
                oat = float(self.oat) if self.oat is not None else 0.0

            controls_int = controls.astype(int)
            power_vec = np.array(
                [
                    float(self.cpm[a]) if a in (2, 3, 4) else -float(self.cpm[a])
                    for a in controls_int
                ],
                dtype=float,
            )

            # Match Building._update_room_temperature() semantics
            Q_hvac = power_vec * 1000.0
            Q_in = np.zeros(len(Q_hvac), dtype=float)
            term1 = np.dot(self.asset.F, oat)
            term2 = np.multiply(
                self.asset.C.T / (self.asset.config.dt * 60), previous_state
            ).flatten()
            Q = Q_hvac + Q_in + term1 + term2
            new_state_values = np.dot(Q, self.asset.Uinv).diagonal()
            new_state_dict = {
                col: float(val) for col, val in zip(state_cols, new_state_values)
            }
            # Predict HVAC state at the next timestep (used as an observation/RL feature).
            # HVAC stages are encoded as control_value - 2, matching Building.step().
            for i in range(n_controls):
                new_state_dict[f"hvac_{i+1}"] = float(controls_int[i] - 2)
        elif asset_type == "energy storage":
            previous_internal_energy = df.loc[df.index == prev_t, state_cols].values.reshape(
                n_state_cols
            )
            power = self.cpm[int(controls[0])]
            max_energy = self.config.data.signals_info["internal_energy"].max_value
            
            # If the internal energy is 0, we cannot deliver power
            if round(previous_internal_energy[0]) == 0 and power < 0:
                power = 0
            # If the internal energy is at max capacity, we cannot receive power
            elif round(previous_internal_energy[0]) == max_energy and power > 0:
                power = 0
            
            # Calculate energy difference
            time_difference = dt.timedelta(minutes=5)
            energy_difference = power * time_difference.total_seconds() / 3600
            
            # Update energy - must remain between 0 and max capacity
            internal_energy = previous_internal_energy + energy_difference
            internal_energy = np.clip(internal_energy, 0, max_energy)
            new_state_dict = {
                col: val for col, val in zip(state_cols, internal_energy)
            }
        else:
            raise ValueError(
                f"Unknown asset type {asset_type}. Please check the configuration."
            )

        return new_state_dict

    def run(self) -> list[DiscreteControl] | None:
        """
        Run the agent for a given time step.
        """
        logger = logging.getLogger(__name__)

        # 1. Collect asset data
        self.asset_data_collection()

        # 2. Define control and store it in memory
        control_selection_config: ControlSelectionConfig = (
            get_active_config_based_on_duration(
                duration_s=self.get_elapsed_time(),
                config_dict=self.config.control.control_selection,
            )
        )
        if control_selection_config.enabled:
            policy = control_selection_config.policy
            policy_kwargs = control_selection_config.policy_kwargs
            controls = self.get_control(policy=policy, policy_kwargs=policy_kwargs)
            output_controls = [DiscreteControl(c) for c in controls]
        else:
            if self.oat is None:
                raise ValueError(
                    "Agent is missing outside air temperature (oat); call update_environment_info(oat=...) before run()."
                )
            output_controls = self.asset.get_auto_control(self.time_info.t, self.oat)
            controls = [c.value for c in output_controls]
        # Store control taken in memory
        if control_selection_config.enabled:
            policy_name = getattr(self, "_last_effective_policy", None) or policy
        else:
            policy_name = "auto_control"
        epsilon = None
        if policy_name in {"q_policy", "hvac_policy"}:
            epsilon = float(policy_kwargs.get("epsilon", 0.0))
        self._last_effective_policy = None

        self._push_data_dict_to_memory_storage(
            storage_key=MS_AGENT_CONTROL_DATA_KEY,
            data_dict={
                **{CONTROL_KEY + f"_{i+1}": controls[i] for i in range(len(controls))},
                CONTROL_SELECTION_ENABLED_KEY: bool(control_selection_config.enabled),
                POLICY_KEY: policy_name,
                EPSILON_KEY: epsilon,
                MODEL_NAME_KEY: self._last_control_provenance.get(MODEL_NAME_KEY),
                MODEL_TRAINING_TYPE_KEY: self._last_control_provenance.get(
                    MODEL_TRAINING_TYPE_KEY
                ),
                INFERENCE_BACKEND_KEY: self._last_control_provenance.get(
                    INFERENCE_BACKEND_KEY
                ),
            },
            timestamp=self.time_info.t,
        )

        # 3. Push in-memory data to database, and clear it
        memory_dump_freq = self.config.data.memory_dump_freq_cron
        if cron_matches(self.time_info.t, memory_dump_freq):
            self.push_in_memory_data_to_db(self.data_dir)
            self.clear_in_memory_storage()

        # 4 Train using simulated data
        sim_lr_config: SimLearningConfig = get_active_config_based_on_duration(
            duration_s=self.get_elapsed_time(),
            config_dict=self.config.control.sim_learning_configs,
        )
        if sim_lr_config.enabled and cron_matches(
            self.time_info.t, sim_lr_config.trigger_freq_cron
        ):
            n_samples = sim_lr_config.n_samples
            n_traj_per_sample = sim_lr_config.n_traj_per_sample
            recent_df = self.get_data(start_time=self.time_info.t - dt.timedelta(days=1))
            product = AvailableProductsEnum.from_string(self.config.product)
            reward_cols = [c for c in product.get_reward_names() if c in recent_df.columns]
            reward_sum = (
                float(recent_df[reward_cols].sum().sum()) if reward_cols else None
            )
            logger.info(
                {
                    LOG_SIM_T_KEY: self.time_info.t,
                    LOG_ENTITY_KEY: f"Agent({self.uid})",
                    LOG_METHOD_KEY: "run",
                    LOG_MESSAGE_KEY: (
                        "Simulated RL training trigger"
                        f" (n_samples={n_samples}, n_traj_per_sample={n_traj_per_sample}, "
                        f"reward_sum_last_day={reward_sum})"
                    ),
                }
            )
            self.simulated_rl_training(
                n_samples=n_samples,
                n_traj_per_sample=n_traj_per_sample,
                traj_len=sim_lr_config.trajectory_len,
                policy=sim_lr_config.policy,
                policy_kwargs=sim_lr_config.policy_kwargs,
            )

        # 5. Train using real data
        real_lr_config: RealLearningConfig = get_active_config_based_on_duration(
            duration_s=self.get_elapsed_time(),
            config_dict=self.config.control.real_learning_configs,
        )
        rl_train_freq = real_lr_config.trigger_freq_cron
        if real_lr_config.enabled and cron_matches(self.time_info.t, rl_train_freq):
            logger.info(
                {
                    LOG_SIM_T_KEY: self.time_info.t,
                    LOG_ENTITY_KEY: f"Agent({self.uid})",
                    LOG_METHOD_KEY: "run",
                    LOG_MESSAGE_KEY: "Real-data RL training trigger",
                }
            )
            self.rl_training()
        return output_controls

    def asset_data_collection(self):
        """
        Collect data from the asset (and shadow asset if available) and
        stores it in the local data store.
        """
        # Main asset data collection
        tracked_signals = self.config.data.tracked_signals
        asset_signals_dict = collect_signals_from_asset(
            asset=self.asset, tracked_signals=tracked_signals
        )
        # Store in memory, with associated timestamp
        self._push_data_dict_to_memory_storage(
            storage_key=MS_ASSET_SIGNAL_DATA_KEY,
            data_dict=asset_signals_dict,
            timestamp=self.time_info.t,
        )

    def assign_to_asset(
        self, asset: AssetType, shadow_asset: AssetType | None = None
    ) -> None:
        """
        Assign the agent to an asset.
        Args:
            asset (AssetType): The asset to assign to the agent.
            shadow_asset (AssetType | None): The shadow asset to assign to the agent. Defaults to None.
        """
        self.asset = asset
        self.shadow_asset = shadow_asset

    def clear_in_memory_storage(self, key: str | None = None) -> None:
        """
        Clear the in-memory storage for a given key.
        Args:
            key (str | None): The key to clear. If None, clear all keys.
        """
        if key is None:
            for k in self._memory_storage.keys():
                self._memory_storage[k].clear()
        else:
            self._memory_storage[key].clear()

    @classmethod
    def from_dir(cls, agent_dir: str) -> None:
        """
        Load an agent from a directory.
        Args:
            agent_dir (str): The directory to load the agent from.
        Returns:
            Agent: The loaded agent.
        """
        pickle_path = os.path.join(agent_dir, "agent.pkl")
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                internal_dict = dill.load(f)
            agent = cls()
            agent.__dict__.update(internal_dict)
            return agent
        raise ValueError(
            f"Agent not found in directory {agent_dir}. Please check the path."
        )

    def get_config(self) -> AgentConfig:
        """
        Get the agent configuration.
        Returns:
            AgentConfig: The agent configuration.
        """
        return self.config

    def get_data(
        self,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        q_factors: bool = False,
    ):
        # Get latest data from the in-memory storage, and use datetime index
        memory_raw_df = self.get_in_memory_data()
        memory_df: pd.DataFrame | None = None

        if len(memory_raw_df) > 0:
            memory_raw_df.sort_values(by=TIMESTAMP_KEY, inplace=True)
            memory_raw_df[TIMESTAMP_KEY] = pd.to_datetime(memory_raw_df[TIMESTAMP_KEY])
            memory_raw_df.set_index(TIMESTAMP_KEY, inplace=True)

            # Augment the data with additional useful columns
            df = memory_raw_df
            df = add_vm_data_to_df(df, self.cpm)
            df = add_tariff_data_to_df(df, self.config.tariff)
            df = add_product_data_to_df(df, self.config.product)
            memory_df = self._coerce_agent_data_schema(tf_all_cyclic(df))
            # TODO: Add q_factors data

            # Return directly in-memory data if sufficient for user request
            if start_time is not None and start_time in memory_df.index:
                memory_df = memory_df.loc[memory_df.index >= start_time]
                if end_time is not None:
                    memory_df = memory_df.loc[memory_df.index <= end_time]
                return memory_df

        # Get longer-term data from the database
        long_term_data = self._coerce_agent_data_schema(self.get_database_data())

        frames: list[pd.DataFrame] = []
        if isinstance(long_term_data, pd.DataFrame) and not long_term_data.empty:
            frames.append(long_term_data)

        if memory_df is not None and not memory_df.empty:
            mem = memory_df.reset_index()
            frames.append(mem)

        if len(frames) == 0:
            df = pd.DataFrame()
        elif len(frames) == 1:
            df = frames[0].copy()
        else:
            df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
            df = self._coerce_agent_data_schema(df)

        if df.empty:
            return df

        # Add Q-factors data if requested
        if q_factors:
            q_factors = self.get_q_factors(df, use_lite_inference=False)

            seq_len = self.config.control.rl_config.history_length
            action_size = self.config.control.rl_config.action_size
            n_controllers = self.config.control.n_controllers
            n_rewards = 1

            for controller in range(n_controllers):
                for r in range(n_rewards):
                    q_cols = [f"C{controller+1}R{r+1}Q{j}" for j in range(action_size)]
                    df.loc[df.index[seq_len - 1 :], q_cols] = q_factors[controller][
                        :, r, :
                    ]

        # Make sure output index is with a datetime index
        df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY])
        df.set_index(TIMESTAMP_KEY, inplace=True)

        # Apply time filters, if submitted
        if start_time is not None:
            df = df.loc[df.index >= start_time]
        if end_time is not None:
            df = df.loc[df.index < end_time]

        return df

    def _coerce_agent_data_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure agent dataframes (memory + parquet) have a stable schema/dtypes across timesteps.

        This avoids pyarrow dataset read failures when different parquet fragments have null-only
        columns (e.g. epsilon/model provenance) or when a timestep is partially recorded.
        """
        df = df.copy()

        n_controllers = int(getattr(getattr(self.config, "control", None), "n_controllers", 1))

        # Controls (nullable int)
        for i in range(n_controllers):
            col = CONTROL_KEY + f"_{i+1}"
            if col not in df.columns:
                df[col] = pd.Series([pd.NA] * len(df), dtype="Int64")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Control selection + provenance
        if CONTROL_SELECTION_ENABLED_KEY not in df.columns:
            df[CONTROL_SELECTION_ENABLED_KEY] = pd.Series(
                [pd.NA] * len(df), dtype="boolean"
            )
        else:
            df[CONTROL_SELECTION_ENABLED_KEY] = df[CONTROL_SELECTION_ENABLED_KEY].astype(
                "boolean"
            )

        if POLICY_KEY not in df.columns:
            df[POLICY_KEY] = pd.Series([pd.NA] * len(df), dtype="string")
        else:
            df[POLICY_KEY] = df[POLICY_KEY].astype("string")

        if EPSILON_KEY not in df.columns:
            df[EPSILON_KEY] = pd.Series([np.nan] * len(df), dtype="float64")
        else:
            df[EPSILON_KEY] = pd.to_numeric(df[EPSILON_KEY], errors="coerce").astype(
                "float64"
            )

        for key in (MODEL_NAME_KEY, MODEL_TRAINING_TYPE_KEY, INFERENCE_BACKEND_KEY):
            if key not in df.columns:
                df[key] = pd.Series([pd.NA] * len(df), dtype="string")
            else:
                df[key] = df[key].astype("string")

        return df

    def get_rl_data_for_timestamp(
        self, timestamp: dt.datetime | None = None, timestep_s: int = 300
    ) -> pd.DataFrame:
        """
        Get the data for the RL agent at a specific time step.
        Args:
            timestamp (dt.datetime | None): The timestamp to get data for. Defaults to None.
            timestep_s (int): The time step in seconds. Defaults to 300.
        Returns:
            pd.DataFrame: The data for the RL agent.
        """
        # Get the data just for current time-step
        if timestamp is None:
            timestamp = self.time_info.t
        rl_config = self.config.control.rl_config
        rl_seq_len = rl_config.history_length
        delta_required = timestep_s * (rl_seq_len - 1)
        start_time = timestamp - dt.timedelta(seconds=delta_required)
        end_time = timestamp + dt.timedelta(seconds=timestep_s)
        df = self.get_data(start_time=start_time, end_time=end_time)
        return df

    def get_database_data(self):
        """
        Get the data from the local database.
        Returns:
            pd.DataFrame: The data from the database.
        """
        return read_parquet_table(self.data_dir)

    def get_elapsed_time(self) -> float:
        """
        Get the elapsed time since the agent initialization.
        Returns:
            float: The elapsed time in seconds.
        """
        return (self.time_info.t - self.initial_time_info.t).total_seconds()

    def get_in_memory_data(
        self,
        keys: list[str] | None = None,
        join: bool = True,
    ) -> dict[str, pd.DataFrame] | pd.DataFrame:
        """
        Get the in-memory data for the agent.
        Args:
            keys (list[str] | None): The keys to retrieve. If None, retrieve all keys.
            join (bool): Whether to join the data into a single DataFrame, using timestamp. Defaults to True.
        Returns:
            dict[str, pd.DataFrame] | pd.DataFrame: The in-memory data.
        """
        # Check if keys are provided, if not, use all keys
        if keys is None:
            keys = self._memory_storage.keys()

        # Check if keys are valid
        data = {}
        for key in keys:
            data[key] = pd.DataFrame(self._memory_storage[key])

        # Return dict if no join is requested
        if not join:
            return data

        # Otherwise ...
        # 1. Filter and prepare the DataFrames that contain the "timestamp" column.
        dfs = [
            df.set_index(TIMESTAMP_KEY)
            for df in data.values()
            if not df.empty and TIMESTAMP_KEY in df.columns
        ]
        # 2. Concatenate them along the columns using an outer join.
        if len(dfs) == 0:
            return pd.DataFrame()
        final_df = pd.concat(dfs, axis=1, join="outer")
        # 3. Reset the index and sort the DataFrame
        final_df.reset_index(inplace=True)
        final_df.sort_values(by=TIMESTAMP_KEY, inplace=True)

        return final_df

    def get_q_estimators_list(
        self, registry_dir: str | None = None, *, training_type: str = "real"
    ) -> list[str]:
        """
        Get the list of Q estimators for the agent, sorted by creation time.

        Args:
            registry_dir (str | None): The directory for the Q estimator registry. Defaults to None,
                which uses the agent's directory.
            training_type (str): One of {"real", "sim", "any"}; defaults to "real" to avoid mixing
                simulated-training checkpoints into production inference.
        Returns:
            list[str]: The list of Q estimators for the agent.
        """
        registry_dir = self.dqn_registry_dir if registry_dir is None else registry_dir
        available_estimators = get_entities_in_registry(registry_dir)
        agent_available_estimators = [e for e in available_estimators if e.startswith(self.uid)]
        match training_type:
            case "real":
                agent_available_estimators = [e for e in agent_available_estimators if not e.endswith("_sim")]
            case "sim":
                agent_available_estimators = [e for e in agent_available_estimators if e.endswith("_sim")]
            case "any":
                pass
            case _:
                raise ValueError(f"Unknown training_type={training_type!r}; expected 'real', 'sim', or 'any'.")
        agent_available_estimators = sorted(agent_available_estimators)
        return agent_available_estimators

    def get_q_estimator(
        self, registry_dir: str | None = None, name: str | None = None
    ) -> tuple[DDQNPREstimator, dict]:
        """
        Get the Q estimator for the agent.

        Args:
            registry_dir (str | None): The directory for the Q estimator registry. Defaults to None,
                which uses the agent's directory.
            name (str | None): The name of the Q estimator. If None, the latest estimator is used, or
                a new one is created if none exist.
        Returns:
            tuple[DDQNPREstimator, dict]: The Q estimator for the agent and its metadata.
        """
        registry_dir = self.dqn_registry_dir if registry_dir is None else registry_dir

        # Fast path: reuse the in-memory estimator for online control to avoid
        # reloading TensorFlow models from disk on every timestep.
        if name is None and registry_dir == self.dqn_registry_dir:
            cached = getattr(self, "_cached_q_estimator", None)
            cached_meta = getattr(self, "_cached_q_estimator_meta", None)
            if cached is not None and cached_meta is not None:
                return cached, dict(cached_meta)

        # Case 1 - User directly specified the name of the estimator
        if name is not None:
            estimator, metadata = load_dqn_estimator_from_registry(registry_dir, name=name)
            metadata = {} if metadata is None else dict(metadata)
            metadata.setdefault("name", name)
            metadata.setdefault("training_type", "sim" if name.endswith("_sim") else "real")
            return estimator, metadata

        agent_available_real_estimators = self.get_q_estimators_list(
            registry_dir=registry_dir, training_type="real"
        )
        agent_available_sim_estimators = self.get_q_estimators_list(
            registry_dir=registry_dir, training_type="sim"
        )

        # Case 2 - Use the latest available estimator (prefer real over simulated).
        if agent_available_real_estimators or agent_available_sim_estimators:
            latest_estimator = (
                agent_available_real_estimators[-1]
                if agent_available_real_estimators
                else agent_available_sim_estimators[-1]
            )
            estimator, metadata = load_dqn_estimator_from_registry(
                registry_dir, name=latest_estimator
            )
            metadata = {} if metadata is None else dict(metadata)
            metadata.setdefault("name", latest_estimator)
            metadata.setdefault(
                "training_type", "sim" if latest_estimator.endswith("_sim") else "real"
            )
            if registry_dir == self.dqn_registry_dir:
                self._cached_q_estimator = estimator
                self._cached_q_estimator_meta = dict(metadata)
            return estimator, metadata

        # Case 3 - No estimators available, create a new one
        rl_config = self.config.control.rl_config
        state_columns = get_full_state_signals_from_rl_config(rl_config)
        estimator = DDQNPREstimator(
            state_size=len(state_columns),
            action_size=rl_config.action_size,
            sequence_len=rl_config.history_length,
            n_controllers=self.config.control.n_controllers,
            # NOTE: Other entries will be overwritten at fit time
        )
        metadata = {"name": None, "training_type": None}
        if registry_dir == self.dqn_registry_dir:
            self._cached_q_estimator = estimator
            self._cached_q_estimator_meta = dict(metadata)
        return estimator, metadata

    def get_q_factors(
        self, df: pd.DataFrame | None = None, use_lite_inference: bool = False
    ) -> list[np.ndarray]:
        """
        Get the Q factors for the agent.
        Args:
            df (pd.DataFrame | None): The data to use for the Q factors. Defaults to None.
            use_lite_inference (bool): Whether to use lite inference. Defaults to True.
        Returns:
            np.ndarray: The Q factors for the agent.
        """
        rl_config = self.config.control.rl_config
        rl_seq_len = rl_config.history_length
        if df is None:
            df = self.get_rl_data_for_timestamp()

        state_columns = get_full_state_signals_from_rl_config(rl_config)
        q_estimator, _ = self.get_q_estimator(self.dqn_registry_dir)
        states = np.array(convert_data_to_state(df, state_columns, rl_seq_len))

        if use_lite_inference and states.shape[0] != 1:
            raise ValueError(
                "Lite inference only supports batch size 1; use get_q_factors_for_action() for control selection."
            )

        q_factors = (
            q_estimator.lite_predict(states)
            if use_lite_inference
            else q_estimator.forward_pass(states, lite_model=False)
        )

        return q_factors

    def get_q_factors_for_action(
        self, df: pd.DataFrame | None = None, *, use_lite_inference: bool = False
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        rl_config = self.config.control.rl_config
        rl_seq_len = rl_config.history_length
        if df is None:
            df = self.get_rl_data_for_timestamp()

        state_columns = get_full_state_signals_from_rl_config(rl_config)
        q_estimator, q_estimator_meta = self.get_q_estimator(self.dqn_registry_dir)
        model_name = q_estimator_meta.get("name")
        model_training_type = q_estimator_meta.get("training_type")

        # Fast path: only build the last sequence needed for action selection.
        state_frame = df[state_columns].tail(rl_seq_len)
        if len(state_frame) < rl_seq_len:
            raise ValueError(
                "Insufficient data to build an RL state sequence "
                f"(need at least history_length={rl_seq_len} rows)."
            )
        if state_frame.isna().any().any():
            raise ValueError(
                "NaN or NA values found in the RL state columns used for action selection. "
                f"Bad columns: {state_frame.columns[state_frame.isna().any()].tolist()}"
            )
        states_last = state_frame.to_numpy(dtype=np.float32)[None, :, :]  # (1, seq_len, state_size)

        inference_backend = "tensorflow"
        if use_lite_inference:
            try:
                q_factors = q_estimator.lite_predict(states_last)
                inference_backend = "tflite"
            except Exception:
                logger = logging.getLogger(__name__)
                logger.exception(
                    {
                        LOG_SIM_T_KEY: getattr(getattr(self, "time_info", None), "t", None),
                        LOG_ENTITY_KEY: f"Agent({self.uid})",
                        LOG_METHOD_KEY: "get_q_factors_for_action",
                        LOG_MESSAGE_KEY: "Lite inference failed; falling back to TensorFlow model.",
                    }
                )
                q_factors = q_estimator.forward_pass(states_last, lite_model=False)
        else:
            q_factors = q_estimator.forward_pass(states_last, lite_model=False)

        provenance: dict[str, Any] = {
            MODEL_NAME_KEY: model_name,
            MODEL_TRAINING_TYPE_KEY: model_training_type,
            INFERENCE_BACKEND_KEY: inference_backend,
        }
        return q_factors, provenance

    def get_replay_buffer(
        self, simulation: bool = False, registry_dir: str | None = None
    ) -> tuple[ReplayBuffer, dict]:
        """
        Get the replay buffer for the agent.

        Args:
            simulation (bool): Whether to get the simulated replay buffer. Defaults to False.
            registry_dir (str | None): The directory for the replay buffer registry. Defaults to None.
        Returns:
            tuple[ReplayBuffer, dict]: The replay buffer for the agent and its metadata.
        """
        # Define variables related to buffer
        registry_dir = (
            self.buffer_registry_dir if registry_dir is None else registry_dir
        )
        buffer_name = self.uid + "_sim" if simulation else self.uid + "_real"

        # Try to retrieve the replay buffer from the registry if it exists
        available_buffers = get_entities_in_registry(registry_dir)
        if buffer_name in available_buffers:
            buffer, buffer_metadata = load_replay_buffer_from_registry(
                registry_dir, name=buffer_name
            )

        # Otherwise, initialize a new replay buffer
        else:
            buffer_len = (
                self.config.control.sim_replay_buffer_size
                if simulation
                else self.config.control.real_replay_buffer_size
            )
            buffer = ReplayBuffer(
                max_len=buffer_len,
                prioritized_replay_alpha=0.6,
            )
            buffer_metadata = {
                "capacity": buffer_len,
                "n_experiences": 0,
            }
        return buffer, buffer_metadata

    def get_uid(self) -> UidType:
        """
        Get the unique identifier of the agent.
        Returns:
            UidType: The unique identifier of the agent.
        """
        return self.uid

    def push_in_memory_data_to_db(self, storage_path: str) -> None:
        # Get the data from the in-memory storage
        df = self.get_in_memory_data()

        # Make the timestamp column a datetime index
        df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY])
        df.set_index(TIMESTAMP_KEY, inplace=True)

        # Augment the data
        df = add_vm_data_to_df(df, self.cpm)
        df = add_tariff_data_to_df(df, self.config.tariff)
        df = add_product_data_to_df(df, self.config.product)
        df = self._coerce_agent_data_schema(tf_all_cyclic(df))

        # Convert the index to a column
        df.reset_index(inplace=True)

        # Save to local storage
        push_df_as_partitionned_parquet(df, storage_path)

    def push_data_to_replay_buffer(
        self,
        data: pd.DataFrame | list[pd.DataFrame],
        registry_dir: str,
        simulation: bool = False,
    ):
        """
        Push data to the replay buffer.
        Args:
            data (pd.DataFrame | list[pd.DataFrame]): The data to push to the replay buffer.
            registry_dir (str): The directory for the replay buffer registry.
            simulation (bool): Whether to push simulated data. Defaults to False.
        """
        # Sanitization and standardization
        if not isinstance(data, list):
            data = [data]

        # Get agent's replay buffer (real or simulated)
        replay_buffer, replay_buffer_metadata = self.get_replay_buffer(
            simulation=simulation, registry_dir=registry_dir
        )

        # Define important quantities to transform data into experience
        rl_config = self.config.control.rl_config
        state_columns = rl_config.state_signals
        state_columns = get_full_state_signals_from_rl_config(rl_config)
        control_columns = [
            CONTROL_KEY + f"_{i+1}" for i in range(rl_config.n_controllers)
        ]
        seq_len = rl_config.history_length
        product = AvailableProductsEnum.from_string(self.config.product)
        reward_columns = product.get_reward_names()

        # Convert data to RL experience
        for df in data:
            experience_batch = convert_data_to_experience(
                df, seq_len, state_columns, control_columns, reward_columns
            )

            # Add the experience samples from the dataframe to the replay buffer
            for experience in zip(*experience_batch):
                replay_buffer.add_experience_sample(experience=experience)

        # Save the replay buffer to the registry
        buffer_name = self.uid + "_sim" if simulation else self.uid + "_real"
        replay_buffer_metadata["n_experiences"] = len(replay_buffer)
        push_replay_buffer_to_registry(
            registry_dir=registry_dir,
            name=buffer_name,
            replay_buffer=replay_buffer,
            metadata=replay_buffer_metadata,
            overwrite=True,
        )

    def rl_training(self) -> None:
        training_started_at = dt.datetime.utcnow()
        logger = logging.getLogger(__name__)
        logger.info(
            {
                LOG_SIM_T_KEY: self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "rl_training",
                LOG_MESSAGE_KEY: "Starting real-data RL training",
            }
        )

        # Define start time based on previous trainings (all if first time)
        start_time = None
        if hasattr(self, "last_rl_training_end"):
            start_time = self.last_rl_training_end

        # Get data, and keep track of last index to avoid future overlaps
        history = self.get_data(start_time=start_time)
        self.last_rl_training_end = history.index[-1]

        # Update scaler with dataframe
        # self.data_module.update_scaler_info_from_df(self.uid, df=history)

        rl_config = self.config.control.rl_config
        state_columns = get_full_state_signals_from_rl_config(rl_config)
        control_columns = [
            CONTROL_KEY + f"_{i+1}" for i in range(rl_config.n_controllers)
        ]
        product = AvailableProductsEnum.from_string(self.config.product)
        reward_columns = product.get_reward_names()
        required_cols = list(
            dict.fromkeys([*state_columns, *control_columns, *reward_columns, DONE_KEY])
        )

        missing_cols = [c for c in required_cols if c not in history.columns]
        if missing_cols:
            raise ValueError(
                "Training data missing required RL columns "
                f"(missing={missing_cols})."
            )

        # Only validate columns used by the RL pipeline; metadata columns may legitimately be null.
        if history[required_cols].isna().any().any():
            bad_rows = history[required_cols].isna().any(axis=1)
            sample = history.loc[bad_rows, required_cols].head(10)
            raise ValueError(
                "NaN values detected in training data for required RL columns. "
                f"Sample:\n{sample}"
            )

        # Add new data in the replay buffer
        self.push_data_to_replay_buffer(
            data=history, registry_dir=self.buffer_registry_dir
        )

        # Retrieve buffer and q-estimator from registry
        buffer, buffer_metadata = self.get_replay_buffer(
            registry_dir=self.buffer_registry_dir
        )
        q_estimator, _ = self.get_q_estimator(registry_dir=self.dqn_registry_dir)

        # Training loop (config-driven)
        real_lr_config: RealLearningConfig = get_active_config_based_on_duration(
            duration_s=self.get_elapsed_time(),
            config_dict=self.config.control.real_learning_configs,
        )
        train_cfg = real_lr_config.rl_training_config
        for _ in range(int(train_cfg.n_target_iterators)):
            q_estimator, buffer, _ = simple_training_loop(
                replay_buffer=buffer,
                q_estimator=q_estimator,
                n_sampling_iters=int(train_cfg.n_sampling_iters),
                sampling_size=int(train_cfg.experience_sampling_size),
                learning_rate=float(train_cfg.learning_rate),
                tf_n_fit_epochs=int(train_cfg.n_fit_epochs),
                tf_batch_size=int(train_cfg.tf_batch_size),
            )
            q_estimator.update_target_model()

        # Save new Q-estimator to registry
        estimator_name = self.uid + "_" + self.time_info.t.strftime(DT_FILE_STR_FORMAT)
        estimator_metadata = {"last_training": self.time_info.get_t_as_str()}
        push_dqn_estimator_to_registry(
            registry_dir=self.dqn_registry_dir,
            name=estimator_name,
            estimator=q_estimator,
            metadata=estimator_metadata,
        )
        self._cached_q_estimator = q_estimator
        self._cached_q_estimator_meta = {
            "name": estimator_name,
            "training_type": "real",
        }

        # Save the replay buffer to the registry
        buffer_name = self.uid + "_real"
        buffer_metadata["n_experiences"] = len(buffer)
        push_replay_buffer_to_registry(
            registry_dir=self.buffer_registry_dir,
            name=buffer_name,
            replay_buffer=buffer,
            metadata=buffer_metadata,
            overwrite=True,
        )

        training_ended_at = dt.datetime.utcnow()
        summary = self._load_training_summary()
        summary["n_real_trainings"] = int(summary.get("n_real_trainings", 0)) + 1
        summary["last_real_training_time"] = self.time_info.get_t_as_str()
        summary["last_real_model_name"] = estimator_name
        summary["last_real_buffer_n_experiences"] = int(buffer_metadata.get("n_experiences", len(buffer)))
        summary["last_real_training_duration_s"] = (
            training_ended_at - training_started_at
        ).total_seconds()
        self._save_training_summary(summary)

        logger.info(
            {
                LOG_SIM_T_KEY: self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "rl_training",
                LOG_MESSAGE_KEY: (
                    "Completed real-data RL training "
                    f"(model_name={estimator_name}, duration_s={summary['last_real_training_duration_s']})"
                ),
            }
        )

        del buffer, q_estimator, estimator_metadata, history

    def simulate_trajectory_at_time(
        self,
        timestamp: dt.datetime,
        sim_len: int = 18,
        policy: str = "random_policy",
        policy_kwargs: dict = None,
        timestep_s: int = 300,
    ) -> pd.DataFrame:
        """
        Simulate a trajectory at a specific time step.
        Args:
            timestamp (dt.datetime): The time step to simulate.
            sim_len (int): The length of the simulation. Defaults to 12.
            policy (str): The policy to use for control selection. Defaults to "random_policy".
            policy_kwargs (dict): Additional arguments for the policy. Defaults to None.
            timestep_s (int): The time step in seconds. Defaults to 300.
        Returns:
            pd.DataFrame: The simulated trajectory.
        """
        # Initial variables definition
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        df = self.get_data(start_time=timestamp).iloc[:sim_len]
        history_len = self.config.control.rl_config.history_length

        # Initialize trajectory
        traj = Trajectory.partial_from_agent_df(agent_config=self.config, df=df)
        t_sim = timestamp + dt.timedelta(seconds=timestep_s * history_len)

        # Loop over the simulation length
        for _ in range(sim_len - history_len):
            # Get controls
            traj_df = traj.as_df()
            rl_df = traj_df[traj_df.index < t_sim]
            controls = self.get_control(
                policy=policy, policy_kwargs=policy_kwargs, df=rl_df
            )
            epsilon = None
            if policy in {"q_policy", "hvac_policy"}:
                epsilon = float(policy_kwargs.get("epsilon", 0.0))
            control_record = {
                **{f"control_{i+1}": controls[i] for i in range(len(controls))},
                CONTROL_SELECTION_ENABLED_KEY: True,
                POLICY_KEY: policy,
                EPSILON_KEY: epsilon,
                MODEL_NAME_KEY: self._last_control_provenance.get(MODEL_NAME_KEY),
                MODEL_TRAINING_TYPE_KEY: self._last_control_provenance.get(
                    MODEL_TRAINING_TYPE_KEY
                ),
                INFERENCE_BACKEND_KEY: self._last_control_provenance.get(
                    INFERENCE_BACKEND_KEY
                ),
            }
            traj.add_control_record(t_sim, control_record)

            # Estimate new state
            prev_t = t_sim - dt.timedelta(seconds=timestep_s)
            new_state_record = self.get_state_prediction(prev_t=prev_t, df=rl_df)
            traj.add_state_record(t_sim, new_state_record)

            # Advance trajectory simulation time
            t_sim += dt.timedelta(seconds=timestep_s)

        # Finally, add computed columns to the trajectory
        final_df = traj.as_df()
        final_df_augmented = self.asset.augment_df(final_df)
        final_df_augmented = add_vm_data_to_df(final_df_augmented, self.cpm)
        final_df_augmented = add_tariff_data_to_df(
            final_df_augmented, self.config.tariff
        )
        final_df_augmented = add_product_data_to_df(
            final_df_augmented, self.config.product
        )
        final_df_augmented = tf_all_cyclic(final_df_augmented)
        return final_df_augmented

    def simulated_rl_training(
        self,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        n_samples: int | None = 100,
        n_traj_per_sample: int = 1,
        traj_len: int = 18,
        policy: str = "q_policy",
        policy_kwargs: dict = {"epsilon": 0.5},
    ) -> None:
        training_started_at = dt.datetime.utcnow()
        logger = logging.getLogger(__name__)
        logger.info(
            {
                LOG_SIM_T_KEY: self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "simulated_rl_training",
                LOG_MESSAGE_KEY: (
                    "Starting simulated RL training "
                    f"(n_samples={n_samples}, n_traj_per_sample={n_traj_per_sample}, traj_len={traj_len}, policy={policy})"
                ),
            }
        )

        # Download necessary data
        history_len = self.config.control.rl_config.history_length
        df = self.get_data(start_time=start_time, end_time=end_time)
        if df.empty:
            logger.warning(
                {
                    LOG_SIM_T_KEY: self.time_info.t,
                    LOG_ENTITY_KEY: f"Agent({self.uid})",
                    LOG_METHOD_KEY: "simulated_rl_training",
                    LOG_MESSAGE_KEY: "Skipping simulated RL training: no data available.",
                }
            )
            return

        eligible = df.index.to_pydatetime()[history_len : -(history_len + traj_len)]
        if len(eligible) == 0:
            logger.warning(
                {
                    LOG_SIM_T_KEY: self.time_info.t,
                    LOG_ENTITY_KEY: f"Agent({self.uid})",
                    LOG_METHOD_KEY: "simulated_rl_training",
                    LOG_MESSAGE_KEY: (
                        "Skipping simulated RL training: insufficient data for sampling "
                        f"(history_len={history_len}, traj_len={traj_len}, n_rows={len(df)})."
                    ),
                }
            )
            return

        if n_samples is None:
            samples_batch = eligible
        else:
            n_samples_effective = min(int(n_samples), len(eligible))
            if n_samples_effective <= 0:
                logger.warning(
                    {
                        LOG_SIM_T_KEY: self.time_info.t,
                        LOG_ENTITY_KEY: f"Agent({self.uid})",
                        LOG_METHOD_KEY: "simulated_rl_training",
                        LOG_MESSAGE_KEY: "Skipping simulated RL training: n_samples_effective <= 0.",
                    }
                )
                return
            samples_batch = np.random.choice(eligible, n_samples_effective, replace=False)

        # Loop and generate trajectory samples, saving them in buffer
        rl_config = self.config.control.rl_config
        state_columns = get_full_state_signals_from_rl_config(rl_config)
        control_columns = [
            CONTROL_KEY + f"_{i+1}" for i in range(rl_config.n_controllers)
        ]
        product = AvailableProductsEnum.from_string(self.config.product)
        reward_columns = product.get_reward_names()
        required_cols = list(dict.fromkeys([*state_columns, *control_columns, *reward_columns, DONE_KEY]))

        trajectories_list = []
        for t in samples_batch:
            for _ in range(n_traj_per_sample):
                # Simulate trajectory
                traj_df = self.simulate_trajectory_at_time(
                    timestamp=t,
                    sim_len=traj_len,
                    policy=policy,
                    policy_kwargs=policy_kwargs,
                )

                missing_cols = [c for c in required_cols if c not in traj_df.columns]
                if missing_cols:
                    raise ValueError(
                        "Simulated trajectory missing required RL columns "
                        f"(missing={missing_cols})."
                    )

                # Raise error if NaN values in required RL columns
                if traj_df[required_cols].isna().any().any():
                    debug_cols = required_cols
                    for extra in (POLICY_KEY, CONTROL_SELECTION_ENABLED_KEY):
                        if extra in traj_df.columns and extra not in debug_cols:
                            debug_cols = [*debug_cols, extra]
                    raise ValueError(
                        f"NaN values detected in simulated trajectory at time {t} "
                        f"for required_cols={required_cols}. \n {traj_df[debug_cols]}"
                    )

                # Add trajectory to the list
                trajectories_list.append(traj_df)

        # Add trajectories to the replay buffer
        self.push_data_to_replay_buffer(
            data=trajectories_list,
            registry_dir=self.buffer_registry_dir,
            simulation=True,
        )

        # Add additional informative columns to each df, and concatenate
        for i, sim_df in enumerate(trajectories_list):
            # Add trajectory counter
            sim_df["traj_counter"] = i + 1
            sim_df["is_real_data"] = False
            sim_df.loc[sim_df.index[:history_len], "is_real_data"] = True

        trajectories_list = [
            traj_df.assign(traj_counter=i + 1)
            for i, traj_df in enumerate(trajectories_list)
        ]
        all_sim_df = pd.concat(trajectories_list, axis=0)
        all_sim_df.reset_index(inplace=True)
        all_sim_df["training_time"] = self.time_info.t.date()

        # Save to local storage
        push_df_as_partitionned_parquet(
            all_sim_df, self.sim_data_dir, partition_cols=["training_time"]
        )

        # Retrieve buffer and q-estimator from registry
        buffer, buffer_metadata = self.get_replay_buffer(
            simulation=True, registry_dir=self.buffer_registry_dir
        )
        q_estimator, estimator_metadata = self.get_q_estimator(
            registry_dir=self.dqn_registry_dir
        )

        # Training loop (config-driven)
        sim_lr_config: SimLearningConfig = get_active_config_based_on_duration(
            duration_s=self.get_elapsed_time(),
            config_dict=self.config.control.sim_learning_configs,
        )
        train_cfg = sim_lr_config.rl_training_config
        for _ in range(int(train_cfg.n_target_iterators)):
            q_estimator, buffer, _ = simple_training_loop(
                replay_buffer=buffer,
                q_estimator=q_estimator,
                n_sampling_iters=int(train_cfg.n_sampling_iters),
                sampling_size=int(train_cfg.experience_sampling_size),
                learning_rate=float(train_cfg.learning_rate),
                tf_n_fit_epochs=int(train_cfg.n_fit_epochs),
                tf_batch_size=int(train_cfg.tf_batch_size),
            )
            q_estimator.update_target_model()

        # Save new Q-estimator to registry
        estimator_name = (
            self.uid + "_" + self.time_info.t.strftime(DT_FILE_STR_FORMAT) + "_sim"
        )
        estimator_metadata["last_sim_training"] = self.time_info.get_t_as_str()
        push_dqn_estimator_to_registry(
            registry_dir=self.dqn_registry_dir,
            name=estimator_name,
            estimator=q_estimator,
            metadata=estimator_metadata,
        )
        self._cached_q_estimator = q_estimator
        self._cached_q_estimator_meta = {
            "name": estimator_name,
            "training_type": "sim",
        }

        # Save the replay buffer to the registry
        buffer_name = self.uid + "_sim"
        buffer_metadata["n_experiences"] = len(buffer)
        push_replay_buffer_to_registry(
            registry_dir=self.buffer_registry_dir,
            name=buffer_name,
            replay_buffer=buffer,
            metadata=buffer_metadata,
            overwrite=True,
        )

        training_ended_at = dt.datetime.utcnow()
        summary = self._load_training_summary()
        summary["n_sim_trainings"] = int(summary.get("n_sim_trainings", 0)) + 1
        summary["last_sim_training_time"] = self.time_info.get_t_as_str()
        summary["last_sim_model_name"] = estimator_name
        summary["last_sim_buffer_n_experiences"] = int(buffer_metadata.get("n_experiences", len(buffer)))
        summary["last_sim_training_duration_s"] = (
            training_ended_at - training_started_at
        ).total_seconds()
        self._save_training_summary(summary)

        logger.info(
            {
                LOG_SIM_T_KEY: self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "simulated_rl_training",
                LOG_MESSAGE_KEY: (
                    "Completed simulated RL training "
                    f"(model_name={estimator_name}, duration_s={summary['last_sim_training_duration_s']})"
                ),
            }
        )

        del buffer, q_estimator, estimator_metadata, traj_df

    def to_file(self, directory: str = "") -> None:
        """
        Save the agent's state to a file using dill.
        Args:
            directory (str): The directory to save the file in.
        """
        filepath = os.path.join(directory, "agent.pkl")
        internal_dict = vars(self)
        with open(filepath, "wb") as f:
            dill.dump(internal_dict, f)

    def update_time_info(self, time_info: TimeInfo) -> None:
        """
        Update the time information of the agent.
        Args:
            time_info (TimeInfo): The time information to update.
        """
        self.time_info = time_info
        self._generate_shortcuts()

    def update_environment_info(self, *, oat: float | None = None) -> None:
        self.oat = oat

    def _get_training_summary_path(self) -> str:
        return os.path.join(self.directory, "training_summary.json")

    def _load_training_summary(self) -> dict[str, Any]:
        filepath = self._get_training_summary_path()
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "r") as f:
            return json.load(f)

    def _save_training_summary(self, summary: dict[str, Any]) -> None:
        filepath = self._get_training_summary_path()
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True, default=str)

    def save_config(self, filepath: str) -> None:
        """
        Save the agent configuration to a JSON file.
        Args:
            filepath (str): The path to the JSON file.
        """
        # Dump agent configuration to JSON file
        with open(filepath, "w") as f:
            config_json = self.config.model_dump()
            json.dump(config_json, f, indent=4)

    def _generate_shortcuts(self):
        if hasattr(self, "config"):
            self.cpm = self.config.data.control_power_mapping
        if hasattr(self, "time_info"):
            self.t = self.time_info.t

    def _push_data_dict_to_memory_storage(
        self,
        storage_key: str,
        data_dict: dict[str, list[str | float | int]],
        timestamp: dt.datetime = None,
    ) -> None:
        """
        Push data to the in-memory storage.
        Args:
            storage_key (str): The key to store the data under.
            data_dict (dict[str, list]): The data to push.
            timestamp (dt.datetime | None): The timestamp to associate with the data. Defaults to None. Defaults to True.
        """
        # Create a copy of the data dictionary
        data_dict = copy(data_dict)

        # Add current timestamp to the data dictionary if required
        if timestamp is not None:
            data_dict[TIMESTAMP_KEY] = timestamp

        # Append the data to the in-memory storage
        self._memory_storage[storage_key].append(data_dict)
