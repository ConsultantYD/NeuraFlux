from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from neuraflux.agency.agent import Agent
from neuraflux.assets.factory import AvailableAssetsEnum
from neuraflux.geography import CityEnum
from neuraflux.global_variables import (
    CONTROL_KEY,
    DT_STR_FORMAT,
    OAT_KEY,
)
from neuraflux.local_typing import AssetType
from neuraflux.schemas.agency import AgentConfig
from neuraflux.schemas.simulation import SimulationConfig
from neuraflux.time_ref import TimeRef
from neuraflux.weather import Weather
from neuraflux.agency.utils_data import cron_matches
from neuraflux.logging_utils import StructuredLogHandler
from neuraflux.global_variables import DT_FILE_STR_FORMAT
from neuraflux.global_variables import WEATHER_DB_NAME


class Simulation:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        *,
        overwrite: bool = False,
        make_unique: bool = True,
        log_level: int | str = "INFO",
        structured_logging: bool = True,
    ) -> None:
        # Initialize key internal attributes
        self.config = simulation_config
        self.directory = self._prepare_output_directory(
            simulation_config.directory,
            overwrite=overwrite,
            make_unique=make_unique,
        )
        self.config.directory = self.directory
        os.makedirs(self.directory, exist_ok=True)

        # Configure logging early (so init errors are recorded).
        self._configure_logging(
            directory=self.directory,
            log_level=log_level,
            structured_logging=structured_logging,
        )

        # Fix seeds for reproducibility
        self._fix_seeds(self.config.seed)

        # ---------------------------------------------------
        # - ENVIRONMENT
        # ---------------------------------------------------
        # Time-related components initialization
        self.sim_start_time = self._parse_datetime(self.config.time.start_time)
        self.sim_end_time = self._parse_datetime(self.config.time.end_time)
        if self.sim_end_time <= self.sim_start_time:
            raise ValueError(
                f"Invalid simulation time range: end_time ({self.sim_end_time}) must be after start_time ({self.sim_start_time})."
            )
        self.time_ref = self._initialize_time_reference(
            start_time=self.sim_start_time, step_size_s=self.config.time.step_size_s
        )
        self.time_info = self.time_ref.get_time_info()
        self.t = self.time_info.t

        # Weather
        self._maybe_copy_weather_db()
        self.weather_ref = self._initialize_weather(
            city=self.config.geography.city,
            db_dir=self.directory,
            start_date=self.sim_start_time - dt.timedelta(days=1),
            end_date=self.sim_end_time + dt.timedelta(days=1),
        )
        self.weather_info = self.weather_ref.get_weather_info_at_time(self.t)
        self.oat = self.weather_info.temperature

        # ---------------------------------------------------
        # - ASSETS
        # ---------------------------------------------------
        # Initialize assets controlled by agents
        self.assets = self._initialize_assets(
            t=self.t,
            oat=self.weather_info.temperature,
            assets_configs_dict=self.config.assets,
        )

        # Initialize comparative 'shadow' assets (copy of real assets)
        self.shadow_assets = self._initialize_shadow_assets(
            real_assets_dict=self.assets,
        )

        # ---------------------------------------------------
        # - AGENTS AND RELATED COMPONENTS
        # ---------------------------------------------------
        # Initialize agents
        self.agents = self._initialize_agents(
            agent_configs_dict=self.config.agents,
            directory=self.directory,
            time_info=self.time_info,
            assets=self.assets,
            shadow_assets=self.shadow_assets,
        )

    def run(self) -> dict[str, Any]:
        logger = logging.getLogger(__name__)

        started_at_utc = dt.datetime.utcnow()
        self.sim_summary: dict[str, Any] = self._build_sim_summary(
            status="running",
            started_at_utc=started_at_utc,
        )
        self._write_json_file(
            os.path.join(self.directory, "config.json"),
            self.config.model_dump(mode="json"),
        )
        self._write_json_file(
            os.path.join(self.directory, "sim_summary.json"),
            self.sim_summary,
        )

        status = "completed"
        try:
            # Main simulation loop
            while self.time_info.t < self.sim_end_time:
                # ---------------------------------------------------
                # - AGENTS EXECUTION
                # ---------------------------------------------------
                agents_controls: dict[str, Any] = {}
                for uid, agent in self.agents.items():
                    agent.update_time_info(self.time_info)
                    agent.update_environment_info(oat=self.oat)
                    agents_controls[uid] = agent.run()

                # Increment simulation by one time step
                # NOTE: Increment is done here to allow agents to run on initial state
                self.time_ref.increment_time()

                # Update time and weather info for simulation
                self.time_info = self.time_ref.get_time_info()
                self.t = self.time_info.t
                self.weather_info = self.weather_ref.get_weather_info_at_time(self.t)
                self.oat = self.weather_info.temperature

                # ---------------------------------------------------
                # - ASSETS EXECUTION
                # ---------------------------------------------------
                for uid in self.agents.keys():
                    asset = self.assets[uid]
                    shadow_asset = self.shadow_assets[uid]

                    # Advance asset simulation, using agent control if available
                    if agents_controls[uid] is not None:
                        asset.step(
                            agents_controls[uid],
                            self.t,
                            self.oat,
                        )
                    else:
                        asset.auto_step(self.t, self.oat)

                    # Keep shadow asset in sync with the real asset for comparison
                    shadow_control = shadow_asset.get_auto_control(self.t, self.oat)
                    shadow_asset.step(shadow_control, self.t, self.oat)

                # Save agent state snapshots when requested
                if cron_matches(self.time_info.t, self.config.agent_save_freq_cron):
                    for uid, agent in self.agents.items():
                        agent_directory = os.path.join(self.directory, uid)
                        agent.to_file(directory=agent_directory)

        except Exception:
            status = "failed"
            self.sim_summary["error"] = {
                "type": "exception",
                "message": traceback.format_exc().splitlines()[-1],
            }
            self.sim_summary["traceback"] = traceback.format_exc()
            logger.exception("Simulation failed")
            raise
        finally:
            ended_at_utc = dt.datetime.utcnow()

            # Ensure all remaining in-memory agent data is flushed to disk
            for uid, agent in getattr(self, "agents", {}).items():
                agent_directory = os.path.join(self.directory, uid)
                self._finalize_agent_artifacts(
                    uid=uid, agent=agent, agent_directory=agent_directory
                )

            # Persist simulation-level artifacts
            try:
                self.time_ref.to_file(self.directory)
            except Exception:
                logger.exception("Failed to write time_ref.json")

            self.sim_summary.update(
                {
                    "status": status,
                    "ended_at_utc": ended_at_utc.strftime(DT_STR_FORMAT),
                    "duration_s": (ended_at_utc - started_at_utc).total_seconds(),
                    "final_sim_time": self.time_info.t.strftime(DT_STR_FORMAT)
                    if hasattr(self, "time_info")
                    else None,
                }
            )
            self._write_json_file(
                os.path.join(self.directory, "sim_summary.json"),
                self.sim_summary,
            )

            # Detach and close the structured logging handler to avoid leaking handlers
            # across multiple Simulation runs in the same Python process.
            if getattr(self, "_structured_log_handler", None) is not None:
                try:
                    root_logger = logging.getLogger()
                    root_logger.removeHandler(self._structured_log_handler)
                    self._structured_log_handler.close()
                except Exception:
                    logger.exception("Failed to close structured logging handler")

        return self.sim_summary

    def _build_sim_summary(
        self, status: str, started_at_utc: dt.datetime
    ) -> dict[str, Any]:
        weather_db_path = os.path.join(self.directory, WEATHER_DB_NAME)
        weather_db_sha256 = (
            self._compute_file_sha256(weather_db_path)
            if os.path.exists(weather_db_path)
            else None
        )

        agents_summary: dict[str, Any] = {}
        for uid, agent_cfg in self.config.agents.items():
            asset_cfg = self.config.assets.get(uid)
            agents_summary[uid] = {
                "asset_type": getattr(asset_cfg, "asset_type", None),
                "tariff": getattr(agent_cfg, "tariff", None),
                "product": getattr(agent_cfg, "product", None),
                "training_summary": f"{uid}/training_summary.json",
            }

        return {
            "status": status,
            "started_at_utc": started_at_utc.strftime(DT_STR_FORMAT),
            "directory": self.directory,
            "seed": self.config.seed,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": getattr(np, "__version__", None),
                "tensorflow": getattr(tf, "__version__", None),
            },
            "geography": {"city": self.config.geography.city},
            "time": {
                "start_time": self.config.time.start_time,
                "end_time": self.config.time.end_time,
                "step_size_s": self.config.time.step_size_s,
            },
            "weather": {
                "db": WEATHER_DB_NAME,
                "db_sha256": weather_db_sha256,
                "source": getattr(getattr(self.config, "data", None), "weather_db_source", None),
                "preloaded": getattr(self.weather_ref, "preloaded", None),
            },
            "agent_save_freq_cron": self.config.agent_save_freq_cron,
            "agents": agents_summary,
            "artifacts": {
                "config": "config.json",
                "summary": "sim_summary.json",
                "time_ref": "time_ref.json",
                "logs_db": "logs.db",
                "weather_db": "weather.db",
            },
        }

    def _write_json_file(self, filepath: str, data: Any) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)

    def _finalize_agent_artifacts(
        self, uid: str, agent: Agent, agent_directory: str
    ) -> None:
        logger = logging.getLogger(__name__)
        os.makedirs(agent_directory, exist_ok=True)

        # Flush in-memory agent data to disk (parquet)
        try:
            memory_df = agent.get_in_memory_data()
            if hasattr(memory_df, "empty") and not memory_df.empty:
                agent.push_in_memory_data_to_db(agent.data_dir)
                agent.clear_in_memory_storage()
        except Exception:
            logger.exception("Failed to flush in-memory data for agent %s", uid)

        # Persist agent (includes asset references) for dashboard consumption
        try:
            agent.to_file(directory=agent_directory)
        except Exception:
            logger.exception("Failed to write agent.pkl for agent %s", uid)

        # Persist assets as standalone artifacts (optional redundancy)
        try:
            self.assets[uid].to_file(directory=agent_directory)
        except Exception:
            logger.exception("Failed to write asset pickle for agent %s", uid)
        try:
            self.shadow_assets[uid].to_file(directory=agent_directory)
        except Exception:
            logger.exception("Failed to write shadow asset pickle for agent %s", uid)

    def _parse_datetime(self, value: str | dt.datetime) -> dt.datetime:
        if isinstance(value, dt.datetime):
            parsed = value
        elif isinstance(value, str):
            v = value.strip()
            # Support common ISO 8601 variants
            try:
                parsed = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            if parsed is None:
                for fmt in (DT_STR_FORMAT, DT_FILE_STR_FORMAT):
                    try:
                        parsed = dt.datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
            if parsed is None:
                raise ValueError(
                    f"Invalid datetime string '{value}'. Expected ISO 8601 like '{DT_STR_FORMAT}' or filesystem format '{DT_FILE_STR_FORMAT}'."
                )
        else:
            raise TypeError(f"Invalid datetime value type: {type(value)}")

        # Normalize timezone-aware datetimes to naive UTC
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return parsed

    def _prepare_output_directory(
        self, directory: str, *, overwrite: bool, make_unique: bool
    ) -> str:
        if not directory:
            raise ValueError("SimulationConfig.directory must be a non-empty path.")

        base = Path(directory)
        if base.exists() and not base.is_dir():
            raise ValueError(
                f"SimulationConfig.directory must be a directory path; found file: {directory}"
            )

        # Safe default: never delete existing results unless explicitly requested.
        if base.is_dir() and base.exists():
            try:
                has_contents = any(base.iterdir())
            except OSError:
                has_contents = True
            if has_contents:
                if overwrite:
                    resolved = base.resolve()
                    if resolved == Path("/"):
                        raise ValueError(
                            "Refusing to overwrite the filesystem root directory."
                        )
                    shutil.rmtree(base)
                    return str(base)

                if not make_unique:
                    raise FileExistsError(
                        f"Simulation output directory already exists and is not empty: {directory}"
                    )

                run_id = dt.datetime.utcnow().strftime(DT_FILE_STR_FORMAT)
                candidate = base.parent / f"{base.name}__{run_id}"
                suffix = 1
                while candidate.exists():
                    candidate = base.parent / f"{base.name}__{run_id}_{suffix}"
                    suffix += 1
                return str(candidate)

        return str(base)

    def _configure_logging(
        self, *, directory: str, log_level: int | str, structured_logging: bool
    ) -> None:
        if isinstance(log_level, str):
            level = logging._nameToLevel.get(log_level.upper(), logging.INFO)
        else:
            level = log_level

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Add a console handler if none exists (helps local debugging).
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            console = logging.StreamHandler()
            console.setLevel(level)
            root_logger.addHandler(console)

        self._structured_log_handler = None
        if structured_logging:
            handler = StructuredLogHandler(db_dir=directory)
            handler.setLevel(level)
            root_logger.addHandler(handler)
            self._structured_log_handler = handler

    def _compute_file_sha256(self, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _maybe_copy_weather_db(self) -> None:
        logger = logging.getLogger(__name__)
        source = getattr(getattr(self.config, "data", None), "weather_db_source", None)
        if not source:
            return

        source_path = Path(source)
        if source_path.is_dir():
            source_file = source_path / WEATHER_DB_NAME
        else:
            source_file = source_path

        if not source_file.exists():
            raise FileNotFoundError(
                f"Weather DB source path does not exist: {source_file}"
            )

        dest_file = Path(self.directory) / WEATHER_DB_NAME
        if dest_file.resolve() == source_file.resolve():
            return

        if dest_file.exists():
            logger.warning(
                "weather.db already exists in output directory; refusing to overwrite (%s)",
                dest_file,
            )
            return

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, dest_file)

    def _fix_seeds(self, seed_value: int) -> None:
        """
        Fixes the random seed for reproducibility. Covers numpy, random, and TensorFlow.
        Args:
            seed_value (int): The seed value to set for random number generation.
        """
        # Best-effort determinism controls (may depend on TF build / hardware).
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)
        try:
            tf.keras.utils.set_random_seed(seed_value)
        except Exception:
            # Older TF/Keras combos may not expose this helper.
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Not supported on all TF versions/platforms.
            pass

    def _initialize_time_reference(
        self, start_time: dt.datetime, step_size_s: int
    ) -> TimeRef:
        """
        Initializes the time reference for the simulation. Sets the start time, end time, and time step size.
        Args:
            start_time (dt.datetime): The start time of the simulation.
            step_size_s (int): The time step size in seconds.
        Returns:
            TimeRef: An instance of the TimeRef class, which encapsulates time-related information.
        """
        return TimeRef(
            start_time_utc=start_time,
            def_time_step=dt.timedelta(seconds=step_size_s),
        )

    def _initialize_weather(
        self,
        city: CityEnum,
        db_dir: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> Weather:
        """
        Initializes the weather reference for the simulation.
        Args:
            city (CityEnum): The city for which the weather information is required.
            db_dir (str): The directory where the weather data will be stored.
        """
        return Weather(
            city=city, db_dir=db_dir, start_date=start_date, end_date=end_date
        )

    def _initialize_modules(self, directory: str):
        """
        Initializes the agency modules for the whole simulation.
        Args:
            directory (str): The directory where the modules will be stored.
        Returns:
            tuple[ControlModule, DataModule]: The control and data modules.
        """
        from neuraflux.agency.control_module import ControlModule
        from neuraflux.agency.data_module import DataModule

        control_module = ControlModule(base_dir=directory)
        data_module = DataModule(base_dir=directory)
        return control_module, data_module

    def _initialize_assets(
        self, t: dt.datetime, oat: float, assets_configs_dict: dict[str, object]
    ) -> dict[str, AssetType]:
        """
        Initializes the assets for the simulation based on the configuration provided.
        Args:
            t (dt.datetime): The current time of the simulation.
            oat (float): The outside air temperature, in DegC.
            assets_configs_dict (dict[str, object]): A dictionary containing the asset configurations.
        Returns:
            dict[str, AssetType]: A dictionary mapping asset UIDs to their respective asset instances.
        """
        # Loop over all inputed asset configs
        assets: dict[str, AssetType] = {}
        for asset_uid, asset_config_dict in assets_configs_dict.items():
            # Initial state definition
            asset_config_dict.initial_state_dict[OAT_KEY] = oat

            # Asset instances creation
            asset_type = asset_config_dict.asset_type
            AssetClass = AvailableAssetsEnum.get_asset_class_from_asset_name(asset_type)
            AssetConfigClass = (
                AvailableAssetsEnum.get_asset_config_class_from_asset_name(asset_type)
            )
            asset = AssetClass(
                asset_uid,
                AssetConfigClass.model_validate(asset_config_dict),
                t,
                oat,
            )
            assets[asset_uid] = asset

        return assets

    def _initialize_shadow_assets(
        self, real_assets_dict: dict[str, AssetType]
    ) -> dict[str, AssetType]:
        """
        Initializes shadow assets as an exact copy of the real assets.
        Args:
            real_assets_dict (dict[str, AssetType]): A dictionary mapping asset UIDs to their respective asset instances.
        Returns:
            dict[str, AssetType]: A dictionary mapping asset UIDs to their respective shadow asset instances.
        """
        # Initialize shadow assets as an exact copy of the real assets
        shadow_assets = {
            uid: deepcopy(asset) for uid, asset in real_assets_dict.items()
        }

        # Modify asset name to differentiate from real asset
        for asset in shadow_assets.values():
            asset.name = f"{asset.name}_shadow"

        return shadow_assets

    def _initialize_agents(
        self,
        directory: str,
        time_info: TimeRef,
        agent_configs_dict: dict[str, AgentConfig],
        assets: dict[str, AssetType],
        shadow_assets: dict[str, AssetType],
    ) -> dict[str, Agent]:
        """
        Initializes the agents for the simulation based on the configuration provided.
        Args:
            directory (str): The directory where the agents will be stored.
            agent_configs_dict (dict[str, AgentConfig]): A dictionary containing the agent configurations.
            assets (dict[str, AssetType]): A dictionary mapping asset UIDs to their respective asset instances.
            shadow_assets (dict[str, AssetType]): A dictionary mapping asset UIDs to their respective shadow asset instances.
        Returns:
            dict[str, Agent]: A dictionary mapping agent UIDs to their respective agent instances.
        """
        agents: dict[str, Agent] = {}

        # Loop over all input agent configs
        for uid, agent_config in agent_configs_dict.items():
            # Initialize agent directory
            agent_dir = os.path.join(directory, uid)
            os.makedirs(agent_dir, exist_ok=False)

            # Initialize agent instance
            agent = Agent(
                uid=uid,
                directory=agent_dir,
                time_info=time_info,
                config=agent_config,
                data_module=None,
                control_module=None,
            )

            # Save Agent config in directory
            config_filepath = os.path.join(agent_dir, "config.json")
            agent.save_config(config_filepath)

            # Assign Agent instance to its corresponding asset and shadow asset
            agent.assign_to_asset(assets[uid], shadow_assets[uid])

            # Add agent to simulation
            agents[uid] = agent
        return agents
