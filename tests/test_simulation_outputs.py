import datetime as dt
from pathlib import Path

import pytest

from neuraflux.geography import CityEnum
from neuraflux.global_variables import OAT_KEY
from neuraflux.schemas.agency import (
    AgentConfig,
    AgentControlConfig,
    AgentDataConfig,
    ControlSelectionConfig,
    RealLearningConfig,
    RLConfig,
    SignalTags,
    SimLearningConfig,
)
from neuraflux.schemas.asset_config import EnergyStorageConfig
from neuraflux.schemas.simulation import (
    SimulationConfig,
    SimulationGeographicalConfig,
    SimulationTimeConfig,
)
from neuraflux.simulation import Simulation


class _DummyWeather:
    def __init__(
        self,
        city,
        db_dir: str,
        start_date: dt.datetime | None = None,
        end_date: dt.datetime | None = None,
    ):
        self.city = city

    def get_weather_info_at_time(self, time: dt.datetime):
        return type("WeatherInfo", (), {"temperature": 10.0})()


def test_simulation_writes_summary_and_agent_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("neuraflux.simulation.Weather", _DummyWeather)

    signals_info = {
        "internal_energy": {
            "tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value],
            "min_value": 0,
            "max_value": 100,
            "scalable": True,
        },
        OAT_KEY: {
            "tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value],
            "min_value": -50,
            "max_value": 50,
            "scalable": True,
        },
    }

    control_power_mapping = {0: -1.0, 1: 0.0, 2: 1.0}
    asset_config = EnergyStorageConfig(
        control_power_mapping=control_power_mapping,
        capacity_kwh=100,
        initial_state_dict={"internal_energy": 50},
    )

    agent_control = AgentControlConfig(
        n_controllers=1,
        rl_config=RLConfig(
            action_size=len(control_power_mapping),
            state_signals=[
                k
                for k, v in signals_info.items()
                if SignalTags.RL_STATE.value in v["tags"]
            ],
        ),
        control_selection={0: ControlSelectionConfig(enabled=False)},
        real_learning_configs={0: RealLearningConfig(enabled=False)},
        sim_learning_configs={0: SimLearningConfig(enabled=False)},
    )
    agent_data = AgentDataConfig(
        control_power_mapping=control_power_mapping,
        signals_info=signals_info,
        tracked_signals=list(signals_info.keys()),
        memory_dump_freq_cron="0 0 1 2 *",
    )
    agent_config = AgentConfig(
        control=agent_control,
        data=agent_data,
        tariff="NO_TARIFF",
        product="ERCOT_ARBITRAGE",
    )

    sim_dir = tmp_path / "sim"
    sim_config = SimulationConfig(
        directory=str(sim_dir),
        time=SimulationTimeConfig(
            start_time="2023-01-01T00:00:00",
            end_time="2023-01-01T00:10:00",
            step_size_s=300,
        ),
        geography=SimulationGeographicalConfig(city=CityEnum.TORONTO),
        agents={"Agent001": agent_config},
        assets={"Agent001": asset_config},
    )

    sim = Simulation(sim_config)
    summary = sim.run()

    assert summary["status"] == "completed"
    assert (sim_dir / "config.json").is_file()
    assert (sim_dir / "sim_summary.json").is_file()
    assert (sim_dir / "time_ref.json").is_file()

    agent_dir = sim_dir / "Agent001"
    assert (agent_dir / "agent.pkl").is_file()

    data_dir = agent_dir / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    assert parquet_files, "Expected parquet data output for the agent"
