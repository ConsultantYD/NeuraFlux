import datetime as dt
from pathlib import Path

import pandas as pd

from neuraflux.agency.agent import Agent
from neuraflux.geography import CityEnum
from neuraflux.global_variables import DONE_KEY, ENERGY_KEY, OAT_KEY, PRICE_KEY, REWARD_KEY, TARIFF_KEY
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
from neuraflux.schemas.asset_config import BuildingConfig
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
        self.preloaded = True

    def get_weather_info_at_time(self, time: dt.datetime):
        return type("WeatherInfo", (), {"temperature": 10.0})()


def _building_signals_info():
    return {
        "temperature_1": {"tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value]},
        "temperature_2": {"tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value]},
        "temperature_3": {"tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value]},
        OAT_KEY: {"tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value]},
        "hvac_1": {"tags": [SignalTags.OBSERVATION.value, SignalTags.RL_STATE.value]},
        "hvac_2": {"tags": [SignalTags.OBSERVATION.value, SignalTags.RL_STATE.value]},
        "hvac_3": {"tags": [SignalTags.OBSERVATION.value, SignalTags.RL_STATE.value]},
        "cool_setpoint": {"tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value]},
        "heat_setpoint": {"tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value]},
        "occupancy": {"tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value]},
    }


def test_commercial_building_writes_artifacts_and_parquet_schema(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("neuraflux.simulation.Weather", _DummyWeather)

    signals_info = _building_signals_info()
    asset_config = BuildingConfig()
    control_power_mapping = asset_config.control_power_mapping

    agent_control = AgentControlConfig(
        n_controllers=3,
        rl_config=RLConfig(
            n_controllers=3,
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
        tariff="ONTARIO_GEN_TOU",
        product="HVAC_TARIFF_COMFORT",
    )

    sim_dir = tmp_path / "sim"
    sim_config = SimulationConfig(
        directory=str(sim_dir),
        time=SimulationTimeConfig(
            start_time="2023-01-01T00:00:00",
            end_time="2023-01-01T01:00:00",
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
    assert (sim_dir / "metrics.json").is_file()
    assert (sim_dir / "logs.db").is_file()

    agent_dir = sim_dir / "Agent001"
    assert (agent_dir / "agent.pkl").is_file()

    data_dir = agent_dir / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    assert parquet_files, "Expected parquet data output for the agent"

    df = pd.read_parquet(data_dir)
    expected_cols = {
        "timestamp",
        "control_1",
        "control_2",
        "control_3",
        "policy",
        ENERGY_KEY,
        TARIFF_KEY,
        REWARD_KEY,
        DONE_KEY,
        PRICE_KEY,
    }
    assert expected_cols.issubset(df.columns)


def test_commercial_building_learning_triggers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("neuraflux.simulation.Weather", _DummyWeather)

    calls = {"sim": 0, "real": 0}

    def _fake_sim_training(self: Agent, *args, **kwargs):
        calls["sim"] += 1

    def _fake_real_training(self: Agent, *args, **kwargs):
        calls["real"] += 1

    monkeypatch.setattr(Agent, "simulated_rl_training", _fake_sim_training)
    monkeypatch.setattr(Agent, "rl_training", _fake_real_training)

    signals_info = _building_signals_info()
    asset_config = BuildingConfig()
    control_power_mapping = asset_config.control_power_mapping

    agent_control = AgentControlConfig(
        n_controllers=3,
        rl_config=RLConfig(
            n_controllers=3,
            action_size=len(control_power_mapping),
            state_signals=[
                k
                for k, v in signals_info.items()
                if SignalTags.RL_STATE.value in v["tags"]
            ],
        ),
        control_selection={0: ControlSelectionConfig(enabled=False)},
        real_learning_configs={
            0: RealLearningConfig(enabled=True, trigger_freq_cron="0 0 * * *")
        },
        sim_learning_configs={
            0: SimLearningConfig(enabled=True, trigger_freq_cron="0 0 * * *")
        },
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
        tariff="ONTARIO_GEN_TOU",
        product="HVAC_TARIFF_COMFORT",
    )

    sim_dir = tmp_path / "sim"
    sim_config = SimulationConfig(
        directory=str(sim_dir),
        time=SimulationTimeConfig(
            start_time="2023-01-01T23:50:00",
            end_time="2023-01-02T00:10:00",
            step_size_s=300,
        ),
        geography=SimulationGeographicalConfig(city=CityEnum.TORONTO),
        agents={"Agent001": agent_config},
        assets={"Agent001": asset_config},
    )

    sim = Simulation(sim_config)
    summary = sim.run()

    assert summary["status"] == "completed"
    assert calls["sim"] == 1
    assert calls["real"] == 1


def test_commercial_building_simulated_trajectory_q_policy_has_no_nan_hvac_rl_state(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("neuraflux.simulation.Weather", _DummyWeather)

    class _DummyQEstimator:
        def __init__(self, *, n_controllers: int, action_size: int):
            self.n_controllers = int(n_controllers)
            self.action_size = int(action_size)

        def forward_pass(self, states, target_model: bool = False, lite_model: bool = False):
            batch = int(states.shape[0])
            return [
                (0.0 * states[:, :1, :1]).astype("float32").reshape(batch, 1, 1)
                .repeat(self.action_size, axis=2)
                for _ in range(self.n_controllers)
            ]

        def lite_predict(self, states):
            return self.forward_pass(states)

    def _fake_get_q_estimator(self: Agent, registry_dir: str):
        return (
            _DummyQEstimator(
                n_controllers=self.config.control.n_controllers,
                action_size=self.config.control.rl_config.action_size,
            ),
            {"name": "dummy", "training_type": "sim"},
        )

    monkeypatch.setattr(Agent, "get_q_estimator", _fake_get_q_estimator)

    signals_info = _building_signals_info()
    asset_config = BuildingConfig()
    control_power_mapping = asset_config.control_power_mapping

    agent_control = AgentControlConfig(
        n_controllers=3,
        rl_config=RLConfig(
            n_controllers=3,
            action_size=len(control_power_mapping),
            state_signals=[
                k for k, v in signals_info.items() if SignalTags.RL_STATE.value in v["tags"]
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
        tariff="ONTARIO_GEN_TOU",
        product="HVAC_TARIFF_COMFORT",
    )

    sim_dir = tmp_path / "sim"
    sim_config = SimulationConfig(
        directory=str(sim_dir),
        time=SimulationTimeConfig(
            start_time="2023-01-01T00:00:00",
            end_time="2023-01-01T02:00:00",
            step_size_s=300,
        ),
        geography=SimulationGeographicalConfig(city=CityEnum.TORONTO),
        agents={"Agent001": agent_config},
        assets={"Agent001": asset_config},
    )

    sim = Simulation(sim_config)
    summary = sim.run()
    assert summary["status"] == "completed"

    agent = sim.agents["Agent001"]
    df = agent.get_data()
    assert len(df) >= 8

    traj_df = agent.simulate_trajectory_at_time(
        timestamp=df.index[0],
        sim_len=8,
        policy="q_policy",
        policy_kwargs={"epsilon": 0.0},
    )

    assert not traj_df[["hvac_1", "hvac_2", "hvac_3"]].isna().any().any()
