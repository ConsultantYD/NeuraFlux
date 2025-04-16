import datetime as dt

import pandas as pd

from neuraflux.agency.utils_data import get_u_columns, get_w_columns, get_x_columns
from neuraflux.global_variables import TIMESTAMP_KEY
from neuraflux.schemas.agency import AgentConfig


class Trajectory:
    """
    A trajectory is a collection of time series data,
    represented internally as list of record dictionaries.
    """

    def __init__(
        self,
        state_cols: list[str],
        control_cols: list[str],
        exogenous_cols: list[str],
        history_records: list[dict] | None = None,
        control_records: list[dict] | None = None,
        exogenous_records: list[dict] | None = None,
        state_records: list[dict] | None = None,
        tf_records: list[dict] | None = None,
    ):
        # Initialize the trajectory with empty lists if None
        self.history_records = [] if history_records is None else history_records
        self.state_records = [] if state_records is None else state_records
        self.control_records = [] if control_records is None else control_records
        self.exogenous_records = [] if exogenous_records is None else exogenous_records
        self.tf_records = [] if tf_records is None else tf_records

    def add_control_record(self, timestamp: dt.datetime, control_record: dict):
        """
        Add a list of control records to the trajectory.
        Each record should be a dictionary with a timestamp and
        the corresponding control values.

        Parameters
        ----------
        timestamp : datetime
            The timestamp for the control record.
        control_record : dict
            A dictionary containing the control record.
            It should have a timestamp and the corresponding
            control values.
        """
        control_record = control_record.copy()
        control_record[TIMESTAMP_KEY] = timestamp
        self.control_records.append(control_record)

    def add_exogenous_record_list(self, records_list: list[dict]):
        """
        Add a list of exogenous records to the trajectory.
        Each record should be a dictionary with a timestamp and
        the corresponding exogenous values.

        Parameters
        ----------
        records_list : list[dict]
            A list of dictionaries containing the exogenous records.
            Each dictionary should have a timestamp and the corresponding
            exogenous values.
        """
        self.exogenous_records.extend(records_list)

    def add_state_record(self, timestamp: dt.datetime, state_record: dict):
        """
        Add a list of state records to the trajectory.
        Each record should be a dictionary with a timestamp and
        the corresponding state values.

        Parameters
        ----------
        timestamp : datetime
            The timestamp for the state record.
        state_record : dict
            A dictionary containing the state record.
            It should have a timestamp and the corresponding
            state values.
        """
        state_record = state_record.copy()
        state_record[TIMESTAMP_KEY] = timestamp
        self.state_records.append(state_record)

    def as_df(self) -> pd.DataFrame:
        """
        Convert the trajectory to a pandas DataFrame.
        The DataFrame will have the timestamp as the index
        and will be sorted by timestamp.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the trajectory data.
        """
        # Combine all records into a single list
        all_records = (
            self.history_records
            + self.control_records
            + self.exogenous_records
            + self.state_records
            + self.tf_records
        )
        df = pd.DataFrame(all_records)
        df = df.set_index(TIMESTAMP_KEY)
        df = df.sort_index()

        # Define an aggregation function that takes the first non-null value
        def first_non_null(series):
            non_null = series.dropna()
            if not non_null.empty:
                return non_null.iloc[0]
            return None  # or np.nan if you prefer

        # Group by the index (timestamp) and aggregate using our custom function
        fused_df = df.groupby(df.index).agg(first_non_null)
        # Make sure the index is sorted (groupby sometimes loses order guarantees)
        fused_df = fused_df.sort_index()

        return fused_df

    @classmethod
    def partial_from_agent_df(
        cls,
        agent_config: AgentConfig,
        df: pd.DataFrame,
    ) -> "Trajectory":
        """
        Create a partial trajectory from the agent DataFrame.
        The DataFrame should have a timestamp index and contain
        columns for state, control, and exogenous variables.
        The history length is determined automatically from the
        rl config.

        Parameters
        ----------
        agent_config : AgentConfig
            The configuration object for the agent.
        df : pd.DataFrame
            The DataFrame containing the agent data.
        """
        x_cols = get_x_columns(agent_config)
        u_cols = get_u_columns(agent_config)
        w_cols = get_w_columns(agent_config)
        tf_cols = [col for col in df.columns if col.startswith("tf_")]
        history_len = agent_config.control.rl_config.history_length
        history_records = (
            df.iloc[0:history_len].reset_index(drop=False).to_dict("records")
        )
        tf_records = df[tf_cols].reset_index(drop=False).to_dict("records")
        return cls(
            state_cols=x_cols,
            control_cols=u_cols,
            exogenous_cols=w_cols,
            history_records=history_records,
            exogenous_records=df.loc[df.index[history_len:], w_cols]
            .reset_index(drop=False)
            .to_dict("records"),
            tf_records=tf_records,
        )
