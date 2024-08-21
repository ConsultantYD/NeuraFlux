import os
from logging import StreamHandler

import pandas as pd

from neuraflux.global_variables import (
    LOG_ENTITY_KEY,
    LOG_LEVEL_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_MODULE_KEY,
    LOG_SIM_T_KEY,
    LOG_TIMESTAMP_KEY,
    LOGGING_DB_NAME,
)
from neuraflux.utils_sql import add_dataframe_to_table, create_connection_to_db


class StructuredLogHandler(StreamHandler):
    def __init__(self, db_dir: str):
        super().__init__()
        self.db_filepath = os.path.join(db_dir, LOGGING_DB_NAME)
        self.conn = create_connection_to_db(self.db_filepath)

    def emit(self, record):
        message_level = record.levelname
        module = record.module
        creation_time = record.created
        logging_dict = record.msg

        # Verify input is a dict
        if not isinstance(logging_dict, dict):
            raise ValueError(
                f"Logging message must be a dictionary. Received {type(logging_dict)}: {logging_dict}"
            )

        sim_time = (
            None if LOG_SIM_T_KEY not in logging_dict else logging_dict[LOG_SIM_T_KEY]
        )
        entity = (
            None if LOG_ENTITY_KEY not in logging_dict else logging_dict[LOG_ENTITY_KEY]
        )
        method = (
            None if LOG_METHOD_KEY not in logging_dict else logging_dict[LOG_METHOD_KEY]
        )
        message = (
            None
            if LOG_MESSAGE_KEY not in logging_dict
            else logging_dict[LOG_MESSAGE_KEY]
        )
        log_df = pd.DataFrame(
            {
                LOG_TIMESTAMP_KEY: [creation_time],
                LOG_MODULE_KEY: [module],
                LOG_LEVEL_KEY: [message_level],
                LOG_SIM_T_KEY: [sim_time],
                LOG_ENTITY_KEY: [entity],
                LOG_METHOD_KEY: [method],
                LOG_MESSAGE_KEY: [message],
            }
        )

        # Insert into SQLite database
        add_dataframe_to_table(
            log_df, self.conn, "execution_logs", index_col=LOG_TIMESTAMP_KEY
        )
