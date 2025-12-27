import os
import json
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
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # Ensure the DB file is a valid SQLite database (non-empty header) and
        # dashboards can reliably query the expected tables even if no logs are
        # emitted during a run.
        cur = self.conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS execution_logs (
                "{LOG_TIMESTAMP_KEY}" REAL,
                "{LOG_MODULE_KEY}" TEXT,
                "{LOG_LEVEL_KEY}" TEXT,
                "{LOG_SIM_T_KEY}" TEXT,
                "{LOG_ENTITY_KEY}" TEXT,
                "{LOG_METHOD_KEY}" TEXT,
                "{LOG_MESSAGE_KEY}" TEXT
            );
            """
        )
        self.conn.commit()

    def emit(self, record):
        message_level = record.levelname
        module = record.module
        creation_time = record.created
        payload = record.msg

        if isinstance(payload, dict):
            sim_time = payload.get(LOG_SIM_T_KEY)
            entity = payload.get(LOG_ENTITY_KEY)
            method = payload.get(LOG_METHOD_KEY)
            message = payload.get(LOG_MESSAGE_KEY)
            if message is None:
                message = json.dumps(payload, default=str)
        else:
            sim_time = None
            entity = None
            method = None
            message = record.getMessage()

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
        try:
            add_dataframe_to_table(
                log_df, self.conn, "execution_logs", index_col=LOG_TIMESTAMP_KEY
            )
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if hasattr(self, "conn") and self.conn is not None:
                self.conn.close()
        finally:
            super().close()
