"""Reusable SQLite access layer.

Everything that touches the database goes through this module: opening
connections, creating the schema, bulk-seeding records and reading rows
back into pandas. Keeping it in one place means the rest of the pipeline
never embeds SQL or connection handling and can be pointed at a different
database purely through configuration.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import Config, config, get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """Thin, reusable wrapper around a SQLite churn database."""

    def __init__(self, db_path: Path | str | None = None, app_config: Config = config):
        """Create a loader.

        Parameters
        ----------
        db_path:
            Optional explicit path to the SQLite file. Defaults to the
            path from :class:`~src.config.Config`.
        app_config:
            Configuration object (injected for testability).
        """
        self.config = app_config
        self.db_path = Path(db_path) if db_path is not None else app_config.database_path

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection, committing on success and always closing.

        Used as a context manager so callers never leak connections even
        when an exception is raised mid-transaction.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Schema / seeding
    # ------------------------------------------------------------------ #
    def execute_script(self, sql_path: Path | str) -> None:
        """Execute a multi-statement ``.sql`` file (e.g. the schema)."""
        sql_path = Path(sql_path)
        script = sql_path.read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(script)
        logger.info("Executed SQL script %s", sql_path.name)

    def seed_dataframe(
        self, df: pd.DataFrame, table: str | None = None, if_exists: str = "replace"
    ) -> int:
        """Bulk-insert a DataFrame into a table and return the row count.

        ``pandas.to_sql`` parameterises inserts internally, so no values
        are interpolated into raw SQL strings here.
        """
        table = table or self.config.customers_table
        with self.connect() as conn:
            df.to_sql(table, conn, if_exists=if_exists, index=False)
        logger.info("Seeded %d rows into table '%s'", len(df), table)
        return len(df)

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def table_exists(self, table: str | None = None) -> bool:
        """Return ``True`` if the given table exists in the database."""
        table = table or self.config.customers_table
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;"
        with self.connect() as conn:
            cursor = conn.execute(query, (table,))
            return cursor.fetchone() is not None

    def row_count(self, table: str | None = None) -> int:
        """Return the number of rows in a table."""
        table = self._safe_table_name(table)
        with self.connect() as conn:
            # Table name cannot be a bound parameter in SQLite; it is
            # validated against the known schema instead (see _safe_table_name).
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table};")  # noqa: S608
            return int(cursor.fetchone()[0])

    def read_query(
        self, query: str, params: tuple | dict | None = None
    ) -> pd.DataFrame:
        """Run an arbitrary SELECT (optionally parameterised) into a DataFrame."""
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def _safe_table_name(self, table: str | None) -> str:
        """Validate a table name against the configured schema.

        Table/identifier names cannot be passed as bound parameters, so we
        guard against injection by only allowing the known table name.
        """
        table = table or self.config.customers_table
        allowed = {self.config.customers_table}
        if table not in allowed:
            raise ValueError(f"Unknown table '{table}'. Allowed: {sorted(allowed)}")
        return table
