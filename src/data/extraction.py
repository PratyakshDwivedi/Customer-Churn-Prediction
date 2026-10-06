"""Data extraction: SQL database -> pandas DataFrame.

This is the single entry point for pulling raw customer records out of
the database. The rest of the pipeline consumes the DataFrame this module
returns and never reads the CSV directly, keeping SQL as the source of
truth.
"""

from __future__ import annotations

import pandas as pd

from src.config import ID_COLUMN, Config, config, get_logger
from src.database.loader import DatabaseLoader

logger = get_logger(__name__)


def extract_customers(
    loader: DatabaseLoader | None = None,
    app_config: Config = config,
    limit: int | None = None,
) -> pd.DataFrame:
    """Extract customer records from the SQL database.

    Parameters
    ----------
    loader:
        Optional pre-built :class:`DatabaseLoader`. A default one is
        created from configuration when omitted.
    app_config:
        Configuration object.
    limit:
        Optional cap on the number of rows (useful for quick tests). Passed
        as a bound parameter, never string-formatted into the SQL.

    Returns
    -------
    pandas.DataFrame
        One row per customer record as stored in the database.

    Raises
    ------
    RuntimeError
        If the customers table is missing or empty.
    """
    loader = loader or DatabaseLoader(app_config=app_config)
    table = app_config.customers_table

    if not loader.table_exists(table):
        raise RuntimeError(
            f"Table '{table}' does not exist. Run scripts/setup_database.py first."
        )

    query = f"SELECT * FROM {table} ORDER BY {ID_COLUMN}"  # noqa: S608 (identifiers from config)
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)

    df = loader.read_query(query, params=params or None)
    if df.empty:
        raise RuntimeError(
            f"Table '{table}' is empty. Run scripts/setup_database.py to seed it."
        )

    logger.info("Extracted %d rows and %d columns from '%s'", len(df), df.shape[1], table)
    return df
