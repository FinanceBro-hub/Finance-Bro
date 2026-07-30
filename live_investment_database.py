# -*- coding: utf-8 -*-
"""SQLite storage for the Finance Bro Live Investment module."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sqlite3

import pandas as pd


DATABASE_PATH = Path(__file__).with_name("finance_bro_live.db")
DEFAULT_PORTFOLIO_NAME = "My Portfolio"


def get_connection() -> sqlite3.Connection:
    """Open a short-lived SQLite connection."""

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[str]:
    rows = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _create_positions_table(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            portfolio_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL CHECK (quantity > 0),
            average_cost REAL NOT NULL CHECK (average_cost > 0),
            purchase_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (portfolio_id, ticker),
            FOREIGN KEY (portfolio_id)
                REFERENCES portfolios(id)
                ON DELETE CASCADE
        )
        """
    )


def _ensure_default_portfolio(
    connection: sqlite3.Connection,
) -> int:
    row = connection.execute(
        """
        SELECT id
        FROM portfolios
        ORDER BY id
        LIMIT 1
        """
    ).fetchone()

    if row is not None:
        return int(row["id"])

    now = datetime.now().isoformat(timespec="seconds")

    cursor = connection.execute(
        """
        INSERT INTO portfolios (
            name,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?)
        """,
        (
            DEFAULT_PORTFOLIO_NAME,
            now,
            now,
        ),
    )

    return int(cursor.lastrowid)


def _migrate_old_positions_table(
    connection: sqlite3.Connection,
) -> None:
    """Move data from the original single-portfolio table."""

    if not _table_exists(connection, "positions"):
        _create_positions_table(connection)
        return

    columns = _table_columns(
        connection,
        "positions",
    )

    if "portfolio_id" in columns:
        return

    default_portfolio_id = _ensure_default_portfolio(
        connection
    )

    connection.execute(
        "ALTER TABLE positions RENAME TO positions_legacy"
    )

    _create_positions_table(connection)

    connection.execute(
        """
        INSERT INTO positions (
            portfolio_id,
            ticker,
            quantity,
            average_cost,
            purchase_date,
            created_at,
            updated_at
        )
        SELECT
            ?,
            ticker,
            quantity,
            CASE
                WHEN average_cost <= 0 THEN 0.000001
                ELSE average_cost
            END,
            purchase_date,
            created_at,
            updated_at
        FROM positions_legacy
        """,
        (default_portfolio_id,),
    )

    connection.execute(
        "DROP TABLE positions_legacy"
    )


def initialize_database() -> None:
    """Create or update the database without deleting saved data."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        _migrate_old_positions_table(connection)
        connection.commit()


def load_portfolios() -> pd.DataFrame:
    """Return all saved portfolios."""

    initialize_database()

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                id,
                name,
                created_at,
                updated_at
            FROM portfolios
            ORDER BY created_at, id
            """,
            connection,
        )


def create_portfolio(name: str) -> int:
    """Create a portfolio and return its identifier."""

    initialize_database()

    clean_name = " ".join(str(name).split()).strip()

    if not clean_name:
        raise ValueError("Enter a portfolio name.")

    if len(clean_name) > 60:
        raise ValueError(
            "Portfolio name must contain 60 characters or fewer."
        )

    now = datetime.now().isoformat(timespec="seconds")

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO portfolios (
                    name,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    clean_name,
                    now,
                    now,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    except sqlite3.IntegrityError as error:
        raise ValueError(
            "A portfolio with that name already exists."
        ) from error


def rename_portfolio(
    portfolio_id: int,
    new_name: str,
) -> None:
    """Rename one portfolio."""

    initialize_database()

    clean_name = " ".join(str(new_name).split()).strip()

    if not clean_name:
        raise ValueError("Enter a new portfolio name.")

    if len(clean_name) > 60:
        raise ValueError(
            "Portfolio name must contain 60 characters or fewer."
        )

    now = datetime.now().isoformat(timespec="seconds")

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE portfolios
                SET
                    name = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    clean_name,
                    now,
                    int(portfolio_id),
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError("Portfolio was not found.")

            connection.commit()

    except sqlite3.IntegrityError as error:
        raise ValueError(
            "A portfolio with that name already exists."
        ) from error


def delete_portfolio(portfolio_id: int) -> None:
    """Delete one portfolio and all its saved positions."""

    initialize_database()

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM portfolios WHERE id = ?",
            (int(portfolio_id),),
        )
        connection.commit()


def load_positions(portfolio_id: int) -> pd.DataFrame:
    """Return the positions saved in one portfolio."""

    initialize_database()

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                ticker,
                quantity,
                average_cost,
                purchase_date,
                created_at,
                updated_at
            FROM positions
            WHERE portfolio_id = ?
            ORDER BY ticker
            """,
            connection,
            params=(int(portfolio_id),),
        )


def save_position(
    portfolio_id: int,
    ticker: str,
    quantity: float,
    average_cost: float,
    purchase_date: date | str,
) -> None:
    """Save or replace one current position."""

    initialize_database()

    clean_ticker = str(ticker).strip().upper()
    quantity = float(quantity)
    average_cost = float(average_cost)

    if not clean_ticker:
        raise ValueError("Ticker cannot be empty.")

    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")

    if average_cost <= 0:
        raise ValueError(
            "Average purchase price must be greater than zero."
        )

    purchase_date_text = (
        purchase_date.isoformat()
        if isinstance(purchase_date, date)
        else str(purchase_date)
    )

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO positions (
                portfolio_id,
                ticker,
                quantity,
                average_cost,
                purchase_date,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(portfolio_id, ticker) DO UPDATE SET
                quantity = excluded.quantity,
                average_cost = excluded.average_cost,
                purchase_date = excluded.purchase_date,
                updated_at = excluded.updated_at
            """,
            (
                int(portfolio_id),
                clean_ticker,
                quantity,
                average_cost,
                purchase_date_text,
                now,
                now,
            ),
        )

        connection.execute(
            """
            UPDATE portfolios
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                int(portfolio_id),
            ),
        )

        connection.commit()


def delete_position(
    portfolio_id: int,
    ticker: str,
) -> None:
    """Delete one position from one portfolio."""

    initialize_database()

    clean_ticker = str(ticker).strip().upper()
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM positions
            WHERE portfolio_id = ?
              AND ticker = ?
            """,
            (
                int(portfolio_id),
                clean_ticker,
            ),
        )

        connection.execute(
            """
            UPDATE portfolios
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                int(portfolio_id),
            ),
        )

        connection.commit()
