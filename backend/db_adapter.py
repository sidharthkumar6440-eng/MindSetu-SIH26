"""Resilient Database Adapter for MindSetu-SIH26.

Provides a unified connection provider that:
1. Attempts PostgreSQL connection using configured credentials with a fast timeout (2.0s).
2. If PostgreSQL is unconfigured or unreachable, automatically falls back to a persistent
   local SQLite database (backend/mindsetu_local.db).
3. Ensures all core and SIH26186 tables and seeded doctors are created seamlessly.
4. Translates PostgreSQL %s placeholders and types so code remains completely compatible.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("mindsetu.db_adapter")

SQLITE_DB_PATH = Path(__file__).resolve().parent / "mindsetu_local.db"

_lock = threading.RLock()
_active_engine: Optional[str] = None  # "postgresql" or "sqlite"
_pg_pool: Any = None
_sqlite_conn: Optional[sqlite3.Connection] = None


class SQLiteCursorWrapper:
    """Wraps an SQLite cursor to provide psycopg-compatible execute and context manager."""

    def __init__(self, cur: sqlite3.Cursor):
        self._cur = cur

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._cur.close()
        except Exception:
            pass

    def execute(self, sql: str, params: Any = None):
        # 1. Translate PostgreSQL %s placeholder to SQLite ? placeholder
        sql = sql.replace("%s", "?")

        # 2. Strip PostgreSQL type casts (e.g. ::jsonb, ::numeric, ::text)
        sql = re.sub(r"::\w+", "", sql)

        # 3. Translate PostgreSQL ILIKE to SQLite LIKE
        sql = re.sub(r"\bILIKE\b", "LIKE", sql, flags=re.IGNORECASE)

        # 4. Translate PostgreSQL INTERVAL arithmetic to SQLite date functions
        sql = re.sub(
            r"NOW\(\)\s*-\s*INTERVAL\s*'(\d+)\s*days?'",
            r"datetime('now', '-\1 days')",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(r"\bNOW\(\)", "(datetime('now'))", sql, flags=re.IGNORECASE)

        # 3. Clean and convert parameters
        if params is not None:
            clean_params = []
            for p in params:
                if isinstance(p, (dict, list)):
                    clean_params.append(json.dumps(p))
                elif isinstance(p, uuid.UUID):
                    clean_params.append(str(p))
                else:
                    clean_params.append(p)
            return self._cur.execute(sql, clean_params)
        return self._cur.execute(sql)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __getattr__(self, name):
        return getattr(self._cur, name)


class SQLiteConnectionWrapper:
    """Wraps an SQLite connection to provide psycopg ConnectionPool / connection semantics."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def cursor(self):
        return SQLiteCursorWrapper(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._conn.rollback()
        else:
            self._conn.commit()


def _get_sqlite_connection() -> sqlite3.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        _sqlite_conn = sqlite3.connect(
            str(SQLITE_DB_PATH),
            check_same_thread=False,
            timeout=10.0,
        )
        _sqlite_conn.row_factory = sqlite3.Row
        # Custom functions for SQLite compatibility with PostgreSQL syntax
        _sqlite_conn.create_function("NOW", 0, lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        _sqlite_conn.create_function("DATE", 1, lambda val: str(val)[:10] if val else None)
        _sqlite_conn.create_function("current_database", 0, lambda: "mindsetu_local_sqlite")
        _init_sqlite_schema(_sqlite_conn)
    return _sqlite_conn


def _init_sqlite_schema(conn: sqlite3.Connection):
    """Initializes all necessary tables in SQLite if they do not already exist."""
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            anonymous INTEGER NOT NULL DEFAULT 1,
            consent_given INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS mood_entries (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id),
            mood INTEGER,
            note TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sih26186_wellness (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            answers TEXT NOT NULL,
            stress_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sih26186_workload (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            role TEXT NOT NULL,
            unit TEXT,
            duty_hours REAL NOT NULL,
            night_duties INTEGER NOT NULL,
            rest_hours REAL NOT NULL,
            days_since_leave INTEGER NOT NULL,
            workload_level INTEGER NOT NULL,
            high_pressure_assignment INTEGER NOT NULL DEFAULT 0,
            duty_change_frequency INTEGER NOT NULL DEFAULT 0,
            workload_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sih26186_analysis (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            wellness_score REAL NOT NULL,
            workload_score REAL NOT NULL,
            combined_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS doctors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            specialization TEXT NOT NULL,
            qualification TEXT NOT NULL,
            experience_years INTEGER NOT NULL,
            bio TEXT NOT NULL,
            avatar_color TEXT NOT NULL,
            rating REAL NOT NULL,
            review_count INTEGER NOT NULL,
            focus_areas TEXT NOT NULL,
            availability_status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            doctor_id TEXT NOT NULL REFERENCES doctors(id),
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'confirmed',
            meeting_id TEXT NOT NULL DEFAULT '',
            notes TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_active_doctor_slot
        ON appointments (doctor_id, appointment_date, appointment_time)
        WHERE status != 'cancelled';
        """
    )
    conn.commit()

    # Seed default doctors if table is empty
    cur.execute("SELECT COUNT(*) FROM doctors")
    if cur.fetchone()[0] == 0:
        _seed_sqlite_doctors(cur)
        conn.commit()


def _seed_sqlite_doctors(cur: sqlite3.Cursor):
    from sih26186_doctor_routes import SEEDED_DOCTORS

    for doc in SEEDED_DOCTORS:
        cur.execute(
            """
            INSERT OR IGNORE INTO doctors
                (id, name, specialization, qualification, experience_years, bio,
                 avatar_color, rating, review_count, focus_areas, availability_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(doc["id"]),
                doc["name"],
                doc["specialization"],
                doc["qualification"],
                doc["experience_years"],
                doc["bio"],
                doc["avatar_color"],
                doc["rating"],
                doc["review_count"],
                json.dumps(doc["focus_areas"]),
                doc["availability_status"],
            ),
        )


def _try_init_postgresql() -> bool:
    """Attempts to connect to PostgreSQL with a 2-second timeout."""
    global _pg_pool
    database_url = (os.getenv("DATABASE_URL") or "").strip()
    password = os.getenv("DB_PASSWORD")

    if not database_url and not password:
        logger.info("No PostgreSQL credentials in environment; using SQLite fallback.")
        return False

    try:
        from psycopg_pool import ConnectionPool

        if database_url:
            conninfo = database_url
        else:
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            dbname = os.getenv("DB_NAME", "mindsetu_db")
            user = os.getenv("DB_USER", "postgres")
            conninfo = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        pool = ConnectionPool(
            conninfo=conninfo,
            min_size=0,
            max_size=10,
            open=True,
            timeout=2.0,  # Fast 2.0s timeout to prevent thread stalls
            max_idle=60.0,
        )
        # Test connection
        with pool.connection(timeout=2.0) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        _pg_pool = pool
        return True
    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s); falling back to persistent SQLite.", exc)
        return False


def init_db():
    """Initializes the database engine (PostgreSQL if available, else SQLite)."""
    global _active_engine
    with _lock:
        if _active_engine is None:
            if _try_init_postgresql():
                _active_engine = "postgresql"
                logger.info("Database initialized with PostgreSQL engine.")
            else:
                _get_sqlite_connection()
                _active_engine = "sqlite"
                logger.info("Database initialized with SQLite persistent fallback engine: %s", SQLITE_DB_PATH)


@contextmanager
def get_connection():
    """Borrow a connection from the active database engine."""
    if _active_engine is None:
        init_db()

    if _active_engine == "postgresql" and _pg_pool is not None:
        try:
            with _pg_pool.connection(timeout=1.5) as conn:
                yield conn
                return
        except Exception as exc:
            logger.warning("PostgreSQL connection error in pool (%s); switching to SQLite permanently.", exc)
            with _lock:
                _active_engine = "sqlite"
                if _pg_pool is not None:
                    try:
                        _pg_pool.close()
                    except Exception:
                        pass
                    _pg_pool = None

    sqlite_raw = _get_sqlite_connection()
    with _lock:
        wrapper = SQLiteConnectionWrapper(sqlite_raw)
        with wrapper:
            yield wrapper


def close_db():
    """Cleanly closes open connections and pools."""
    global _pg_pool, _sqlite_conn, _active_engine
    with _lock:
        if _pg_pool is not None:
            try:
                _pg_pool.close()
            except Exception:
                pass
            _pg_pool = None
        if _sqlite_conn is not None:
            try:
                _sqlite_conn.close()
            except Exception:
                pass
            _sqlite_conn = None
        _active_engine = None


def get_engine_name() -> str:
    if _active_engine is None:
        init_db()
    return _active_engine or "sqlite"
