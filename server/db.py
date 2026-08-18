"""
Local persistence for the TRELLIS.2 web app -- a single SQLite file on this Mac.

Stdlib-only (sqlite3, hashlib, hmac, secrets): no extra dependencies. Holds
three things:

  operators  -- the real login allow-list (username + salted PBKDF2 hash). The
                presence of >= 1 operator is what flips the app from open
                bootstrap mode to enforced auth (see server/main.py).
  models     -- one row per *completed* generation, so the lab has a persistent,
                shared archive that survives restarts (jobs themselves are still
                in-memory in JobStore; this is the durable record of them).
  meta       -- small key/value bag; currently just the auto-generated
                session_secret used to sign login tokens.

Concurrency: FastAPI runs sync routes in a threadpool and the generation worker
is its own thread, so several threads touch this DB. We open a *fresh*
connection per call (sqlite3 connections aren't meant to cross threads) and run
in WAL mode so readers don't block the single writer. A module-level lock
serializes writes as an extra belt-and-braces guard on top of SQLite's own
locking.
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from . import config

_write_lock = threading.Lock()
_init_lock = threading.Lock()
_initialized = False

# PBKDF2-HMAC-SHA256 work factor. High enough to be a real speed bump for
# offline guessing, low enough to keep login instant on this Mac.
_PBKDF2_ITERATIONS = 200_000


# --------------------------------------------------------------------------
# Connection / schema
# --------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    """Create tables if missing. Idempotent; safe to call on every startup."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operators (
                    username      TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at    REAL NOT NULL,
                    disabled      INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    job_id           TEXT PRIMARY KEY,
                    operator         TEXT,
                    title            TEXT,
                    created_at       REAL NOT NULL,
                    duration_seconds REAL,
                    params           TEXT,
                    vertices         INTEGER,
                    faces            INTEGER,
                    watertight       INTEGER,
                    files            TEXT,
                    preview_filename TEXT,
                    thumb            TEXT,
                    status           TEXT
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.commit()
        _initialized = True


# --------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, stdlib)
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Return an opaque, self-describing hash string: `pbkdf2$iter$salt$hash`."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
def add_operator(username: str, password: str) -> None:
    """Insert or update an operator (upsert = also used by `passwd`)."""
    username = username.strip()
    if not username:
        raise ValueError("Username must not be empty.")
    if not password:
        raise ValueError("Password must not be empty.")
    with _write_lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO operators (username, password_hash, created_at, disabled)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash, disabled=0
            """,
            (username, hash_password(password), time.time()),
        )
        conn.commit()


def remove_operator(username: str) -> bool:
    with _write_lock, _connect() as conn:
        cur = conn.execute("DELETE FROM operators WHERE username = ?", (username.strip(),))
        conn.commit()
        return cur.rowcount > 0


def list_operators() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT username, created_at, disabled FROM operators ORDER BY username"
        ).fetchall()
    return [dict(r) for r in rows]


def count_operators() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM operators WHERE disabled = 0").fetchone()
    return int(row["n"]) if row else 0


def operator_exists(username: str) -> bool:
    """True if the operator is present and not disabled (used to honour removal
    of an account whose session token is still unexpired)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM operators WHERE username = ? AND disabled = 0",
            (username.strip(),),
        ).fetchone()
    return row is not None


def verify_operator(username: str, password: str) -> bool:
    """Constant-time-ish credential check against the stored hash."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM operators WHERE username = ? AND disabled = 0",
            (username.strip(),),
        ).fetchone()
    if not row:
        # Still run a dummy verify so a missing user and a wrong password take a
        # similar amount of time (blunts username enumeration by timing).
        _verify_password(password, "pbkdf2$1$00$00")
        return False
    return _verify_password(password, row["password_hash"])


# --------------------------------------------------------------------------
# Models (the shared, persistent archive of completed generations)
# --------------------------------------------------------------------------
def upsert_model(
    job_id: str,
    operator: Optional[str],
    title: str,
    created_at: float,
    duration_seconds: Optional[float],
    params: Optional[dict],
    vertices: Optional[int],
    faces: Optional[int],
    watertight: Optional[bool],
    files: Optional[list],
    preview_filename: Optional[str],
    status: str = "done",
) -> None:
    with _write_lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO models
                (job_id, operator, title, created_at, duration_seconds, params,
                 vertices, faces, watertight, files, preview_filename, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                operator=excluded.operator, title=excluded.title,
                duration_seconds=excluded.duration_seconds, params=excluded.params,
                vertices=excluded.vertices, faces=excluded.faces,
                watertight=excluded.watertight, files=excluded.files,
                preview_filename=excluded.preview_filename, status=excluded.status
            """,
            (
                job_id,
                operator,
                title,
                created_at,
                duration_seconds,
                json.dumps(params) if params is not None else None,
                vertices,
                faces,
                (None if watertight is None else int(bool(watertight))),
                json.dumps(files) if files is not None else None,
                preview_filename,
                status,
            ),
        )
        conn.commit()


def set_thumb(job_id: str, thumb: str) -> bool:
    """Attach the client-rendered 3D snapshot (a data: URL) to an existing row."""
    with _write_lock, _connect() as conn:
        cur = conn.execute("UPDATE models SET thumb = ? WHERE job_id = ?", (thumb, job_id))
        conn.commit()
        return cur.rowcount > 0


def delete_model(job_id: str) -> bool:
    with _write_lock, _connect() as conn:
        cur = conn.execute("DELETE FROM models WHERE job_id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0


def list_models(limit: int = 200) -> list[dict]:
    """Newest-first list for the shared Library. `params`/`files` are decoded."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d["params"]) if d.get("params") else {}
        d["files"] = json.loads(d["files"]) if d.get("files") else []
        d["watertight"] = None if d["watertight"] is None else bool(d["watertight"])
        out.append(d)
    return out


def get_model(job_id: str) -> Optional[dict]:
    """One archived model by job id, or None.

    The Library outlives the process; JobStore does not. Serving a file for an
    old job needs this to find out the job was ever real.
    """
    with _connect() as conn:
        r = conn.execute("SELECT * FROM models WHERE job_id = ?", (job_id,)).fetchone()
    if r is None:
        return None
    d = dict(r)
    d["params"] = json.loads(d["params"]) if d.get("params") else {}
    d["files"] = json.loads(d["files"]) if d.get("files") else []
    d["watertight"] = None if d["watertight"] is None else bool(d["watertight"])
    return d


def avg_job_seconds(window: int = 20) -> Optional[float]:
    """Average duration of the most recent completed jobs, or None if no data."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT duration_seconds FROM models
            WHERE status = 'done' AND duration_seconds IS NOT NULL AND duration_seconds > 0
            ORDER BY created_at DESC LIMIT ?
            """,
            (window,),
        ).fetchall()
    vals = [r["duration_seconds"] for r in rows]
    if not vals:
        return None
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------
# Meta / session secret
# --------------------------------------------------------------------------
def get_or_create_session_secret() -> str:
    """Stable per-install secret for signing login tokens. Generated once and
    persisted so sessions survive server restarts."""
    with _write_lock, _connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = 'session_secret'").fetchone()
        if row and row["value"]:
            return row["value"]
        secret = secrets.token_hex(32)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('session_secret', ?)",
            (secret,),
        )
        conn.commit()
        return secret


# Ensure the schema exists the moment this module is imported, so every entry
# point (server startup, the admin CLI, a request handler) can use it without
# worrying about init ordering. init_db() is idempotent and cheap after the
# first call; lifespan/admin still call it explicitly, which is a harmless no-op.
init_db()
