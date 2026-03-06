from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProfileStore:
    def __init__(self, db_path: str = "data/app_profiles.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                  username TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  archetype TEXT NOT NULL,
                  archetype_description TEXT NOT NULL,
                  archetype_graphic TEXT NOT NULL,
                  observations TEXT NOT NULL,
                  joke TEXT NOT NULL,
                  profile_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  image_path TEXT NOT NULL,
                  prediction_json TEXT NOT NULL,
                  FOREIGN KEY(username) REFERENCES users(username)
                )
                """
            )
            conn.commit()

    def get_user(self, username: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if row is None:
                return None
            out = dict(row)
            out["profile"] = json.loads(out.pop("profile_json"))
            return out

    def upsert_user(
        self,
        *,
        username: str,
        created_at: str,
        archetype: str,
        archetype_description: str,
        archetype_graphic: str,
        observations: str,
        joke: str,
        profile: dict,
    ) -> None:
        payload = json.dumps(profile)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users(username, created_at, archetype, archetype_description, archetype_graphic, observations, joke, profile_json)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(username) DO UPDATE SET
                  archetype=excluded.archetype,
                  archetype_description=excluded.archetype_description,
                  archetype_graphic=excluded.archetype_graphic,
                  observations=excluded.observations,
                  joke=excluded.joke,
                  profile_json=excluded.profile_json
                """,
                (
                    username,
                    created_at,
                    archetype,
                    archetype_description,
                    archetype_graphic,
                    observations,
                    joke,
                    payload,
                ),
            )
            conn.commit()

    def add_upload(self, *, username: str, image_path: str, prediction: dict) -> int:
        created_at = utc_now_iso()
        payload = json.dumps(prediction)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO uploads(username, created_at, image_path, prediction_json)
                VALUES(?,?,?,?)
                """,
                (username, created_at, image_path, payload),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_uploads(self, username: str, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, created_at, image_path, prediction_json
                FROM uploads
                WHERE username = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (username, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["prediction"] = json.loads(d.pop("prediction_json"))
            out.append(d)
        return out

    def list_users(self, exclude_username: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if exclude_username:
                rows = conn.execute("SELECT * FROM users WHERE username != ?", (exclude_username,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM users").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["profile"] = json.loads(d.pop("profile_json"))
            out.append(d)
        return out
