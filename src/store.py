import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StoredAgentRecord:
    name: str
    description: str
    yaml_path: str
    embedding: np.ndarray
    embedding_model: str
    embedding_dim: int
    content_hash: str
    indexed_at: str


class AgentIndexStore:
    """Persist agent metadata and vector embeddings in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else self._default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _default_db_path(self) -> Path:
        project_root = Path(__file__).parent.parent
        return project_root / "data" / "agents.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            table_exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agents'"
            ).fetchone()

            if table_exists:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(agents)").fetchall()
                }
                expected = {
                    "name",
                    "description",
                    "yaml_path",
                    "embedding",
                    "embedding_model",
                    "embedding_dim",
                    "content_hash",
                    "indexed_at",
                }
                if columns != expected:
                    connection.execute("DROP TABLE agents")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    yaml_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )

    def upsert_agent(
        self,
        name: str,
        description: str,
        yaml_path: str,
        embedding: np.ndarray,
        embedding_model: str,
        content_hash: str,
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        indexed_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agents (
                    name,
                    description,
                    yaml_path,
                    embedding,
                    embedding_model,
                    embedding_dim,
                    content_hash,
                    indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    yaml_path = excluded.yaml_path,
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model,
                    embedding_dim = excluded.embedding_dim,
                    content_hash = excluded.content_hash,
                    indexed_at = excluded.indexed_at
                """,
                (
                    name,
                    description,
                    yaml_path,
                    vector.astype(np.float32).tobytes(),
                    embedding_model,
                    int(vector.shape[0]),
                    content_hash,
                    indexed_at,
                ),
            )

    def delete_missing(self, valid_names: Sequence[str]) -> None:
        valid_names = list(valid_names)
        with self._connect() as connection:
            if not valid_names:
                row = connection.execute("SELECT COUNT(*) AS count FROM agents").fetchone()
                row_count = int(row["count"]) if row else 0
                logger.warning(
                    "store.delete_missing called with no valid agent names; keeping %d indexed rows",
                    row_count,
                )
                return

            placeholders = ",".join("?" for _ in valid_names)
            connection.execute(
                f"DELETE FROM agents WHERE name NOT IN ({placeholders})",
                valid_names,
            )

    def clear(self) -> None:
        """Explicitly delete every indexed agent."""
        with self._connect() as connection:
            connection.execute("DELETE FROM agents")

    def all_agents(self) -> list[StoredAgentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    name,
                    description,
                    yaml_path,
                    embedding,
                    embedding_model,
                    embedding_dim,
                    content_hash,
                    indexed_at
                FROM agents
                ORDER BY name
                """
            ).fetchall()

        records: list[StoredAgentRecord] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            records.append(
                StoredAgentRecord(
                    name=row["name"],
                    description=row["description"],
                    yaml_path=row["yaml_path"],
                    embedding=vector,
                    embedding_model=row["embedding_model"],
                    embedding_dim=row["embedding_dim"],
                    content_hash=row["content_hash"],
                    indexed_at=row["indexed_at"],
                )
            )
        return records

    def match(
        self,
        query_embedding: np.ndarray,
        limit: int = 1,
    ) -> list[tuple[StoredAgentRecord, float]]:
        records = self.all_agents()
        if not records:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        matrix = np.stack([record.embedding for record in records], axis=0).astype(np.float32)

        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: store={matrix.shape[1]} query={query.shape[0]}"
            )

        scores = matrix @ query
        top_indices = np.argsort(scores)[::-1][:limit]
        return [(records[i], float(scores[i])) for i in top_indices]
