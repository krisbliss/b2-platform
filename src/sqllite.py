import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class StoredAgentRecord:
	name: str
	description: str
	yaml_path: str
	token_counts_json: str


class AgentIndexStore:
	"""Persist agent metadata and lightweight text embeddings in SQLite."""

	def __init__(self, db_path: Optional[str] = None):
		self.db_path = Path(db_path) if db_path else self._default_db_path()
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._initialize()

	def _default_db_path(self) -> Path:
		project_root = Path(__file__).parent.parent.parent
		return project_root / "data" / "agents.sqlite3"

	def _connect(self) -> sqlite3.Connection:
		connection = sqlite3.connect(self.db_path)
		connection.row_factory = sqlite3.Row
		return connection

	def _initialize(self) -> None:
		with self._connect() as connection:
			connection.execute(
				"""
				CREATE TABLE IF NOT EXISTS agents (
					name TEXT PRIMARY KEY,
					description TEXT NOT NULL,
					yaml_path TEXT NOT NULL,
					token_counts_json TEXT NOT NULL
				)
				"""
			)

	@staticmethod
	def _normalize_text(text: str) -> List[str]:
		return [token for token in " ".join(text.lower().split()).replace("\n", " ").split(" ") if token]

	@staticmethod
	def _token_counts(text: str) -> Dict[str, int]:
		counts: Dict[str, int] = {}
		for token in AgentIndexStore._normalize_text(text):
			counts[token] = counts.get(token, 0) + 1
		return counts

	@staticmethod
	def _cosine_similarity(left: Dict[str, int], right: Dict[str, int]) -> float:
		if not left or not right:
			return 0.0

		shared_terms = set(left).intersection(right)
		dot_product = sum(left[term] * right[term] for term in shared_terms)
		left_norm = sum(value * value for value in left.values()) ** 0.5
		right_norm = sum(value * value for value in right.values()) ** 0.5

		if left_norm == 0 or right_norm == 0:
			return 0.0

		return dot_product / (left_norm * right_norm)

	def upsert_agent(self, name: str, description: str, yaml_path: str) -> None:
		token_counts = self._token_counts(description)
		with self._connect() as connection:
			connection.execute(
				"""
				INSERT INTO agents (name, description, yaml_path, token_counts_json)
				VALUES (?, ?, ?, ?)
				ON CONFLICT(name) DO UPDATE SET
					description = excluded.description,
					yaml_path = excluded.yaml_path,
					token_counts_json = excluded.token_counts_json
				""",
				(name, description, yaml_path, json.dumps(token_counts)),
			)

	def delete_missing(self, valid_names: Sequence[str]) -> None:
		valid_names = list(valid_names)
		with self._connect() as connection:
			if not valid_names:
				connection.execute("DELETE FROM agents")
				return

			placeholders = ",".join("?" for _ in valid_names)
			connection.execute(
				f"DELETE FROM agents WHERE name NOT IN ({placeholders})",
				valid_names,
			)

	def all_agents(self) -> List[StoredAgentRecord]:
		with self._connect() as connection:
			rows = connection.execute(
				"SELECT name, description, yaml_path, token_counts_json FROM agents ORDER BY name"
			).fetchall()

		return [
			StoredAgentRecord(
				name=row["name"],
				description=row["description"],
				yaml_path=row["yaml_path"],
				token_counts_json=row["token_counts_json"],
			)
			for row in rows
		]

	def match(self, query: str, limit: int = 1) -> List[Tuple[StoredAgentRecord, float]]:
		query_counts = self._token_counts(query)
		scored: List[Tuple[StoredAgentRecord, float]] = []

		for record in self.all_agents():
			record_counts = json.loads(record.token_counts_json)
			score = self._cosine_similarity(query_counts, record_counts)
			scored.append((record, score))

		scored.sort(key=lambda item: item[1], reverse=True)
		return scored[:limit]
