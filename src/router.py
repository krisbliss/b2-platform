from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

from .embeddings import GoogleEmbedder
from .orchestrator import agent as agent_module
from .store import AgentIndexStore, StoredAgentRecord

logger = logging.getLogger(__name__)


class AgentRouter:
	"""Index agent YAML files and route queries to the best matching agent."""

	def __init__(
		self,
		agents_dir: Optional[str] = None,
		db_path: Optional[str] = None,
		embedder: Optional[GoogleEmbedder] = None,
		min_score: float = 0.5,
	):
		start = perf_counter()
		self.agents_dir = self._agents_dir(agents_dir)
		self.store = AgentIndexStore(db_path=db_path)
		self.embedder = embedder or GoogleEmbedder()
		self.min_score = min_score
		self.refresh_index()
		logger.info("router.init done elapsed=%.3fs", perf_counter() - start)

	def _agents_dir(self, agents_dir: Optional[str] = None) -> Path:
		if agents_dir is None:
			project_root = Path(__file__).parent.parent
			return project_root / "agents"
		return Path(agents_dir)

	@staticmethod
	def _hash_content(description: str, system_prompt: str) -> str:
		content = f"{description}\n---\n{system_prompt}".encode("utf-8")
		return hashlib.sha256(content).hexdigest()

	@staticmethod
	def _load_yaml_metadata(yaml_path: Path) -> dict[str, str]:
		yaml_module = agent_module._load_yaml_module()

		with open(yaml_path, "r", encoding="utf-8") as handle:
			agent_data = yaml_module.safe_load(handle) or {}

		name = agent_data.get("name")
		system_prompt = agent_data.get("system_prompt", "")
		description = agent_data.get("description") or (system_prompt.splitlines()[0] if system_prompt else "")

		if not name:
			raise ValueError(f"Agent file '{yaml_path}' is missing required 'name' field")

		embed_text = f"{name}\n{description}\n{system_prompt}"
		return {
			"name": name,
			"description": description,
			"system_prompt": system_prompt,
			"yaml_path": str(yaml_path),
			"embed_text": embed_text,
		}

	def refresh_index(self) -> None:
		start = perf_counter()
		if not self.agents_dir.exists():
			raise FileNotFoundError(f"Agents directory not found at {self.agents_dir}")

		existing: dict[str, StoredAgentRecord] = {r.name: r for r in self.store.all_agents()}
		indexed_names: list[str] = []
		to_embed_rows: list[dict[str, str]] = []
		unchanged = 0

		for yaml_path in self.agents_dir.glob("*.yaml"):
			metadata = self._load_yaml_metadata(yaml_path)
			indexed_names.append(metadata["name"])

			content_hash = self._hash_content(metadata["description"], metadata["system_prompt"])
			metadata["content_hash"] = content_hash

			existing_row = existing.get(metadata["name"])
			if (
				existing_row
				and existing_row.content_hash == content_hash
				and existing_row.embedding_model == self.embedder.model
				and existing_row.embedding_dim == self.embedder.output_dimensionality
			):
				unchanged += 1
				continue

			to_embed_rows.append(metadata)

		embed_start = perf_counter()
		embeddings = self.embedder.embed_documents([row["embed_text"] for row in to_embed_rows])
		logger.info(
			"router.refresh_index embeddings elapsed=%.3fs rows=%d",
			perf_counter() - embed_start,
			len(to_embed_rows),
		)

		for i, row in enumerate(to_embed_rows):
			self.store.upsert_agent(
				name=row["name"],
				description=row["description"],
				yaml_path=row["yaml_path"],
				embedding=embeddings[i],
				embedding_model=self.embedder.model,
				content_hash=row["content_hash"],
			)

		self.store.delete_missing(indexed_names)
		logger.info(
			"router.refresh_index done elapsed=%.3fs new=%d unchanged=%d",
			perf_counter() - start,
			len(to_embed_rows),
			unchanged,
		)

	def route(self, query: str) -> agent_module.Agent:
		agent, _metadata = self.route_with_metadata(query)
		return agent

	def route_with_metadata(self, query: str) -> tuple[agent_module.Agent, dict[str, str | float]]:
		"""Return the routed agent plus the match metadata."""
		start = perf_counter()
		embed_start = perf_counter()
		query_embedding = self.embedder.embed_query(query)
		logger.info("router.route query embedding elapsed=%.3fs", perf_counter() - embed_start)
		match_start = perf_counter()
		matches = self.store.match(query_embedding, limit=1)
		logger.info("router.route store match elapsed=%.3fs", perf_counter() - match_start)
		if not matches:
			raise ValueError("No agents are indexed")

		record, score = matches[0]
		if score < self.min_score:
			raise ValueError(
				f"No agent met routing threshold ({score:.3f} < {self.min_score:.3f})"
			)

		load_start = perf_counter()
		agent = agent_module.Agent.from_yaml(record.yaml_path)
		logger.info("router.route agent load elapsed=%.3fs name=%s", perf_counter() - load_start, record.name)
		logger.info("router.route done elapsed=%.3fs score=%.3f", perf_counter() - start, score)

		return agent, {
			"name": record.name,
			"description": record.description,
			"yaml_path": record.yaml_path,
			"score": score,
		}


def route_agent(
	query: str,
	agents_dir: Optional[str] = None,
	db_path: Optional[str] = None,
	min_score: float = 0.5,
) -> agent_module.Agent:
	"""Convenience function for routing a query to the best agent."""
	router = AgentRouter(agents_dir=agents_dir, db_path=db_path, min_score=min_score)
	return router.route(query)
