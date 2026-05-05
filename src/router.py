from pathlib import Path
from typing import Dict, List, Optional, Tuple

from orchestrator import agent as agent_module

import sqllite


class AgentRouter:
	"""Index agent YAML files and route queries to the best matching agent."""

	def __init__(self, agents_dir: Optional[str] = None, db_path: Optional[str] = None):
		self.agents_dir = self._agents_dir(agents_dir)
		self.store = sqllite.AgentIndexStore(db_path=db_path)
		self.refresh_index()

	def _agents_dir(self, agents_dir: Optional[str] = None) -> Path:
		if agents_dir is None:
			project_root = Path(__file__).parent.parent
			return project_root / "agents"
		return Path(agents_dir)

	@staticmethod
	def _load_yaml_metadata(yaml_path: Path) -> Dict[str, str]:
		yaml_module = agent_module._load_yaml_module()

		with open(yaml_path, "r") as handle:
			agent_data = yaml_module.safe_load(handle) or {}

		name = agent_data.get("name")
		system_prompt = agent_data.get("system_prompt", "")
		description = agent_data.get("description") or (system_prompt.splitlines()[0] if system_prompt else "")

		if not name:
			raise ValueError(f"Agent file '{yaml_path}' is missing required 'name' field")

		return {
			"name": name,
			"description": description,
			"yaml_path": str(yaml_path),
		}

	def refresh_index(self) -> None:
		if not self.agents_dir.exists():
			raise FileNotFoundError(f"Agents directory not found at {self.agents_dir}")

		indexed_names: List[str] = []
		for yaml_path in self.agents_dir.glob("*.yaml"):
			metadata = self._load_yaml_metadata(yaml_path)
			indexed_names.append(metadata["name"])
			self.store.upsert_agent(
				name=metadata["name"],
				description=metadata["description"],
				yaml_path=metadata["yaml_path"],
			)

		self.store.delete_missing(indexed_names)

	def route(self, query: str) -> agent_module.Agent:
		matches = self.store.match(query, limit=1)
		if not matches:
			raise ValueError("No agents are indexed")

		record, _score = matches[0]

		return agent_module.Agent.from_yaml(record.yaml_path)

	def route_with_metadata(self, query: str) -> Tuple[agent_module.Agent, Dict[str, str | float]]:
		"""Return the routed agent plus the match metadata."""
		matches = self.store.match(query, limit=1)
		if not matches:
			raise ValueError("No agents are indexed")

		record, score = matches[0]

		return agent_module.Agent.from_yaml(record.yaml_path), {
			"name": record.name,
			"description": record.description,
			"yaml_path": record.yaml_path,
			"score": score,
		}


def route_agent(query: str, agents_dir: Optional[str] = None, db_path: Optional[str] = None) -> agent_module.Agent:
	"""Convenience function for routing a query to the best agent."""
	router = AgentRouter(agents_dir=agents_dir, db_path=db_path)
	return router.route(query)
