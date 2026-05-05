from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import llm, prompts, tools


@dataclass
class AgentDefinition:
    """Represents an agent definition loaded from YAML."""

    name: str
    system_prompt: str
    skills: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    provider: Dict[str, Any] = field(default_factory=dict)
    behavior: Dict[str, Any] = field(default_factory=dict)


def _agents_dir(agents_dir: Optional[str] = None) -> Path:
    if agents_dir is None:
        project_root = Path(__file__).parent.parent.parent
        return project_root / "agents"
    return Path(agents_dir)


def _load_yaml_module():
    try:
        return importlib.import_module("yaml")
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to load agent definitions. Install it to use Agent YAML files."
        ) from exc


def _load_agent_definition_from_file(agent_file: Path) -> AgentDefinition:
    yaml_module = _load_yaml_module()

    with open(agent_file, "r") as handle:
        agent_data = yaml_module.safe_load(handle) or {}

    name = agent_data.get("name")
    system_prompt = agent_data.get("system_prompt")

    if not name:
        raise ValueError(f"Agent file '{agent_file}' is missing required 'name' field")
    if not system_prompt:
        raise ValueError(f"Agent file '{agent_file}' is missing required 'system_prompt' field")

    return AgentDefinition(
        name=name,
        system_prompt=system_prompt,
        skills=agent_data.get("skills", []) or [],
        tools=agent_data.get("tools", []) or [],
        provider=agent_data.get("provider", {}) or {},
        behavior=agent_data.get("behavior", {}) or {},
    )


def get_agent(agent_name: str, agents_dir: Optional[str] = None) -> Optional[AgentDefinition]:
    """
    Pull an agent by name from YAML files and return its definition.
    """
    directory = _agents_dir(agents_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Agents directory not found at {directory}")

    yaml_module = _load_yaml_module()

    for yaml_file in directory.glob("*.yaml"):
        try:
            agent_definition = _load_agent_definition_from_file(yaml_file)
            if agent_definition.name == agent_name:
                return agent_definition
        except yaml_module.YAMLError as exc:
            print(f"Warning: Failed to parse {yaml_file}: {exc}")

    return None


class Agent:
    """An AI agent configured from an agent YAML file."""

    def __init__(
        self,
        agent_name: Optional[str] = None,
        yaml_path: Optional[str] = None,
        agents_dir: Optional[str] = None,
    ):
        """
        Initialize an Agent from YAML.

        Args:
            agent_name: Name of the agent to load from the agents directory.
            yaml_path: Explicit path to an agent YAML file.
            agents_dir: Optional override for the agents directory.

        Raises:
            ValueError: If neither agent_name nor yaml_path is provided.
            FileNotFoundError: If the YAML file or agents directory cannot be found.
        """
        if yaml_path is None and agent_name is None:
            raise ValueError("Either agent_name or yaml_path must be provided")

        if yaml_path is not None:
            agent_definition = _load_agent_definition_from_file(Path(yaml_path))
        else:
            agent_definition = get_agent(agent_name or "", agents_dir)
            if agent_definition is None:
                raise ValueError(f"Agent '{agent_name}' not found")

        self.definition = agent_definition
        self.name = agent_definition.name
        self.base_prompt = agent_definition.system_prompt
        self.skill_names = list(agent_definition.skills)
        self.tool_names = list(agent_definition.tools)
        self.llm_provider = agent_definition.provider.get("type")
        self.llm_config = {
            key: value for key, value in agent_definition.provider.items() if key != "type"
        }
        self.behavior = agent_definition.behavior

        self.enhanced_prompt = (
            prompts.add_skills_to_prompt(self.base_prompt, self.skill_names)
            if self.skill_names
            else self.base_prompt
        )

        self.available_tools = []
        for tool_name in self.tool_names:
            tool_def = tools.get_tool(tool_name)
            if tool_def is None:
                raise ValueError(f"Tool '{tool_name}' not found")
            self.available_tools.append(tool_def)

    @classmethod
    def from_name(cls, agent_name: str, agents_dir: Optional[str] = None) -> "Agent":
        """Construct an Agent from an agent name."""
        return cls(agent_name=agent_name, agents_dir=agents_dir)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Agent":
        """Construct an Agent from an explicit YAML path."""
        return cls(yaml_path=yaml_path)

    def call(self, user_message: str, **kwargs) -> str:
        """
        Call the LLM with a user message.

        Args:
            user_message: The user's message to process
            **kwargs: Additional arguments to pass to the LLM provider

        Returns:
            The LLM's response
        """
        full_prompt = f"{self.enhanced_prompt}\n\nUser: {user_message}"
        tool_dicts = [tool.to_dict() for tool in self.available_tools]
        call_config = {**self.llm_config, **kwargs}

        return llm.call_llm(
            full_prompt,
            provider_name=self.llm_provider,
            tools=tool_dicts if tool_dicts else None,
            **call_config,
        )

    def get_tool(self, tool_name: str) -> Optional[tools.ToolDefinition]:
        """Get a tool definition by name from the agent's available tools."""
        for tool in self.available_tools:
            if tool.name == tool_name:
                return tool
        return None

    def add_tool(self, tool_name: str) -> None:
        """Add a tool to the agent's available tools."""
        if tool_name not in self.tool_names:
            tool_def = tools.get_tool(tool_name)
            if tool_def is None:
                raise ValueError(f"Tool '{tool_name}' not found")
            self.tool_names.append(tool_name)
            self.available_tools.append(tool_def)

    def add_skill(self, skill_name: str) -> None:
        """Add a skill to the agent's prompt."""
        if skill_name not in self.skill_names:
            self.skill_names.append(skill_name)
            self.enhanced_prompt = prompts.add_skills_to_prompt(
                self.base_prompt,
                self.skill_names,
            )
