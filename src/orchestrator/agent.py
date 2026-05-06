from dataclasses import dataclass, field
import importlib
from inspect import Parameter, Signature
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, Optional, Sequence

from pydantic import Field
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.google import GoogleModel
from dotenv import load_dotenv

from . import prompts, tools

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """Represents an agent definition loaded from YAML."""

    name: str
    system_prompt: str
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    provider: dict[str, Any] = field(default_factory=dict)
    behavior: dict[str, Any] = field(default_factory=dict)


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


class Agent:
    """An immutable AI agent configured from an agent YAML file."""

    def __init__(
        self,
        yaml_path: str,
    ):
        """
        Initialize an Agent from YAML.

        Args:
            yaml_path: Path to an agent YAML file.

        Raises:
            FileNotFoundError: If the YAML file cannot be found.
        """
        start = perf_counter()
        load_start = perf_counter()
        agent_definition = _load_agent_definition_from_file(Path(yaml_path))
        logger.info("agent.load_yaml elapsed=%.3fs name=%s", perf_counter() - load_start, agent_definition.name)

        self.definition = agent_definition
        self.name = agent_definition.name
        self.base_prompt = agent_definition.system_prompt
        self.skill_names = list(agent_definition.skills)
        self.tool_names = list(agent_definition.tools)
        self.behavior = agent_definition.behavior
        provider_type = str(agent_definition.provider.get("type", "")).lower()
        model_name = agent_definition.provider.get("model")
        if provider_type != "gemini":
            raise ValueError(f"Unsupported provider type '{provider_type}'. Only 'gemini' is supported.")
        if not model_name:
            raise ValueError(f"Agent '{self.name}' is missing provider.model")
        self.model_settings = self._model_settings_from_provider(agent_definition.provider, str(model_name))

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to run Gemini agents.")
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = api_key

        enhanced_prompt = (
            prompts.add_skills_to_prompt(self.base_prompt, self.skill_names)
            if self.skill_names
            else self.base_prompt
        )

        model_start = perf_counter()
        model = GoogleModel(model_name)
        self.pydantic_ai_agent = PydanticAgent(
            model=model,
            system_prompt=enhanced_prompt,
            name=self.name,
        )
        logger.info(
            "agent.model_init elapsed=%.3fs name=%s model=%s settings=%s",
            perf_counter() - model_start,
            self.name,
            model_name,
            sorted(self.model_settings.keys()),
        )

        tools_start = perf_counter()
        for tool_name in self.tool_names:
            self._register_tool(tool_name)
        logger.info(
            "agent.tools_registered elapsed=%.3fs name=%s count=%d",
            perf_counter() - tools_start,
            self.name,
            len(self.tool_names),
        )
        logger.info("agent.init done elapsed=%.3fs name=%s", perf_counter() - start, self.name)

    @staticmethod
    def _resolve_handler(handler_path: str) -> Any:
        module_path, function_name = handler_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, function_name)

    @staticmethod
    def _model_settings_from_provider(provider: dict[str, Any], model_name: str) -> dict[str, Any]:
        settings = dict(provider.get("settings", {}) or {})
        thinking_budget = settings.pop("thinking_budget", None)
        if thinking_budget is not None and "google_thinking_config" not in settings:
            if model_name.startswith("gemini-3") and int(thinking_budget) == 0:
                settings["google_thinking_config"] = {"thinking_level": "MINIMAL"}
            else:
                settings["google_thinking_config"] = {"thinking_budget": int(thinking_budget)}
        return settings

    @classmethod
    def _json_schema_to_python_type(cls, schema: dict[str, Any]) -> Any:
        enum_values = schema.get("enum")
        if enum_values:
            return Literal[tuple(enum_values)]

        json_type = schema.get("type", "string")
        if isinstance(json_type, list):
            non_null_types = [item for item in json_type if item != "null"]
            json_type = non_null_types[0] if non_null_types else "string"

        if json_type == "array":
            item_type = cls._json_schema_to_python_type(schema.get("items", {}) or {})
            return list[item_type]

        if json_type == "object":
            return dict[str, Any]

        return {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }.get(str(json_type), str)

    def _apply_tool_signature(self, tool_func: Any, tool_def: tools.ToolDefinition) -> None:
        schema = tool_def.input_schema or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        parameters: list[Parameter] = []
        annotations: dict[str, Any] = {}
        for field_name, prop_schema in properties.items():
            prop_schema = prop_schema or {}
            description = prop_schema.get("description", "")
            annotation = self._json_schema_to_python_type(prop_schema)
            if description:
                annotation = Annotated[annotation, Field(description=description)]

            if field_name in required:
                default = Parameter.empty
            else:
                default = prop_schema.get("default", None)
                annotation = Optional[annotation]

            annotations[field_name] = annotation
            parameters.append(
                Parameter(
                    field_name,
                    Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )

        tool_func.__signature__ = Signature(parameters=parameters)
        tool_func.__annotations__ = annotations

    def _register_tool(self, tool_name: str) -> None:
        start = perf_counter()
        tool_def = tools.get_tool(tool_name)
        if tool_def is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        handler_path = tool_def.get_handler()
        if not handler_path:
            raise ValueError(f"Tool '{tool_name}' has no implementation.handler")

        handler = self._resolve_handler(handler_path)

        def _tool_fn(**kwargs: Any) -> Any:
            tool_start = perf_counter()
            args = {key: value for key, value in kwargs.items() if value is not None}
            logger.info("agent.tool_call start tool=%s args=%s", tool_def.name, args)
            try:
                result = handler(**args)
            except Exception:
                logger.exception("agent.tool_call failed tool=%s elapsed=%.3fs", tool_def.name, perf_counter() - tool_start)
                raise
            logger.info(
                "agent.tool_call done tool=%s elapsed=%.3fs result=%s",
                tool_def.name,
                perf_counter() - tool_start,
                result,
            )
            return result

        _tool_fn.__name__ = f"tool_{tool_def.name}"
        _tool_fn.__doc__ = tool_def.description or f"Tool: {tool_def.name}"
        self._apply_tool_signature(_tool_fn, tool_def)

        self.pydantic_ai_agent.tool_plain(
            _tool_fn,
            name=tool_def.name,
            description=tool_def.description,
        )
        logger.info("agent.tool_registered elapsed=%.3fs tool=%s", perf_counter() - start, tool_name)

    def run(
        self,
        user_message: str,
        message_history: Optional[Sequence[ModelMessage]] = None,
    ) -> Any:
        start = perf_counter()
        logger.info("agent.run start name=%s history=%d", self.name, len(message_history or []))
        result = self.pydantic_ai_agent.run_sync(
            user_message,
            message_history=message_history,
            model_settings=self.model_settings,
        )
        logger.info("agent.run done elapsed=%.3fs name=%s", perf_counter() - start, self.name)
        return result

    def run_stream(
        self,
        user_message: str,
        message_history: Optional[Sequence[ModelMessage]] = None,
    ) -> Any:
        start = perf_counter()
        logger.info("agent.run_stream start name=%s history=%d", self.name, len(message_history or []))
        streamed = self.pydantic_ai_agent.run_stream_sync(
            user_message,
            message_history=message_history,
            model_settings=self.model_settings,
        )
        logger.info("agent.run_stream returned elapsed=%.3fs name=%s", perf_counter() - start, self.name)
        return streamed

    def get_tool(self, tool_name: str) -> Optional[tools.ToolDefinition]:
        """Get a tool definition by name from the agent's declared tools."""
        if tool_name not in self.tool_names:
            return None
        return tools.get_tool(tool_name)
