import logging
import yaml
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolDefinition:
    def __init__(self, data: dict[str, Any]):
        self.name = data.get("name")
        self.description = data.get("description")
        self.input_schema = data.get("input_schema", {})
        self.implementation = data.get("implementation", {})

    def get_handler(self) -> Optional[str]:
        return self.implementation.get("handler")


def get_tool(tool_name: str, tools_dir: Optional[str] = None) -> Optional[ToolDefinition]:
    """Return ToolDefinition for tool_name, or None if not found."""
    if tools_dir is None:
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
    else:
        tools_dir = Path(tools_dir)

    if not tools_dir.exists():
        raise FileNotFoundError(f"Tools directory not found at {tools_dir}")

    for yaml_file in tools_dir.glob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                tool_data = yaml.safe_load(f)
            if tool_data and tool_data.get("name") == tool_name:
                return ToolDefinition(tool_data)
        except yaml.YAMLError as e:
            logger.warning("Failed to parse %s: %s", yaml_file, e)

    return None
