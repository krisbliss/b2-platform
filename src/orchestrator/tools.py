import yaml
from pathlib import Path
from typing import Any, Optional


class ToolDefinition:
    """Represents a tool definition loaded from YAML."""
    
    def __init__(self, data: dict[str, Any]):
        self.name = data.get("name")
        self.description = data.get("description")
        self.input_schema = data.get("input_schema", {})
        self.implementation = data.get("implementation", {})

    def get_handler(self) -> Optional[str]:
        """Get the implementation handler."""
        return self.implementation.get("handler")


def get_tool(tool_name: str, tools_dir: Optional[str] = None) -> Optional[ToolDefinition]:
    """
    Pull a tool by name from YAML files and return its definition.
    
    Args:
        tool_name: Name of the tool to retrieve
        tools_dir: Path to the tools directory. Defaults to 'tools/' relative to project root.
    
    Returns:
        A ToolDefinition object, or None if tool not found.
    
    Raises:
        FileNotFoundError: If the tools directory does not exist.
        ValueError: If the tool exists but is invalid.
    """
    if tools_dir is None:
        # Find the project root and use tools/ directory
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
    else:
        tools_dir = Path(tools_dir)
    
    if not tools_dir.exists():
        raise FileNotFoundError(f"Tools directory not found at {tools_dir}")
    
    # Search for the tool file matching the name
    for yaml_file in tools_dir.glob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                tool_data = yaml.safe_load(f)
            
            # Check if this is the tool we're looking for
            if tool_data and tool_data.get("name") == tool_name:
                if not tool_data.get("name"):
                    raise ValueError(f"Tool in {yaml_file} missing required 'name' field")
                
                return ToolDefinition(tool_data)
        
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse {yaml_file}: {e}")
            continue
    
    return None
