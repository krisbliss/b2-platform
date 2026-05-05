import os
import yaml
from pathlib import Path
from typing import Optional


def get_skill_prompt(skill_name: str, skills_dir: Optional[str] = None) -> Optional[str]:
    """
    Pull a skill by name from YAML files and return its prompt.
    
    Args:
        skill_name: Name of the skill to retrieve
        skills_dir: Path to the skills directory. Defaults to 'skills/' relative to project root.
    
    Returns:
        The prompt string from the skill's YAML file, or None if skill not found.
    
    Raises:
        FileNotFoundError: If the skills directory does not exist.
        ValueError: If the skill exists but has no prompt field.
    """
    if skills_dir is None:
        # Find the project root and use skills/ directory
        project_root = Path(__file__).parent.parent.parent
        skills_dir = project_root / "skills"
    else:
        skills_dir = Path(skills_dir)
    
    if not skills_dir.exists():
        raise FileNotFoundError(f"Skills directory not found at {skills_dir}")
    
    # Search for the skill file matching the name
    for yaml_file in skills_dir.glob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                skill_data = yaml.safe_load(f)
            
            # Check if this is the skill we're looking for
            if skill_data and skill_data.get("name") == skill_name:
                if "prompt" not in skill_data:
                    raise ValueError(f"Skill '{skill_name}' found but has no 'prompt' field")
                return skill_data["prompt"]
        
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse {yaml_file}: {e}")
            continue
    
    return None
