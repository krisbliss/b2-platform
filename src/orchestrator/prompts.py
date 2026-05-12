from typing import Optional
from . import skills


def add_skills_to_prompt(prompt: str, skill_names: list[str], skills_dir: Optional[str] = None) -> str:
    """
    Add skills to a prompt by retrieving their prompts and appending them.
    
    Args:
        prompt: The base prompt to which skills will be added
        skill_names: List of skill names to include
        skills_dir: Optional path to the skills directory
    
    Returns:
        The enhanced prompt with skills integrated
    
    Raises:
        ValueError: If any skill is not found
    """
    if not skill_names:
        return prompt
    
    skill_prompts = []
    
    for skill_name in skill_names:
        skill_prompt = skills.get_skill_prompt(skill_name, skills_dir)
        
        if skill_prompt is None:
            raise ValueError(f"Skill '{skill_name}' not found")
        
        skill_prompts.append(skill_prompt)
    
    # Combine the base prompt with skill prompts
    combined_prompt = prompt + "\n\n" + "---\n\n".join(
        [f"SKILL: {skill_name.upper()}\n{skill_prompt}" 
         for skill_name, skill_prompt in zip(skill_names, skill_prompts)]
    )
    
    return combined_prompt
