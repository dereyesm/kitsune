"""Skill-based prompt loader. Injects language-specific knowledge into system prompts."""

from functools import lru_cache
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
}

TASK_INSTRUCTIONS = {
    "explain": "Task: Explain what this code does. Lead with purpose, then key components.",
    "ask": "Task: Answer the user's question about this code. Be direct and specific.",
}


@lru_cache(maxsize=16)
def _load_skill(name: str) -> str:
    path = SKILLS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def detect_language(file_path: str) -> str | None:
    if not file_path or file_path == "<stdin>":
        return None
    ext = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(ext)


def build_system_prompt(task_type: str, file_path: str = "") -> str:
    parts = [_load_skill("base")]

    lang = detect_language(file_path)
    if lang:
        skill = _load_skill(lang)
        if skill:
            parts.append(skill)

    instruction = TASK_INSTRUCTIONS.get(task_type, "")
    if instruction:
        parts.append(instruction)

    return "\n\n".join(parts)
