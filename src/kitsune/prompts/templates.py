"""Static templates (non-LLM). Skill-based prompts are in loader.py."""

FALLBACK_MSG = """\
This task exceeds what a 1.5B local model can handle reliably.
**Reason**: {reason}

Suggested approach — use Claude Code:

  {prompt}

Context file: {file_path}"""
