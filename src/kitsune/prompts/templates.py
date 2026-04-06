"""Static templates (non-LLM). Skill-based prompts are in loader.py."""

FALLBACK_MSG = """\
This task exceeds what the current local model can handle reliably.
**Reason**: {reason}

**Available escalation tiers** (choose one and re-run):

{tiers}

Or hand off the prompt below to an external assistant:

  {prompt}

Context file: {file_path}"""

#: Shown when no remote providers have keys configured — pure local fallback.
FALLBACK_TIERS_LOCAL_ONLY = """\
- **Local (tier up)**: export `KITSUNE_MODEL_TIER=medium` for Qwen3.5-4B
  or `large` for Qwen3.5-9B, then restart your model server.
- **Free remote (opt-in)**: set `OPENROUTER_API_KEY` and
  `KITSUNE_PROVIDER=openrouter` to use Qwen3-Coder 480B or Nemotron 3 Super
  at $0/token (rate limits apply, code leaves the machine).
- **Paid fallback**: use Claude Code as a last resort."""
