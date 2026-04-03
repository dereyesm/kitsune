"""Multi-gate task router with escalation logic."""

import re

from kitsune.graph.state import KitsuneState

EXPLAIN_PATTERNS = re.compile(
    r"\b(explain|what does|how does|what is|describe|walk me through)\b", re.IGNORECASE
)

# Gate 1: Categories that ALWAYS escalate
SECURITY_PATTERNS = re.compile(
    r"\b(security|vulnerab|auth bypass|SQL injection|code injection"
    r"|XSS|CSRF|exploit|CVE|pentest)\b",
    re.IGNORECASE,
)
ARCHITECTURE_PATTERNS = re.compile(
    r"\b(architect|microservice|system design|infrastructure|scalab|redesign|migrat)\b",
    re.IGNORECASE,
)

# Gate 2: Tasks that always escalate
COMPLEX_TASK_PATTERNS = re.compile(
    r"\b(refactor|redesign|migrate|rewrite from scratch|optimize performance)\b", re.IGNORECASE
)

# Token estimate: ~0.75 tokens per word (rough heuristic)
TOKEN_THRESHOLD = 1500


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


def _check_escalation(user_input: str, code_context: str) -> str | None:
    """Returns escalation reason or None if task should stay local."""
    # Gate 1: Security — always escalate
    if SECURITY_PATTERNS.search(user_input):
        return "security analysis requires a larger model"

    # Gate 2: Architecture — always escalate
    if ARCHITECTURE_PATTERNS.search(user_input):
        return "architecture/design tasks need deeper reasoning"

    # Gate 3: Complex tasks — always escalate
    if COMPLEX_TASK_PATTERNS.search(user_input):
        return "refactoring/migration tasks are too complex for a 1.5B model"

    # Gate 4: Token budget — escalate if input is too large
    total_tokens = _estimate_tokens(user_input + code_context)
    if total_tokens > TOKEN_THRESHOLD:
        return f"input too large ({total_tokens} est. tokens > {TOKEN_THRESHOLD} threshold)"

    return None


def route(state: KitsuneState) -> KitsuneState:
    user_input = state["user_input"]
    code_context = state.get("code_context", "")

    # Check escalation gates first
    reason = _check_escalation(user_input, code_context)
    if reason:
        return {**state, "task_type": "fallback", "escalation_reason": reason}

    # Classify local task
    if EXPLAIN_PATTERNS.search(user_input) or (not user_input.strip() and code_context):
        task_type = "explain"
    else:
        task_type = "ask"

    return {**state, "task_type": task_type, "escalation_reason": ""}


def get_next_node(state: KitsuneState) -> str:
    return state["task_type"]
