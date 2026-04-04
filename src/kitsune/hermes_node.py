"""Kitsune HERMES Node — watch bus.jsonl for dispatch messages.

Kitsune acts as a HERMES node that receives dispatch messages and
executes local code tasks (explain, ask, search) autonomously.

Usage:
    uv run python -m kitsune.hermes_node

Message format (bus.jsonl):
    {"ts":"2026-04-03","src":"claude-code","dst":"kitsune","type":"dispatch",
     "msg":"explain ~/Dev/kitsune/src/kitsune/cli.py","ttl":3,"ack":[]}
"""

import json
import time
from datetime import datetime
from pathlib import Path

BUS_PATH = Path.home() / ".claude" / "sync" / "bus.jsonl"
POLL_INTERVAL = 5  # seconds


def _read_bus() -> list[dict]:
    if not BUS_PATH.exists():
        return []
    lines = BUS_PATH.read_text(encoding="utf-8").strip().split("\n")
    msgs = []
    for line in lines:
        if line.strip():
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return msgs


def _write_bus_msg(msg: dict) -> None:
    with open(BUS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n")


def _ack_message(msgs: list[dict], idx: int) -> None:
    """Add 'kitsune' to the ack array of a message."""
    if "ack" not in msgs[idx]:
        msgs[idx]["ack"] = []
    if "kitsune" not in msgs[idx]["ack"]:
        msgs[idx]["ack"].append("kitsune")
    # Rewrite entire bus (simple, not concurrent-safe — fine for single node)
    with open(BUS_PATH, "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m) + "\n")


def _process_dispatch(msg: dict) -> str | None:
    """Process a dispatch message. Returns response or None."""
    payload = msg.get("msg", "")
    if not payload:
        return None

    parts = payload.strip().split(" ", 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "explain":
        from kitsune.graph.build import app as graph_app

        p = Path(args).expanduser().resolve()
        if not p.is_file():
            return f"Error: file not found: {p}"
        code = p.read_text(encoding="utf-8", errors="replace")
        result = graph_app.invoke(
            {
                "user_input": "",
                "task_type": "explain",
                "code_context": code,
                "file_path": str(p),
                "response": "",
                "escalation_reason": "",
            }
        )
        return result["response"]

    elif command == "ask":
        from kitsune.graph.build import app as graph_app

        result = graph_app.invoke(
            {
                "user_input": args,
                "task_type": "ask",
                "code_context": "",
                "file_path": "",
                "response": "",
                "escalation_reason": "",
            }
        )
        return result["response"]

    elif command == "search":
        from kitsune.rag.bm25_backend import BM25Backend

        rag = BM25Backend()
        rag.index(".")
        results = rag.search(args, top_k=3)
        if not results:
            return "No results found."
        return "\n".join(f"{r.file_path}:{r.start_line} (score:{r.score})" for r in results)

    return f"Unknown command: {command}"


def run_node():
    """Main loop: poll bus.jsonl for kitsune-destined dispatches."""
    print(f"Kitsune HERMES node started. Watching {BUS_PATH}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print("Ctrl+C to stop.\n")

    processed_ts = set()

    while True:
        try:
            msgs = _read_bus()
            for idx, msg in enumerate(msgs):
                # Only process dispatch messages destined for kitsune
                if msg.get("type") != "dispatch":
                    continue
                dst = msg.get("dst", "")
                if dst not in ("kitsune", "*"):
                    continue
                if "kitsune" in msg.get("ack", []):
                    continue

                # Dedup by timestamp + message
                key = f"{msg.get('ts')}:{msg.get('msg')}"
                if key in processed_ts:
                    continue

                ts = datetime.now().strftime("%H:%M:%S")
                payload = msg.get("msg", "")[:60]
                print(f"[{ts}] Processing: {payload}")

                response = _process_dispatch(msg)
                if response:
                    # Write response to bus
                    _write_bus_msg(
                        {
                            "ts": datetime.now().strftime("%Y-%m-%d"),
                            "src": "kitsune",
                            "dst": msg.get("src", "*"),
                            "type": "data_cross",
                            "msg": response[:500],  # cap response length for bus
                            "ttl": 3,
                            "ack": [],
                        }
                    )
                    print(f"  -> Response sent ({len(response)} chars)")

                # ACK the message
                _ack_message(msgs, idx)
                processed_ts.add(key)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\nKitsune HERMES node stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_node()
