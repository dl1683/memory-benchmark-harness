from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.loads(sys.stdin.read())
    event = request.get("event")
    if event == "observe":
        print(json.dumps({"ok": True}))
        return 0

    response = {
        "prediction": "",
        "retrieved_context": f"history_length={request.get('history_length', 0)}",
        "metadata": {
            "adapter": "echo",
            "history_length": request.get("history_length", 0),
        },
    }
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
