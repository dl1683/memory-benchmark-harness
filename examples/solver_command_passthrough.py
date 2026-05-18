from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.loads(sys.stdin.read())
    memory_response = request.get("memory_response", {})
    response = {
        "prediction": memory_response.get("prediction"),
        "retrieved_context": {"mode": "passthrough"},
        "metadata": {"solver": "passthrough"},
    }
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

