from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import asdict
from typing import Any

from memory_bench_harness.types import AdapterRequest, AdapterResponse, ObservationRequest


class CommandAdapter:
    def __init__(self, command: str) -> None:
        self.command = command
        self.name = f"command:{command}"

    async def _invoke(self, payload: dict[str, Any], require_response: bool) -> dict[str, Any]:
        argv = shlex.split(self.command)
        if not argv:
            raise ValueError("Adapter command cannot be empty.")
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        stdout, stderr = await process.communicate(encoded)
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())
        if not require_response and not stdout.strip():
            return {}
        try:
            decoded = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Adapter command returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Adapter command must return a JSON object.")
        return decoded

    async def answer(self, request: AdapterRequest) -> AdapterResponse:
        payload = {"event": "answer", **asdict(request)}
        decoded = await self._invoke(payload, require_response=True)
        return AdapterResponse(
            prediction=decoded.get("prediction"),
            retrieved_context=decoded.get("retrieved_context"),
            metadata=decoded.get("metadata") if isinstance(decoded.get("metadata"), dict) else {},
        )

    async def observe(self, request: ObservationRequest) -> None:
        payload = {"event": "observe", **asdict(request)}
        await self._invoke(payload, require_response=False)
