from __future__ import annotations

import uuid
from typing import Any

import pytest

from memory_bench_harness.adapters.mapu import MapUAdapter
from memory_bench_harness.runner import run_benchmarks
from memory_bench_harness.types import (
    AdapterRequest,
    AdapterResponse,
    ObservationRequest,
    Scenario,
    Turn,
)


def _toy_loader(limit: int = 0) -> list[Scenario]:
    return [
        Scenario(
            benchmark="toy",
            config="contract",
            scenario_id="toy:1",
            seed_context={"profile": "seed"},
            sessions=[
                Turn(turn_index=0, prompt="stored answer", expected_answer="stored answer"),
                Turn(turn_index=1, prompt="second", expected_answer="stored answer"),
            ],
        )
    ]


class ContractAdapter:
    name = "contract"

    def __init__(self) -> None:
        self.observed_answers: list[Any] = []

    async def answer(self, request: AdapterRequest) -> AdapterResponse:
        assert request.turn.expected_answer is None
        return AdapterResponse(prediction="")

    async def observe(self, request: ObservationRequest) -> None:
        self.observed_answers.append(request.actual_answer)


class FakeMapUAdapter(MapUAdapter):
    def __init__(self) -> None:
        super().__init__(base_url="http://mapu.test", run_id="test")
        self.corpus_id = str(uuid.uuid4())
        self.documents: list[str] = []
        self.feedback: list[dict[str, Any]] = []

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if method == "POST" and path == "/corpora":
            return {"id": self.corpus_id}
        if method == "POST" and path.endswith("/documents"):
            self.documents.append(str(kwargs["json"]["content"]))
            return {
                "document_id": str(uuid.uuid4()),
                "expression_id": str(uuid.uuid4()),
                "spans": 1,
                "chunks": 1,
                "embeddings": 0,
                "propositions": 0,
            }
        if method == "POST" and path.endswith("/query"):
            return {
                "synthesis": self._last_observed_answer(),
                "hits": [],
                "chunk_hits": [],
                "gaps": [],
                "epistemic_status": "known" if self._last_observed_answer() else "unknown",
                "tier_used": "DIRECT",
            }
        if method == "GET" and path.endswith("/resume"):
            return {
                "continuity_frontier": {
                    "frontier_completeness": "complete",
                    "missing_gap_contract_count": 0,
                }
            }
        if method == "POST" and path.endswith("/activity/feedback"):
            self.feedback.append(dict(kwargs["json"]))
            return {"success": True, "event_id": str(uuid.uuid4())}
        raise AssertionError(f"Unexpected MapU request: {method} {path}")

    def _last_observed_answer(self) -> str:
        for content in reversed(self.documents):
            marker = "Prompt:\n"
            if marker not in content:
                continue
            raw = content.split(marker, 1)[1].split("\n\nBackground:", 1)[0].strip()
            return raw
        return ""


@pytest.mark.asyncio
async def test_runner_redacts_expected_answer_before_real_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memory_bench_harness.runner as runner

    monkeypatch.setitem(runner.LOADERS, "toy", _toy_loader)
    adapter = ContractAdapter()
    report = await run_benchmarks(["toy"], adapter=adapter, limit=0)

    assert report["turn_count"] == 2
    assert adapter.observed_answers == ["stored answer", "stored answer"]


@pytest.mark.asyncio
async def test_mapu_adapter_observes_turns_then_queries_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memory_bench_harness.runner as runner

    monkeypatch.setitem(runner.LOADERS, "toy", _toy_loader)
    adapter = FakeMapUAdapter()
    report = await run_benchmarks(["toy"], adapter=adapter, limit=0)

    assert report["turn_count"] == 2
    assert report["summary"]["evaluated"] == 2
    assert report["summary"]["correct"] == 1
    assert len(adapter.documents) >= 3
    assert all("Ground truth" not in content for content in adapter.documents)
    assert len(adapter.feedback) == 2
