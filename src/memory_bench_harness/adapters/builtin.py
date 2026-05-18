from __future__ import annotations

from memory_bench_harness.types import AdapterRequest, AdapterResponse, ObservationRequest


class NullAdapter:
    name = "null"

    async def answer(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(
            prediction="",
            retrieved_context=None,
            metadata={"history_length": request.history_length},
        )

    async def observe(self, request: ObservationRequest) -> None:
        return None


class OracleAdapter:
    name = "oracle"

    async def answer(self, request: AdapterRequest) -> AdapterResponse:
        if not request.allow_oracle:
            return AdapterResponse(
                prediction="",
                retrieved_context=None,
                metadata={"blocked": "oracle requires allow_oracle=true"},
            )
        return AdapterResponse(
            prediction=request.turn.expected_answer,
            retrieved_context="ground_truth",
            metadata={"oracle": True},
        )

    async def observe(self, request: ObservationRequest) -> None:
        return None
