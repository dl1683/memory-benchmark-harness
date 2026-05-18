from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Turn:
    turn_index: int
    prompt: str
    expected_answer: Any
    background: Any | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    benchmark: str
    scenario_id: str
    sessions: list[Turn]
    config: str | None = None
    category: str | None = None
    seed_context: Any | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterRequest:
    benchmark: str
    config: str | None
    scenario_id: str
    seed_context: Any | None
    turn: Turn
    history_length: int = 0
    allow_oracle: bool = False


@dataclass(frozen=True)
class ObservationRequest:
    benchmark: str
    config: str | None
    scenario_id: str
    seed_context: Any | None
    turn: Turn
    prediction: Any
    actual_answer: Any


@dataclass(frozen=True)
class AdapterResponse:
    prediction: Any
    retrieved_context: Any | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class TurnResult:
    benchmark: str
    config: str | None
    scenario_id: str
    turn_index: int
    prediction: Any
    expected_answer: Any
    exact_match: bool
    retrieved_context: Any | None
    latency_ms: float
    error: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    semantic_match: bool | None = None


class MemoryAdapter(Protocol):
    name: str

    async def answer(self, request: AdapterRequest) -> AdapterResponse:
        """Return an answer for one benchmark turn."""

    async def observe(self, request: ObservationRequest) -> None:
        """Persist a completed turn for future benchmark turns."""


class Judge(Protocol):
    name: str

    async def judge(
        self,
        request: AdapterRequest,
        prediction: Any,
        expected: Any,
    ) -> tuple[bool, JsonObject]:
        """Judge semantic equivalence between a prediction and expected answer."""
