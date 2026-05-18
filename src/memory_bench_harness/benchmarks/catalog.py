from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from memory_bench_harness.types import Scenario

ScenarioLoader = Callable[[int], list[Scenario]]


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    status: str
    source: str
    notes: str


BENCHMARKS: dict[str, BenchmarkInfo] = {
    "memoryarena": BenchmarkInfo(
        name="memoryarena",
        status="runnable",
        source="https://huggingface.co/datasets/ZexueHe/memoryarena",
        notes="Multi-session agentic memory tasks; Hugging Face test split.",
    ),
    "ama_bench": BenchmarkInfo(
        name="ama_bench",
        status="runnable",
        source="https://huggingface.co/datasets/AMA-bench/AMA-bench",
        notes="Long-horizon trajectory-memory QA; official scoring uses LLM-as-judge.",
    ),
}


EXTERNAL_BENCHMARKS: dict[str, BenchmarkInfo] = {
    "agent_memory_bench": BenchmarkInfo(
        name="agent_memory_bench",
        status="external_docker",
        source="https://github.com/s010m00n/AgentMemoryBench",
        notes="Best broad continual-agent-memory benchmark, but needs Docker task servers.",
    ),
    "memory_agent_bench": BenchmarkInfo(
        name="memory_agent_bench",
        status="external_heavy",
        source="https://github.com/hust-ai-hyz/MemoryAgentBench",
        notes="Incremental multi-turn agent memory benchmark; custom framework integration needed.",
    ),
    "longmemeval": BenchmarkInfo(
        name="longmemeval",
        status="planned",
        source="https://github.com/xiaowu0162/longmemeval",
        notes="Long-term conversational memory baseline; useful but less agentic.",
    ),
    "locomo": BenchmarkInfo(
        name="locomo",
        status="planned",
        source="https://github.com/snap-research/locomo",
        notes="Long-conversation memory benchmark; useful as a recall baseline.",
    ),
}

