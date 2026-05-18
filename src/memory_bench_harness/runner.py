from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memory_bench_harness.benchmarks.ama_bench import load_ama_bench
from memory_bench_harness.benchmarks.memoryarena import load_memoryarena
from memory_bench_harness.scoring import exact_match, summarize_results
from memory_bench_harness.types import (
    AdapterRequest,
    Judge,
    MemoryAdapter,
    ObservationRequest,
    Scenario,
    TurnResult,
)

LOADERS = {
    "memoryarena": load_memoryarena,
    "ama_bench": load_ama_bench,
}


def load_scenarios(benchmarks: list[str], limit: int = 0, offset: int = 0) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for benchmark in benchmarks:
        if benchmark not in LOADERS:
            raise ValueError(f"Unsupported runnable benchmark: {benchmark}")
        scenarios.extend(LOADERS[benchmark](limit=limit, offset=offset))
    return scenarios


async def _run_turn(
    adapter: MemoryAdapter,
    scenario: Scenario,
    turn_index: int,
    allow_oracle: bool,
    semaphore: asyncio.Semaphore,
    judge: Judge | None = None,
) -> TurnResult:
    turn = scenario.sessions[turn_index]
    answer_turn = turn if allow_oracle else replace(turn, expected_answer=None)
    request = AdapterRequest(
        benchmark=scenario.benchmark,
        config=scenario.config,
        scenario_id=scenario.scenario_id,
        seed_context=scenario.seed_context,
        turn=answer_turn,
        history_length=turn_index,
        allow_oracle=allow_oracle,
    )
    start = time.perf_counter()
    async with semaphore:
        try:
            response = await adapter.answer(request)
            latency_ms = (time.perf_counter() - start) * 1000
            is_exact = exact_match(response.prediction, turn.expected_answer)
            semantic_match = None
            metadata = dict(response.metadata)
            metadata.setdefault("turn_metadata", dict(turn.metadata))
            if judge is not None and not is_exact and turn.expected_answer is not None:
                try:
                    semantic_match, judge_metadata = await judge.judge(
                        request,
                        response.prediction,
                        turn.expected_answer,
                    )
                    metadata["judge_metadata"] = judge_metadata
                except Exception as exc:
                    metadata["judge_error"] = str(exc)
            return TurnResult(
                benchmark=scenario.benchmark,
                config=scenario.config,
                scenario_id=scenario.scenario_id,
                turn_index=turn.turn_index,
                prediction=response.prediction,
                expected_answer=turn.expected_answer,
                exact_match=is_exact,
                semantic_match=semantic_match,
                retrieved_context=response.retrieved_context,
                latency_ms=latency_ms,
                metadata=metadata,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return TurnResult(
                benchmark=scenario.benchmark,
                config=scenario.config,
                scenario_id=scenario.scenario_id,
                turn_index=turn.turn_index,
                prediction=None,
                expected_answer=turn.expected_answer,
                exact_match=False,
                retrieved_context=None,
                latency_ms=latency_ms,
                error=str(exc),
                metadata={"turn_metadata": dict(turn.metadata)},
            )


async def _run_scenario(
    adapter: MemoryAdapter,
    scenario: Scenario,
    allow_oracle: bool,
    semaphore: asyncio.Semaphore,
    max_turns_per_scenario: int = 0,
    judge: Judge | None = None,
) -> list[TurnResult]:
    results: list[TurnResult] = []
    turn_count = len(scenario.sessions)
    if max_turns_per_scenario > 0:
        turn_count = min(turn_count, max_turns_per_scenario)
    for turn_index in range(turn_count):
        result = await _run_turn(
            adapter,
            scenario,
            turn_index,
            allow_oracle,
            semaphore,
            judge=judge,
        )
        results.append(result)
        if result.error is not None:
            continue
        turn = scenario.sessions[turn_index]
        try:
            async with semaphore:
                await adapter.observe(
                    ObservationRequest(
                        benchmark=scenario.benchmark,
                        config=scenario.config,
                        scenario_id=scenario.scenario_id,
                        seed_context=scenario.seed_context,
                        turn=turn,
                        prediction=result.prediction,
                        actual_answer=turn.expected_answer,
                    )
                )
        except Exception as exc:
            results[-1] = replace(
                result,
                exact_match=False,
                error=f"observe_failed: {exc}",
            )
    return results


async def run_benchmarks(
    benchmarks: list[str],
    adapter: MemoryAdapter,
    limit: int = 0,
    offset: int = 0,
    max_turns_per_scenario: int = 0,
    concurrency: int = 8,
    allow_oracle: bool = False,
    judge: Judge | None = None,
) -> dict[str, Any]:
    scenarios = load_scenarios(benchmarks, limit=limit, offset=offset)
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    tasks = [
        _run_scenario(
            adapter,
            scenario,
            allow_oracle,
            semaphore,
            max_turns_per_scenario=max_turns_per_scenario,
            judge=judge,
        )
        for scenario in scenarios
    ]
    try:
        nested_results = await asyncio.gather(*tasks)
        results = [result for scenario_results in nested_results for result in scenario_results]
        return {
            "created_at": datetime.now(UTC).isoformat(),
            "adapter": adapter.name,
            "benchmarks": benchmarks,
            "scenario_count": len(scenarios),
            "offset": offset,
            "max_turns_per_scenario": max_turns_per_scenario,
            "turn_count": len(results),
            "summary": summarize_results(results),
            "results": [asdict(result) for result in results],
        }
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            await close()
        if judge is not None:
            close_judge = getattr(judge, "close", None)
            if close_judge is not None:
                await close_judge()


def write_report(path: str, report: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
