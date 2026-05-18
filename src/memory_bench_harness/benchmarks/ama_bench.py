from __future__ import annotations

from typing import Any

from memory_bench_harness.types import Scenario, Turn

DATASET_NAME = "AMA-bench/AMA-bench"


def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency `datasets`. Install with `uv pip install -e .[datasets]`."
        ) from exc
    return load_dataset(DATASET_NAME, split="test")


def _row_to_scenario(row: dict[str, Any]) -> Scenario:
    turns: list[Turn] = []
    for idx, pair in enumerate(row.get("qa_pairs") or []):
        if not isinstance(pair, dict):
            continue
        turns.append(
            Turn(
                turn_index=idx,
                prompt=str(pair.get("question") or ""),
                expected_answer=str(pair.get("answer") or ""),
                metadata={
                    "question_uuid": pair.get("question_uuid"),
                    "question_type": pair.get("type"),
                },
            )
        )

    return Scenario(
        benchmark="ama_bench",
        config=str(row.get("domain") or "unknown"),
        scenario_id=str(row.get("episode_id")),
        category=str(row.get("task_type") or "unknown"),
        seed_context={
            "task": row.get("task"),
            "domain": row.get("domain"),
            "task_type": row.get("task_type"),
            "success": row.get("success"),
            "num_turns": row.get("num_turns"),
            "total_tokens": row.get("total_tokens"),
            "trajectory": row.get("trajectory") or [],
        },
        sessions=turns,
        metadata={
            "num_turns": row.get("num_turns"),
            "total_tokens": row.get("total_tokens"),
        },
    )


def load_ama_bench(limit: int = 0, offset: int = 0) -> list[Scenario]:
    dataset = _load_dataset()
    scenarios: list[Scenario] = []
    for idx, row in enumerate(dataset):
        if idx < offset:
            continue
        if limit > 0 and len(scenarios) >= limit:
            break
        scenarios.append(_row_to_scenario(dict(row)))
    return scenarios
