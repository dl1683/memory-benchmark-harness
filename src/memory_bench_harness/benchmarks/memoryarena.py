from __future__ import annotations

from typing import Any

from memory_bench_harness.types import Scenario, Turn

DATASET_NAME = "ZexueHe/memoryarena"
CONFIGS = (
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)


def _load_dataset(config: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency `datasets`. Install with `uv pip install -e .[datasets]`."
        ) from exc
    return load_dataset(DATASET_NAME, config, split="test")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _row_to_scenario(config: str, row: dict[str, Any]) -> Scenario:
    questions = _as_list(row.get("questions"))
    answers = _as_list(row.get("answers"))
    backgrounds = _as_list(row.get("backgrounds"))
    turns: list[Turn] = []

    for idx, question in enumerate(questions):
        turns.append(
            Turn(
                turn_index=idx,
                prompt=str(question),
                expected_answer=answers[idx] if idx < len(answers) else None,
                background=backgrounds[idx] if idx < len(backgrounds) else None,
            )
        )

    return Scenario(
        benchmark="memoryarena",
        config=config,
        scenario_id=f"{config}:{row.get('id')}",
        category=row.get("category"),
        seed_context=row.get("base_person"),
        sessions=turns,
    )


def load_memoryarena(
    limit: int = 0,
    offset: int = 0,
    configs: list[str] | None = None,
) -> list[Scenario]:
    selected_configs = tuple(configs or CONFIGS)
    unknown = sorted(set(selected_configs) - set(CONFIGS))
    if unknown:
        raise ValueError(f"Unsupported MemoryArena configs: {', '.join(unknown)}")
    scenarios: list[Scenario] = []
    seen = 0
    for config in selected_configs:
        dataset = _load_dataset(config)
        for row in dataset:
            if seen < offset:
                seen += 1
                continue
            scenarios.append(_row_to_scenario(config, dict(row)))
            seen += 1
            if limit > 0 and len(scenarios) >= limit:
                return scenarios
    return scenarios
