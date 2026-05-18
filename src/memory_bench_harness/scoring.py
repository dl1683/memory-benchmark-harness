from __future__ import annotations

import json
from typing import Any

from memory_bench_harness.types import TurnResult


def normalize_answer(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).lower()


def normalize_mcq_label(value: Any) -> str:
    normalized = normalize_answer(value)
    if len(normalized) == 3 and normalized[0] == "(" and normalized[2] == ")":
        return normalized[1]
    return normalized


def exact_match(prediction: Any, expected: Any) -> bool:
    if isinstance(expected, list) and all(
        isinstance(item, str | int | float | bool) or item is None for item in expected
    ):
        prediction_norm = normalize_mcq_label(prediction)
        return any(prediction_norm == normalize_mcq_label(item) for item in expected)
    return normalize_answer(prediction) == normalize_answer(expected)


def summarize_results(results: list[TurnResult]) -> dict[str, Any]:
    evaluated = len([result for result in results if result.error is None])
    correct = len([result for result in results if result.error is None and result.exact_match])
    semantic_correct = len(
        [
            result
            for result in results
            if result.error is None and (result.exact_match or result.semantic_match is True)
        ]
    )
    errors = len([result for result in results if result.error is not None])
    by_benchmark: dict[str, dict[str, Any]] = {}
    by_config: dict[str, dict[str, Any]] = {}

    def add(bucket: dict[str, dict[str, Any]], key: str, result: TurnResult) -> None:
        item = bucket.setdefault(
            key,
            {"evaluated": 0, "correct": 0, "semantic_correct": 0, "errors": 0},
        )
        if result.error is not None:
            item["errors"] += 1
            return
        item["evaluated"] += 1
        if result.exact_match:
            item["correct"] += 1
        if result.exact_match or result.semantic_match is True:
            item["semantic_correct"] += 1

    for result in results:
        add(by_benchmark, result.benchmark, result)
        add(by_config, f"{result.benchmark}:{result.config or 'default'}", result)

    for bucket in (by_benchmark, by_config):
        for item in bucket.values():
            item["exact_match"] = item["correct"] / max(item["evaluated"], 1)
            item["semantic_accuracy"] = item["semantic_correct"] / max(item["evaluated"], 1)

    return {
        "evaluated": evaluated,
        "correct": correct,
        "semantic_correct": semantic_correct,
        "errors": errors,
        "exact_match": correct / max(evaluated, 1),
        "semantic_accuracy": semantic_correct / max(evaluated, 1),
        "scenario_metrics": summarize_scenarios(results),
        "by_benchmark": by_benchmark,
        "by_config": by_config,
    }


def summarize_scenarios(results: list[TurnResult]) -> dict[str, Any]:
    by_scenario: dict[tuple[str, str | None, str], list[TurnResult]] = {}
    for result in results:
        key = (result.benchmark, result.config, result.scenario_id)
        by_scenario.setdefault(key, []).append(result)

    scenario_count = len(by_scenario)
    full_success = 0
    progress_sum = 0.0
    by_benchmark: dict[str, dict[str, Any]] = {}

    for (benchmark, config, _scenario_id), scenario_results in by_scenario.items():
        no_errors = all(result.error is None for result in scenario_results)
        turn_count = len(scenario_results)
        correct_count = len(
            [
                result
                for result in scenario_results
                if result.error is None and result.exact_match
            ]
        )
        progress = correct_count / max(turn_count, 1)
        success = no_errors and turn_count > 0 and correct_count == turn_count
        if success:
            full_success += 1
        progress_sum += progress

        benchmark_bucket = by_benchmark.setdefault(
            benchmark,
            {
                "scenario_count": 0,
                "full_success": 0,
                "progress_sum": 0.0,
                "by_config": {},
            },
        )
        benchmark_bucket["scenario_count"] += 1
        benchmark_bucket["full_success"] += int(success)
        benchmark_bucket["progress_sum"] += progress

        config_key = config or "default"
        config_bucket = benchmark_bucket["by_config"].setdefault(
            config_key,
            {"scenario_count": 0, "full_success": 0, "progress_sum": 0.0},
        )
        config_bucket["scenario_count"] += 1
        config_bucket["full_success"] += int(success)
        config_bucket["progress_sum"] += progress

    for benchmark_bucket in by_benchmark.values():
        count = max(benchmark_bucket["scenario_count"], 1)
        benchmark_bucket["success_rate"] = benchmark_bucket["full_success"] / count
        benchmark_bucket["progress_score"] = benchmark_bucket["progress_sum"] / count
        for config_bucket in benchmark_bucket["by_config"].values():
            config_count = max(config_bucket["scenario_count"], 1)
            config_bucket["success_rate"] = config_bucket["full_success"] / config_count
            config_bucket["progress_score"] = config_bucket["progress_sum"] / config_count

    return {
        "scenario_count": scenario_count,
        "full_success": full_success,
        "success_rate": full_success / max(scenario_count, 1),
        "progress_score": progress_sum / max(scenario_count, 1),
        "by_benchmark": by_benchmark,
    }
