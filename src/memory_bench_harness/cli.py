from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memory_bench_harness.adapters import CommandAdapter, NullAdapter, OracleAdapter
from memory_bench_harness.benchmarks.catalog import BENCHMARKS, EXTERNAL_BENCHMARKS
from memory_bench_harness.runner import LOADERS, run_benchmarks, write_report
from memory_bench_harness.solvers import (
    CommandSolver,
    MemoryWithSolverAdapter,
    OpenAICompatibleJudge,
    OpenAICompatibleSolver,
)
from memory_bench_harness.types import Judge, MemoryAdapter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public memory benchmarks against adapters.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("catalog", help="List runnable and tracked memory benchmarks.")

    export = subparsers.add_parser("export", help="Export normalized scenarios as JSONL.")
    export.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    export.add_argument("--out", required=True)
    export.add_argument("--limit", type=int, default=0)
    export.add_argument("--offset", type=int, default=0)

    ama_submission = subparsers.add_parser(
        "ama-submission",
        help="Export a judged AMA-Bench run report as an official leaderboard JSONL.",
    )
    ama_submission.add_argument("--report", nargs="+", required=True)
    ama_submission.add_argument("--out", required=True)
    ama_submission.add_argument("--expected-episodes", type=int, default=208)
    ama_submission.add_argument("--expected-questions-per-episode", type=int, default=12)
    ama_submission.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a partial/debug submission instead of failing on missing official coverage.",
    )

    leaderboard = subparsers.add_parser(
        "ama-leaderboard",
        help="Fetch the live AMA-Bench leaderboard and optionally compare a score.",
    )
    leaderboard.add_argument("--kind", choices=["agent", "model", "all"], default="agent")
    leaderboard.add_argument(
        "--compare-score",
        type=float,
        default=None,
        help="Optional score to rank against the leaderboard, as 0.65 or 65.",
    )
    leaderboard.add_argument("--top", type=int, default=10)

    run = subparsers.add_parser("run", help="Run benchmarks against one adapter.")
    run.add_argument("--benchmarks", nargs="+", required=True, choices=sorted(LOADERS))
    run.add_argument("--adapter", choices=["null", "oracle", "mapu"], default="null")
    run.add_argument("--adapter-command", default="")
    run.add_argument("--allow-oracle", action="store_true")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--max-turns-per-scenario", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--out", default="results/run.json")
    run.add_argument("--mapu-base-url", default="http://127.0.0.1:8000")
    run.add_argument("--mapu-api-key", default="")
    run.add_argument("--mapu-corpus-prefix", default="memorybench")
    run.add_argument("--mapu-run-id", default="")
    run.add_argument("--mapu-max-results", type=int, default=20)
    run.add_argument("--mapu-max-seed-chars", type=int, default=500_000)
    run.add_argument(
        "--observe-environment-feedback",
        action="store_true",
        help=(
            "Persist post-turn environment feedback/observed outcomes for benchmarks "
            "that explicitly model a memory-agent-environment loop. Off by default."
        ),
    )
    run.add_argument(
        "--solver",
        choices=["none", "openai", "gemini", "ollama", "liquid"],
        default="none",
    )
    run.add_argument("--solver-command", default="")
    run.add_argument("--solver-api-key", default="")
    run.add_argument("--solver-api-key-env", default="MEMORYBENCH_LLM_API_KEY")
    run.add_argument("--solver-model", default="gpt-4o-mini")
    run.add_argument("--solver-base-url", default="https://api.openai.com/v1")
    run.add_argument("--solver-max-tokens", type=int, default=2048)
    run.add_argument(
        "--judge",
        choices=["none", "openai", "gemini", "ollama", "liquid"],
        default="none",
    )
    run.add_argument("--judge-api-key", default="")
    run.add_argument("--judge-api-key-env", default="")
    run.add_argument("--judge-model", default="")
    run.add_argument("--judge-base-url", default="")
    run.add_argument("--judge-max-tokens", type=int, default=512)

    return parser.parse_args()


def _adapter_from_args(ns: argparse.Namespace) -> MemoryAdapter:
    if ns.adapter_command:
        adapter: MemoryAdapter = CommandAdapter(ns.adapter_command)
    elif ns.adapter == "mapu":
        from memory_bench_harness.adapters.mapu import MapUAdapter

        adapter = MapUAdapter(
            base_url=ns.mapu_base_url,
            api_key=ns.mapu_api_key or None,
            corpus_prefix=ns.mapu_corpus_prefix,
            run_id=ns.mapu_run_id or None,
            max_results=ns.mapu_max_results,
            max_seed_chars=ns.mapu_max_seed_chars,
            observe_environment_feedback=ns.observe_environment_feedback,
        )
    elif ns.adapter == "oracle":
        adapter = OracleAdapter()
    else:
        adapter = NullAdapter()

    if ns.solver_command:
        return MemoryWithSolverAdapter(adapter, CommandSolver(ns.solver_command))
    if ns.solver == "openai":
        return MemoryWithSolverAdapter(
            adapter,
            OpenAICompatibleSolver(
                api_key=ns.solver_api_key or None,
                api_key_env=ns.solver_api_key_env,
                model=ns.solver_model,
                base_url=ns.solver_base_url,
                max_tokens=ns.solver_max_tokens,
            ),
        )
    if ns.solver == "gemini":
        gemini_key = (
            ns.solver_api_key
            or os.environ.get(ns.solver_api_key_env)
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        return MemoryWithSolverAdapter(
            adapter,
            OpenAICompatibleSolver(
                api_key=gemini_key,
                api_key_env="GEMINI_API_KEY",
                model=(
                    ns.solver_model
                    if ns.solver_model != "gpt-4o-mini"
                    else "gemini-3.1-flash-lite"
                ),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                max_tokens=ns.solver_max_tokens,
            ),
        )
    if ns.solver == "ollama":
        return MemoryWithSolverAdapter(
            adapter,
            OpenAICompatibleSolver(
                api_key=ns.solver_api_key or "ollama",
                api_key_env=ns.solver_api_key_env,
                model=(
                    ns.solver_model
                    if ns.solver_model != "gpt-4o-mini"
                    else "phi4:latest"
                ),
                base_url=(
                    ns.solver_base_url
                    if ns.solver_base_url != "https://api.openai.com/v1"
                    else "http://127.0.0.1:11434/v1"
                ),
                max_tokens=ns.solver_max_tokens,
            ),
        )
    if ns.solver == "liquid":
        return MemoryWithSolverAdapter(
            adapter,
            OpenAICompatibleSolver(
                api_key=ns.solver_api_key or "ollama",
                api_key_env=ns.solver_api_key_env,
                model=(
                    ns.solver_model
                    if ns.solver_model != "gpt-4o-mini"
                    else "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
                ),
                base_url=(
                    ns.solver_base_url
                    if ns.solver_base_url != "https://api.openai.com/v1"
                    else "http://127.0.0.1:11434/v1"
                ),
                max_tokens=ns.solver_max_tokens,
            ),
        )
    return adapter


def _judge_from_args(ns: argparse.Namespace) -> Judge | None:
    if ns.judge == "none":
        return None
    judge_api_key = ns.judge_api_key or ns.solver_api_key
    judge_api_key_env = ns.judge_api_key_env or ns.solver_api_key_env
    judge_model = ns.judge_model or ns.solver_model
    judge_base_url = ns.judge_base_url or ns.solver_base_url
    if ns.judge == "gemini":
        gemini_key = (
            judge_api_key
            or os.environ.get(judge_api_key_env)
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        return OpenAICompatibleJudge(
            api_key=gemini_key,
            api_key_env="GEMINI_API_KEY",
            model=(
                judge_model
                if judge_model != "gpt-4o-mini"
                else "gemini-3.1-flash-lite"
            ),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            max_tokens=ns.judge_max_tokens,
        )
    if ns.judge == "ollama":
        return OpenAICompatibleJudge(
            api_key=judge_api_key or "ollama",
            api_key_env=judge_api_key_env,
            model=judge_model if judge_model != "gpt-4o-mini" else "phi4:latest",
            base_url=(
                judge_base_url
                if judge_base_url != "https://api.openai.com/v1"
                else "http://127.0.0.1:11434/v1"
            ),
            max_tokens=ns.judge_max_tokens,
        )
    if ns.judge == "liquid":
        return OpenAICompatibleJudge(
            api_key=judge_api_key or "ollama",
            api_key_env=judge_api_key_env,
            model=(
                judge_model
                if judge_model != "gpt-4o-mini"
                else "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
            ),
            base_url=(
                judge_base_url
                if judge_base_url != "https://api.openai.com/v1"
                else "http://127.0.0.1:11434/v1"
            ),
            max_tokens=ns.judge_max_tokens,
        )
    return OpenAICompatibleJudge(
        api_key=judge_api_key or None,
        api_key_env=judge_api_key_env,
        model=judge_model,
        base_url=judge_base_url,
        max_tokens=ns.judge_max_tokens,
    )


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


AMA_LEADERBOARD_BASE = "https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard/raw/main/data"
AMA_DOMAIN_ORDER = ["TEXT2SQL", "SOFTWARE", "WEB", "GAME", "EMBODIED_AI", "OPENWORLD_QA"]
AMA_CAP_ORDER = ["A", "B", "C", "D"]


def _ama_score_macro_average(score: dict[str, Any]) -> float:
    domain_averages: list[float] = []
    for domain in AMA_DOMAIN_ORDER:
        caps = score.get(domain, [])
        cap_values: dict[str, float] = {}
        for item in caps:
            if isinstance(item, dict):
                for cap, value in item.items():
                    if cap in AMA_CAP_ORDER and isinstance(value, int | float):
                        cap_values[cap] = float(value)
        values = [cap_values.get(cap, 0.0) for cap in AMA_CAP_ORDER]
        domain_averages.append(sum(values) / len(values))
    return sum(domain_averages) / len(domain_averages)


def _fetch_ama_leaderboard(kind: str) -> list[dict[str, Any]]:
    url = f"{AMA_LEADERBOARD_BASE}/{kind}.jsonl"
    with urllib.request.urlopen(url, timeout=20) as response:
        lines = response.read().decode("utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        score = entry.get("Score")
        if isinstance(score, dict):
            entry["official_macro_accuracy"] = _ama_score_macro_average(score)
        else:
            entry["official_macro_accuracy"] = 0.0
        rows.append(entry)
    rows.sort(key=lambda item: item["official_macro_accuracy"], reverse=True)
    return rows


def catalog() -> int:
    payload = {
        "runnable": {name: asdict(info) for name, info in BENCHMARKS.items()},
        "external": {name: asdict(info) for name, info in EXTERNAL_BENCHMARKS.items()},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


def ama_leaderboard(kind: str, compare_score: float | None, top: int) -> int:
    kinds = ["agent", "model"] if kind == "all" else [kind]
    normalized_compare = None
    if compare_score is not None:
        normalized_compare = compare_score / 100 if compare_score > 1 else compare_score
    payload: dict[str, Any] = {"status": "ok", "leaderboards": {}}
    for current_kind in kinds:
        rows = _fetch_ama_leaderboard(current_kind)
        top_rows = []
        name_key = "agent_name" if current_kind == "agent" else "model"
        for rank, row in enumerate(rows[: max(top, 0)], 1):
            top_rows.append(
                {
                    "rank": rank,
                    "name": row.get(name_key),
                    "model_family": row.get("model_family", ""),
                    "date": row.get("Date"),
                    "verified": row.get("verified"),
                    "official_macro_accuracy": row["official_macro_accuracy"],
                }
            )
        board: dict[str, Any] = {
            "top": top_rows,
            "entry_count": len(rows),
        }
        if normalized_compare is not None:
            better_count = sum(
                1 for row in rows if row["official_macro_accuracy"] > normalized_compare
            )
            board["comparison"] = {
                "score": normalized_compare,
                "would_rank": better_count + 1,
                "leader_score": rows[0]["official_macro_accuracy"] if rows else None,
                "clears_current_leader": (
                    bool(rows) and normalized_compare >= rows[0]["official_macro_accuracy"]
                ),
            }
        payload["leaderboards"][current_kind] = board
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


def export(benchmark: str, out: str, limit: int, offset: int) -> int:
    scenarios = LOADERS[benchmark](limit=limit, offset=offset)
    _write_jsonl(out, [asdict(scenario) for scenario in scenarios])
    print(
        json.dumps(
            {
                "status": "ok",
                "benchmark": benchmark,
                "path": out,
                "offset": offset,
                "scenario_count": len(scenarios),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


def _result_question_uuid(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    turn_metadata = metadata.get("turn_metadata")
    if not isinstance(turn_metadata, dict):
        return ""
    question_uuid = turn_metadata.get("question_uuid")
    return question_uuid.strip() if isinstance(question_uuid, str) else ""


def _result_question_type(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return "A"
    turn_metadata = metadata.get("turn_metadata")
    if not isinstance(turn_metadata, dict):
        return "A"
    question_type = turn_metadata.get("question_type")
    return _normalize_ama_cap(question_type if isinstance(question_type, str) else "A")


def _normalize_ama_cap(value: str) -> str:
    normalized = value.strip()
    mapping = {
        "A": "A",
        "Recall": "A",
        "B": "B",
        "Causal Inference": "B",
        "Causal": "B",
        "C": "C",
        "State Updating": "C",
        "State": "C",
        "D": "D",
        "State Abstraction": "D",
        "Abstraction": "D",
    }
    return mapping.get(normalized, "A")


def _result_is_correct(result: dict[str, Any]) -> bool:
    if result.get("exact_match") is True:
        return True
    return result.get("semantic_match") is True


def ama_submission(
    report_paths: list[str],
    out: str,
    expected_episodes: int,
    expected_questions_per_episode: int,
    allow_partial: bool,
) -> int:
    results: list[Any] = []
    for report_path in report_paths:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        report_results = report.get("results")
        if not isinstance(report_results, list):
            raise ValueError(f"Run report does not contain a results list: {report_path}")
        results.extend(report_results)

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    errors: list[str] = []
    domain_cap_scores: dict[str, dict[str, list[bool]]] = {}

    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            errors.append(f"result {idx} is not an object")
            continue
        if result.get("benchmark") != "ama_bench":
            errors.append(f"result {idx} is not from ama_bench")
            continue
        episode_id = str(result.get("scenario_id") or "")
        if not episode_id:
            errors.append(f"result {idx} is missing scenario_id")
            continue
        question_uuid = _result_question_uuid(result)
        if not question_uuid:
            errors.append(
                f"result {idx} episode {episode_id} turn {result.get('turn_index')} "
                "is missing metadata.turn_metadata.question_uuid"
            )
            continue
        if episode_id not in grouped:
            grouped[episode_id] = []
            order.append(episode_id)
        domain = str(result.get("config") or "UNKNOWN").upper()
        cap = _result_question_type(result)
        score = _result_is_correct(result)
        domain_cap_scores.setdefault(domain, {}).setdefault(cap, []).append(score)
        grouped[episode_id].append(
            {
                "question_uuid": question_uuid,
                "answer": str(result.get("prediction") or ""),
                "score": score,
                "turn_index": int(result.get("turn_index") or 0),
            }
        )

    if expected_episodes > 0 and len(grouped) != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes, found {len(grouped)}")

    rows: list[dict[str, Any]] = []
    duplicate_questions = 0
    short_episodes = 0
    for episode_id in order:
        answers = sorted(grouped[episode_id], key=lambda item: item["turn_index"])
        seen: set[str] = set()
        for item in answers:
            if item["question_uuid"] in seen:
                duplicate_questions += 1
            seen.add(item["question_uuid"])
        if (
            expected_questions_per_episode > 0
            and len(answers) != expected_questions_per_episode
        ):
            short_episodes += 1
        rows.append(
            {
                "episode_id": episode_id,
                "question_uuid_list": [item["question_uuid"] for item in answers],
                "answer_list": [item["answer"] for item in answers],
                "llm_as_judge_score_list": [item["score"] for item in answers],
                "reasoning_trace": (
                    "Generated by memory-benchmark-harness from a judged AMA-Bench "
                    "run report; scores come from exact_match or semantic_match."
                ),
            }
        )

    if duplicate_questions:
        errors.append(f"found {duplicate_questions} duplicate question UUID(s)")
    if short_episodes:
        errors.append(
            f"{short_episodes} episode(s) do not contain "
            f"{expected_questions_per_episode} questions"
        )

    if errors and not allow_partial:
        print(
            json.dumps(
                {
                    "status": "error",
                    "path": out,
                    "errors": errors[:20],
                    "error_count": len(errors),
                    "hint": (
                        "Run AMA-Bench with --max-turns-per-scenario 0 after this "
                        "metadata patch, or pass --allow-partial for debugging only."
                    ),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1

    _write_jsonl(out, rows)
    scored = sum(score for row in rows for score in row["llm_as_judge_score_list"])
    total = sum(len(row["answer_list"]) for row in rows)
    cap_order = ["A", "B", "C", "D"]
    domain_order = ["TEXT2SQL", "SOFTWARE", "WEB", "GAME", "EMBODIED_AI", "OPENWORLD_QA"]
    score_by_domain_cap: dict[str, list[dict[str, float]]] = {}
    domain_averages: dict[str, float] = {}
    for domain in domain_order:
        cap_scores = domain_cap_scores.get(domain, {})
        score_by_domain_cap[domain] = []
        cap_avgs: list[float] = []
        for cap in cap_order:
            values = cap_scores.get(cap, [])
            avg = sum(1.0 for value in values if value) / len(values) if values else 0.0
            score_by_domain_cap[domain].append({cap: round(avg, 4)})
            cap_avgs.append(avg)
        domain_averages[domain] = sum(cap_avgs) / len(cap_avgs) if cap_avgs else 0.0
    official_macro_accuracy = (
        sum(domain_averages.values()) / len(domain_order) if domain_order else 0.0
    )
    warnings = list(errors)
    missing_domains = [domain for domain in domain_order if domain not in domain_cap_scores]
    if missing_domains:
        warnings.append(
            "missing AMA domains in report: "
            + ", ".join(missing_domains)
            + "; official_macro_accuracy is only leaderboard-comparable on full coverage"
        )
    print(
        json.dumps(
            {
                "status": "ok" if not errors else "partial",
                "path": out,
                "episode_count": len(rows),
                "answer_count": total,
                "local_accuracy": scored / total if total else 0.0,
                "official_macro_accuracy": official_macro_accuracy,
                "domain_averages": domain_averages,
                "score_by_domain_cap": score_by_domain_cap,
                "warnings": warnings[:20],
                "warning_count": len(warnings),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if not errors else 1


def main() -> int:
    ns = _parse_args()
    if ns.command == "catalog":
        return catalog()
    if ns.command == "export":
        return export(ns.benchmark, ns.out, ns.limit, ns.offset)
    if ns.command == "ama-leaderboard":
        return ama_leaderboard(ns.kind, ns.compare_score, ns.top)
    if ns.command == "ama-submission":
        return ama_submission(
            ns.report,
            ns.out,
            ns.expected_episodes,
            ns.expected_questions_per_episode,
            ns.allow_partial,
        )
    if ns.command == "run":
        adapter = _adapter_from_args(ns)
        judge = _judge_from_args(ns)
        report = asyncio.run(
            run_benchmarks(
                benchmarks=ns.benchmarks,
                adapter=adapter,
                limit=ns.limit,
                offset=ns.offset,
                max_turns_per_scenario=ns.max_turns_per_scenario,
                concurrency=ns.concurrency,
                allow_oracle=ns.allow_oracle,
                judge=judge,
            )
        )
        write_report(ns.out, report)
        error_count = int(report["summary"].get("errors", 0))
        print(
            json.dumps(
                {
                    "status": "ok" if error_count == 0 else "error",
                    "path": ns.out,
                    "summary": report["summary"],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1 if error_count else 0
    raise AssertionError(f"Unhandled command {ns.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
