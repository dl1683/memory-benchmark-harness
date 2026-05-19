from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Any

from memory_bench_harness.types import Scenario, Turn

AMB_RAW_BASE = (
    "https://raw.githubusercontent.com/vectorize-io/agent-memory-benchmark/main/data"
)
CACHE_DIR = Path(".cache") / "memorybench" / "amb"


def _download_gz_json(dataset: str, split: str, filename: str) -> Any:
    cache_path = CACHE_DIR / dataset / split / filename
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{AMB_RAW_BASE}/{dataset}/{split}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            cache_path.write_bytes(response.read())
    return json.loads(gzip.decompress(cache_path.read_bytes()).decode("utf-8"))


def load_amb_personamem_32k(limit: int = 0, offset: int = 0) -> list[Scenario]:
    documents = _download_gz_json("personamem", "32k", "documents.json.gz")
    queries = _download_gz_json("personamem", "32k", "queries.json.gz")
    if not isinstance(documents, list) or not isinstance(queries, list):
        raise RuntimeError("AMB PersonaMem data must decode to list documents and queries.")

    selected_queries = queries[max(offset, 0):]
    if limit > 0:
        selected_queries = selected_queries[:limit]

    docs_by_user: dict[str, list[dict[str, Any]]] = {}
    for row in documents:
        if not isinstance(row, dict):
            continue
        user_id = str(row.get("user_id") or "")
        if not user_id:
            continue
        docs_by_user.setdefault(user_id, []).append(
            {
                "id": row.get("id"),
                "content": row.get("content") or "",
                "metadata": {
                    "user_id": row.get("user_id"),
                    "timestamp": row.get("timestamp"),
                },
            }
        )

    scenarios: list[Scenario] = []
    for idx, row in enumerate(selected_queries):
        if not isinstance(row, dict):
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        user_id = str(row.get("user_id") or "")
        query_id = str(row.get("id") or f"query_{max(offset, 0) + idx}")
        memory_documents = docs_by_user.get(user_id, [])
        turn = Turn(
            turn_index=0,
            prompt=str(row.get("query") or ""),
            expected_answer=row.get("gold_answers") or [],
            metadata={
                "amb_query_id": row.get("id"),
                "amb_user_id": row.get("user_id"),
                "amb_question_type": meta.get("question_type"),
                "amb_topic": meta.get("topic"),
                "amb_retrieval_query": meta.get("retrieval_query"),
                "question_type": meta.get("question_type"),
                "topic": meta.get("topic"),
                "retrieval_query": meta.get("retrieval_query"),
            },
        )
        scenarios.append(
            Scenario(
                benchmark="amb_personamem_32k",
                config="personamem:32k",
                scenario_id=f"personamem_32k:{query_id}",
                category="personamem",
                seed_context={
                    "benchmark": "agent-memory-benchmark",
                    "dataset": "personamem",
                    "split": "32k",
                    "memory_documents": memory_documents,
                    "document_count": len(memory_documents),
                },
                metadata={
                    "source": "https://github.com/vectorize-io/agent-memory-benchmark",
                    "query_count": 1,
                    "document_count": len(memory_documents),
                },
                sessions=[turn],
            )
        )
    return scenarios
