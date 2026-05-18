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
    selected_user_ids = {
        str(row.get("user_id"))
        for row in selected_queries
        if isinstance(row, dict) and row.get("user_id")
    }
    turns: list[Turn] = []
    for idx, row in enumerate(selected_queries):
        if not isinstance(row, dict):
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        turns.append(
            Turn(
                turn_index=idx,
                prompt=str(row.get("query") or ""),
                expected_answer=row.get("gold_answers") or [],
                metadata={
                    "amb_query_id": row.get("id"),
                    "amb_user_id": row.get("user_id"),
                    "amb_gold_ids": row.get("gold_ids") or [],
                    "amb_question_type": meta.get("question_type"),
                    "amb_topic": meta.get("topic"),
                    "amb_retrieval_query": meta.get("retrieval_query"),
                },
            )
        )

    if not turns:
        return []
    memory_documents = []
    for row in documents:
        if not isinstance(row, dict):
            continue
        if selected_user_ids and str(row.get("user_id")) not in selected_user_ids:
            continue
        memory_documents.append(
            {
                "id": row.get("id"),
                "content": row.get("content") or "",
                "metadata": {
                    "user_id": row.get("user_id"),
                    "timestamp": row.get("timestamp"),
                },
            }
        )

    return [
        Scenario(
            benchmark="amb_personamem_32k",
            config="personamem:32k",
            scenario_id=f"personamem_32k_offset{max(offset, 0)}_limit{limit or 'all'}",
            category="personamem",
            seed_context={
                "benchmark": "agent-memory-benchmark",
                "dataset": "personamem",
                "split": "32k",
                "memory_documents": memory_documents,
                "document_count": len(memory_documents),
            },
            sessions=turns,
            metadata={
                "source": "https://github.com/vectorize-io/agent-memory-benchmark",
                "query_count": len(turns),
                "document_count": len(memory_documents),
            },
        )
    ]
