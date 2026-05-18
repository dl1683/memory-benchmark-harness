from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from memory_bench_harness.types import AdapterRequest, AdapterResponse, ObservationRequest


class MapUAdapter:
    name = "mapu"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        corpus_prefix: str = "memory-eval",
        run_id: str | None = None,
        timeout: float = 60.0,
        max_results: int = 20,
        max_seed_chars: int = 500_000,
        observe_environment_feedback: bool = False,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "MapU adapter requires httpx. Install with `uv pip install -e .[mapu]` "
                "or `uv pip install -e .[all]`."
            ) from exc

        self.base_url = base_url
        self.corpus_prefix = corpus_prefix
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.max_results = max_results
        self.max_seed_chars = max_seed_chars
        self.observe_environment_feedback = observe_environment_feedback
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)
        self._corpus_by_key: dict[str, uuid.UUID] = {}
        self._seed_written: set[str] = set()
        self._trajectory_steps_by_key: dict[str, dict[int, dict[str, Any]]] = {}
        self._environment_feedback_by_key: dict[str, list[dict[str, Any]]] = {}
        self._typed_seed_index_by_key: dict[str, dict[str, list[dict[str, Any]]]] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._client.request(method, path, **kwargs)
        try:
            response.raise_for_status()
        except Exception as exc:
            detail = response.text.strip()
            if detail:
                raise RuntimeError(f"{exc}; response={detail[:500]}") from exc
            raise
        return response.json()

    def _scenario_key(self, benchmark: str, config: str | None, scenario_id: str) -> str:
        return f"{benchmark}:{config or 'default'}:{scenario_id}"

    def _source_prefix(self, scenario_key: str) -> str:
        digest = hashlib.sha256(scenario_key.encode("utf-8")).hexdigest()[:16]
        return f"memory-eval://{self.run_id}/{digest}"

    def _corpus_name(self, scenario_key: str) -> str:
        digest = hashlib.sha256(scenario_key.encode("utf-8")).hexdigest()[:16]
        return f"{self.corpus_prefix}:{self.run_id}:{digest}"

    async def _ensure_corpus(
        self,
        benchmark: str,
        config: str | None,
        scenario_id: str,
        seed_context: Any | None,
    ) -> tuple[uuid.UUID, str]:
        scenario_key = self._scenario_key(benchmark, config, scenario_id)
        if scenario_key not in self._corpus_by_key:
            created = await self._request(
                "POST",
                "/corpora",
                json={
                    "name": self._corpus_name(scenario_key),
                    "description": (
                        "Episodic memory evaluation corpus. "
                        f"suite={benchmark}; config={config}; case_id={scenario_id}"
                    ),
                },
            )
            self._corpus_by_key[scenario_key] = uuid.UUID(str(created["id"]))

        corpus_id = self._corpus_by_key[scenario_key]
        if scenario_key not in self._seed_written and seed_context is not None:
            await self._ingest_seed(
                corpus_id=corpus_id,
                scenario_key=scenario_key,
                benchmark=benchmark,
                config=config,
                scenario_id=scenario_id,
                seed_context=seed_context,
            )
            self._seed_written.add(scenario_key)
        return corpus_id, scenario_key

    async def _ingest_seed(
        self,
        corpus_id: uuid.UUID,
        scenario_key: str,
        benchmark: str,
        config: str | None,
        scenario_id: str,
        seed_context: Any,
    ) -> None:
        memory_documents = self._memory_documents_from_seed(seed_context)
        compact_seed_context = self._compact_seed_context(seed_context)
        self._typed_seed_index_by_key[scenario_key] = self._typed_memory_index(
            compact_seed_context,
            source="seed_context",
            max_facts=800,
            max_records=160,
        )
        if isinstance(seed_context, dict):
            trajectory = seed_context.get("trajectory")
            if isinstance(trajectory, list):
                self._store_trajectory_step_index(scenario_key, trajectory)
                await self._ingest_trajectory_action_index(
                    corpus_id=corpus_id,
                    scenario_key=scenario_key,
                    config=config,
                    scenario_id=scenario_id,
                    trajectory=trajectory,
                )

        for document in memory_documents:
            content = str(document.get("content") or "")
            if not content.strip():
                continue
            doc_id = str(document.get("id") or hashlib.sha256(content.encode()).hexdigest()[:16])
            metadata = (
                document.get("metadata")
                if isinstance(document.get("metadata"), dict)
                else {}
            )
            header = {
                "suite": benchmark,
                "config": config or "default",
                "case_id": scenario_id,
                "document_id": doc_id,
                "metadata": metadata,
            }
            await self._ingest_document(
                corpus_id=corpus_id,
                content=(
                    "# Memory source document\n\n"
                    f"{json.dumps(header, ensure_ascii=True, sort_keys=True)}\n\n"
                    f"{content}"
                ),
                source_uri=f"{self._source_prefix(scenario_key)}/document/{doc_id}",
                document_type="memory_source_document",
                independence_group=scenario_key,
            )

        rendered = json.dumps(compact_seed_context, ensure_ascii=True, sort_keys=True, indent=2)
        truncated = False
        if len(rendered) > self.max_seed_chars:
            rendered = rendered[: self.max_seed_chars]
            truncated = True
        content = (
            "# Memory seed context\n\n"
            f"suite: {benchmark}\n"
            f"config: {config or 'default'}\n"
            f"case_id: {scenario_id}\n"
            f"truncated: {str(truncated).lower()}\n\n"
            "Seed context:\n"
            f"{rendered}\n"
        )
        await self._ingest_document(
            corpus_id=corpus_id,
            content=content,
            source_uri=f"{self._source_prefix(scenario_key)}/seed",
            document_type="memory_seed_context",
            independence_group=scenario_key,
        )

    def _memory_documents_from_seed(self, seed_context: Any) -> list[dict[str, Any]]:
        if not isinstance(seed_context, dict):
            return []
        raw_documents = seed_context.get("memory_documents")
        if not isinstance(raw_documents, list):
            return []
        documents: list[dict[str, Any]] = []
        for item in raw_documents:
            if not isinstance(item, dict):
                continue
            documents.append(dict(item))
        return documents

    def _compact_seed_context(self, seed_context: Any) -> Any:
        if not isinstance(seed_context, dict) or "memory_documents" not in seed_context:
            return seed_context
        compact = dict(seed_context)
        documents = self._memory_documents_from_seed(seed_context)
        compact["memory_documents"] = [
            {
                "id": document.get("id"),
                "metadata": (
                    document.get("metadata")
                    if isinstance(document.get("metadata"), dict)
                    else {}
                ),
                "content_chars": len(str(document.get("content") or "")),
            }
            for document in documents
        ]
        return compact

    def _store_trajectory_step_index(self, scenario_key: str, trajectory: list[Any]) -> None:
        index: dict[int, dict[str, Any]] = {}
        for fallback_idx, raw_turn in enumerate(trajectory):
            if not isinstance(raw_turn, dict):
                continue
            turn_idx = raw_turn.get("turn_idx", fallback_idx)
            try:
                step = int(turn_idx)
            except (TypeError, ValueError):
                continue
            index[step] = {
                "step": step,
                "action": raw_turn.get("action"),
                "observation": raw_turn.get("observation"),
            }
        self._trajectory_steps_by_key[scenario_key] = index

    async def _ingest_trajectory_action_index(
        self,
        corpus_id: uuid.UUID,
        scenario_key: str,
        config: str | None,
        scenario_id: str,
        trajectory: list[Any],
    ) -> None:
        lines = [
            "# Trajectory action index",
            "",
            "This compact index lists exact actions by step for retrieval.",
            f"config: {config or 'default'}",
            f"case_id: {scenario_id}",
            "",
        ]
        for fallback_idx, raw_turn in enumerate(trajectory):
            if not isinstance(raw_turn, dict):
                continue
            turn_idx = raw_turn.get("turn_idx", fallback_idx)
            action = raw_turn.get("action")
            lines.append(f"Step {turn_idx}: action={action}")
        content = "\n".join(lines)
        await self._ingest_document(
            corpus_id=corpus_id,
            content=content,
            source_uri=f"{self._source_prefix(scenario_key)}/seed/action_index",
            document_type="trajectory_action_index",
            independence_group=scenario_key,
        )

    async def _ingest_document(
        self,
        corpus_id: uuid.UUID,
        content: str,
        source_uri: str,
        document_type: str,
        independence_group: str,
    ) -> None:
        await self._request(
            "POST",
            f"/corpora/{corpus_id}/documents",
            json={
                "content": content,
                "mime_type": "text/plain",
                "source_uri": source_uri,
                "document_type": document_type,
                "publication_context": "evaluation",
                "source_identity": "memory-evaluation-harness",
                "independence_group": independence_group,
            },
        )

    def _render_observation(self, request: ObservationRequest) -> str:
        content = (
            "# Memory observed turn\n\n"
            f"suite: {request.benchmark}\n"
            f"config: {request.config or 'default'}\n"
            f"case_id: {request.scenario_id}\n"
            f"turn_index: {request.turn.turn_index}\n\n"
            "Prompt:\n"
            f"{request.turn.prompt}\n\n"
            "Background:\n"
            f"{json.dumps(request.turn.background, ensure_ascii=True, sort_keys=True)}\n\n"
            "Adapter prediction:\n"
            f"{json.dumps(request.prediction, ensure_ascii=True, sort_keys=True)}\n\n"
        )
        if self.observe_environment_feedback:
            content += (
                "Environment feedback / observed outcome:\n"
                f"{json.dumps(request.actual_answer, ensure_ascii=True, sort_keys=True)}\n\n"
            )
        return content

    def _prediction_from_query(self, query_result: dict[str, Any]) -> str:
        synthesis = query_result.get("synthesis")
        if isinstance(synthesis, str) and synthesis.strip():
            return synthesis.strip()
        hits = query_result.get("hits") or []
        if isinstance(hits, list):
            texts = [
                str(hit.get("normalized_text"))
                for hit in hits[:3]
                if isinstance(hit, dict) and hit.get("normalized_text")
            ]
            if texts:
                return "\n".join(texts)
        chunk_hits = query_result.get("chunk_hits") or []
        if isinstance(chunk_hits, list):
            chunks = [
                str(hit.get("text"))
                for hit in chunk_hits[:3]
                if isinstance(hit, dict) and hit.get("text")
            ]
            if chunks:
                return "\n".join(chunks)
        return ""

    def _question_for_turn(self, request: AdapterRequest) -> str:
        if request.turn.background is None:
            return request.turn.prompt
        return (
            "Use the current task background plus any durable memory already stored in MapU.\n\n"
            "Current task background:\n"
            f"{json.dumps(request.turn.background, ensure_ascii=True, sort_keys=True)}\n\n"
            "Question:\n"
            f"{request.turn.prompt}"
        )

    def _mentioned_steps(self, prompt: str) -> list[int]:
        steps = set()
        for match in re.finditer(
            r"\b(?:step|turn)(?:\s*\(turn_idx\))?\s+(\d+)\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            steps.add(int(match.group(1)))
        for match in re.finditer(
            r"\bbetween\s+(?:steps?|turns?)\s+(\d+)\s+and\s+(?:(?:step|turn)\s+)?(\d+)\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            start = int(match.group(1))
            end = int(match.group(2))
            steps.update(range(min(start, end), max(start, end) + 1))
        for match in re.finditer(
            r"\bfrom\s+(?:step|turn)\s+(\d+)\s+to\s+(?:step|turn)\s+(\d+)\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            start = int(match.group(1))
            end = int(match.group(2))
            steps.update(range(min(start, end), max(start, end) + 1))
        return sorted(steps)

    def _sidecar_facts(self, scenario_key: str, prompt: str) -> dict[str, Any]:
        index = self._trajectory_steps_by_key.get(scenario_key) or {}
        mentioned_steps = set(self._mentioned_steps(prompt))
        keyword_steps = self._keyword_steps_in_prompt(index, prompt)
        mentioned_steps.update(keyword_steps)
        observation_steps = self._observation_steps_in_prompt(index, prompt)
        transition = self._observation_transition_in_prompt(index, prompt)
        if transition is not None:
            start = int(transition["start_step"]) + 1
            end = int(transition["end_step"])
            mentioned_steps.update(range(start, end + 1))
        elif len(observation_steps) >= 2:
            start = min(observation_steps)
            end = max(observation_steps)
            mentioned_steps.update(range(start, end + 1))
        facts = []
        for step in sorted(mentioned_steps):
            if step not in index:
                continue
            record = index[step]
            observation = record.get("observation")
            facts.append(
                {
                    "step": step,
                    "action": record.get("action"),
                    "observation": (
                        str(observation)[:12000] if observation is not None else None
                    ),
                    "source": "adapter_seed_sidecar",
                }
            )
        return {
            "mentioned_step_facts": facts,
            "matched_observation_steps": observation_steps,
            "matched_observation_transition": transition,
            "keyword_matched_steps": keyword_steps,
            "event_index": self._event_index(index),
            "event_ledger": self._event_ledger(index),
            "environment_feedback": [
                dict(item)
                for item in self._environment_feedback_by_key.get(scenario_key, [])
            ],
            "typed_seed_index": self._relevant_typed_index(
                self._typed_seed_index_by_key.get(
                    scenario_key,
                    {"facts": [], "records": []},
                ),
                prompt,
                fact_limit=160,
                record_limit=60,
            ),
            "typed_environment_index": self._relevant_typed_index(
                self._typed_memory_index(
                    self._environment_feedback_by_key.get(scenario_key, []),
                    source="environment_feedback",
                    max_facts=320,
                    max_records=80,
                ),
                prompt,
                fact_limit=120,
                record_limit=40,
            ),
        }

    def _typed_memory_index(
        self,
        value: Any,
        *,
        source: str,
        max_facts: int,
        max_records: int,
    ) -> dict[str, list[dict[str, Any]]]:
        facts: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []

        def scalar_text(item: Any) -> str | None:
            if item is None:
                return None
            if isinstance(item, bool):
                return "true" if item else "false"
            if isinstance(item, (int, float, str)):
                text = str(item).strip()
                return text[:2000] if text else None
            return None

        def add_record(path: str, item: dict[str, Any]) -> None:
            if len(records) >= max_records:
                return
            fields: dict[str, str] = {}
            for key, raw in item.items():
                text = scalar_text(raw)
                if text is not None:
                    fields[str(key)] = text
            if len(fields) < 2:
                return
            records.append(
                {
                    "path": path,
                    "source": source,
                    "fields": fields,
                }
            )

        def walk(item: Any, path: str) -> None:
            if len(facts) >= max_facts and len(records) >= max_records:
                return
            text = scalar_text(item)
            if text is not None:
                if len(facts) < max_facts:
                    facts.append(
                        {
                            "path": path,
                            "value": text,
                            "source": source,
                            "value_type": type(item).__name__,
                        }
                    )
                return
            if isinstance(item, dict):
                add_record(path, item)
                for key, child in item.items():
                    child_key = str(key).replace(".", "_")
                    walk(child, f"{path}.{child_key}" if path else child_key)
                return
            if isinstance(item, list):
                for idx, child in enumerate(item):
                    walk(child, f"{path}[{idx}]")

        walk(value, source)
        return {"facts": facts, "records": records}

    def _relevant_typed_index(
        self,
        index: dict[str, list[dict[str, Any]]],
        prompt: str,
        *,
        fact_limit: int,
        record_limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        prompt_tokens = self._typed_index_tokens(prompt)
        if not prompt_tokens:
            return {
                "facts": list(index.get("facts", []))[: min(fact_limit, 40)],
                "records": list(index.get("records", []))[: min(record_limit, 20)],
            }

        def score_text(text: str) -> int:
            tokens = self._typed_index_tokens(text)
            return len(prompt_tokens & tokens)

        scored_facts: list[tuple[int, int, dict[str, Any]]] = []
        for idx, fact in enumerate(index.get("facts", [])):
            haystack = f"{fact.get('path', '')} {fact.get('value', '')}"
            score = score_text(haystack)
            if score:
                scored_facts.append((score, -idx, fact))

        scored_records: list[tuple[int, int, dict[str, Any]]] = []
        for idx, record in enumerate(index.get("records", [])):
            fields = record.get("fields", {})
            field_text = " ".join(
                f"{key} {value}" for key, value in fields.items()
            ) if isinstance(fields, dict) else ""
            score = score_text(f"{record.get('path', '')} {field_text}")
            if score:
                scored_records.append((score, -idx, record))

        scored_facts.sort(reverse=True)
        scored_records.sort(reverse=True)
        return {
            "facts": [fact for _score, _idx, fact in scored_facts[:fact_limit]],
            "records": [
                record for _score, _idx, record in scored_records[:record_limit]
            ],
        }

    def _typed_index_tokens(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9_.$/-]+", text.lower()))
        expanded: set[str] = set()
        ordinal = {
            "first": "1",
            "second": "2",
            "third": "3",
            "fourth": "4",
            "fifth": "5",
            "sixth": "6",
            "seventh": "7",
        }
        for token in tokens:
            if len(token) > 1:
                expanded.add(token)
            if token in ordinal:
                expanded.add(ordinal[token])
        return expanded

    def _event_ledger(self, index: dict[int, dict[str, Any]]) -> dict[str, Any]:
        action_counts: Counter[str] = Counter()
        action_steps: defaultdict[str, list[int]] = defaultdict(list)
        file_accesses: defaultdict[str, list[int]] = defaultdict(list)
        url_accesses: defaultdict[str, list[int]] = defaultdict(list)
        inventory_changes: list[dict[str, Any]] = []

        for step in sorted(index):
            record = index[step]
            action_text = "" if record.get("action") is None else str(record.get("action"))
            observation_text = (
                "" if record.get("observation") is None else str(record.get("observation"))
            )
            combined_text = f"{action_text}\n{observation_text}"
            action_name = self._action_name(action_text) or "unknown"
            action_counts[action_name] += 1
            action_steps[action_name].append(step)

            for path in self._file_refs_from_text(action_text):
                file_accesses[path].append(step)
            for url in self._urls_from_text(combined_text):
                url_accesses[url].append(step)

            change = self._inventory_change_from_action(step, action_text)
            if change is not None:
                inventory_changes.append(change)

        return {
            "action_counts": dict(action_counts),
            "action_steps": dict(action_steps),
            "file_accesses": dict(file_accesses),
            "url_accesses": dict(url_accesses),
            "inventory_changes": inventory_changes,
        }

    def _file_refs_from_text(self, text: str) -> list[str]:
        refs: list[str] = []
        seen = set()
        patterns = [
            r'"(?:file_path|path|filename|source|target)"\s*:\s*"([^"]+)"',
            r"'(?:file_path|path|filename|source|target)'\s*:\s*'([^']+)'",
            (
                r"(?:(?:[A-Za-z]:)?[/\\][\w .@%+=,~:/\\-]+\."
                r"(?:csv|json|sql|py|js|ts|tsx|jsx|md|txt|html|css|yaml|yml|toml|ini|cfg))"
            ),
            (
                r"\b[\w.@%+=,~/-]+\."
                r"(?:csv|json|sql|py|js|ts|tsx|jsx|md|txt|html|css|yaml|yml|toml|ini|cfg)\b"
            ),
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1) if match.lastindex else match.group(0)
                value = value.strip().strip("'\"`.,;:()[]{}")
                if not value or value in seen:
                    continue
                seen.add(value)
                refs.append(value)
                if len(refs) >= 50:
                    return refs
        return refs

    def _inventory_change_from_action(
        self,
        step: int,
        action: str,
    ) -> dict[str, Any] | None:
        cleaned = " ".join(action.strip().split())
        if not cleaned:
            return None
        patterns = [
            ("added", r"\b(?:take|get|pick up|grab)\s+(.+?)(?:\s+from\b|$)"),
            (
                "removed",
                r"\b(?:drop|put|place|insert|move)\s+(.+?)"
                r"(?:\s+(?:in|into|on|onto|at|to)\b|$)",
            ),
        ]
        for direction, pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match is None:
                continue
            item = match.group(1).strip().strip("'\"`.,;:")
            if not item:
                continue
            return {
                "step": step,
                "direction": direction,
                "item": item,
                "action": cleaned,
            }
        return None

    def _event_index(
        self,
        index: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        observation_limit = 1200
        compact = []
        for step in sorted(index):
            record = index[step]
            action = record.get("action")
            observation = record.get("observation")
            action_text = "" if action is None else str(action)
            observation_text = "" if observation is None else str(observation)
            combined_text = f"{action_text}\n{observation_text}"
            mark_step = self._mark_step_summary(action_text, observation_text)
            compact.append(
                {
                    "step": step,
                    "action": action,
                    "action_name": self._action_name(action_text),
                    "urls": self._urls_from_text(combined_text),
                    "titles": self._field_values_from_text(combined_text, "title")[:8],
                    "queries": self._field_values_from_text(combined_text, "query")[:5],
                    "saved_file_paths": self._field_values_from_text(
                        combined_text,
                        "file_path",
                    )[:5],
                    "mark_step_index": mark_step.get("step_index"),
                    "mark_step_status": mark_step.get("step_status"),
                    "mark_step_notes": mark_step.get("step_notes"),
                    "observation_excerpt": (
                        observation_text[:observation_limit] if observation is not None else None
                    ),
                }
            )
        return compact

    def _action_name(self, action: str) -> str | None:
        stripped = action.strip()
        if not stripped:
            return None
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)", stripped)
        return match.group(1) if match else stripped.split()[0]

    def _urls_from_text(self, text: str) -> list[str]:
        urls: list[str] = []
        seen = set()
        for match in re.finditer(r"https?://[^\s\"')},\]]+", text):
            url = match.group(0).rstrip(".,;")
            if url not in seen:
                seen.add(url)
                urls.append(url)
            if len(urls) >= 20:
                break
        return urls

    def _field_values_from_text(self, text: str, field: str) -> list[str]:
        values: list[str] = []
        seen = set()
        patterns = [
            rf'"{re.escape(field)}"\s*:\s*"([^"]+)"',
            rf"'{re.escape(field)}'\s*:\s*'([^']+)'",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = match.group(1).strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
                if len(values) >= 20:
                    return values
        return values

    def _mark_step_summary(self, action: str, observation: str) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for field in ("step_index", "step_status", "step_notes"):
            values = self._field_values_from_text(action, field)
            if not values:
                values = self._field_values_from_text(observation, field)
            if values:
                summary[field] = values[0]
        if "step_notes" not in summary:
            match = re.search(r"step_notes is (.*)", observation, flags=re.IGNORECASE | re.DOTALL)
            if match is not None:
                summary["step_notes"] = match.group(1).strip()[:3000]
        return summary

    def _keyword_steps_in_prompt(
        self,
        index: dict[int, dict[str, Any]],
        prompt: str,
    ) -> list[int]:
        terms = self._salient_prompt_terms(prompt)
        if not terms:
            return []
        matched: list[int] = []
        for step in sorted(index):
            record = index[step]
            haystack = self._normalize_observation_text(
                f"{record.get('action')}\n{record.get('observation')}"
            )
            if any(term in haystack for term in terms):
                matched.append(step)
            if len(matched) >= 20:
                break
        return matched

    def _salient_prompt_terms(self, prompt: str) -> list[str]:
        raw_terms: list[str] = []
        raw_terms.extend(re.findall(r"'([^']{3,120})'", prompt))
        raw_terms.extend(re.findall(r"`([^`]{3,120})`", prompt))
        raw_terms.extend(re.findall(r"https?://\S+", prompt))
        raw_terms.extend(re.findall(r"(?:[A-Za-z]:)?[/\\][\w./\\-]{3,}", prompt))
        raw_terms.extend(re.findall(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b", prompt))
        raw_terms.extend(re.findall(r"\b[A-Z0-9][A-Z0-9.-]{3,}[A-Z0-9]\b", prompt))
        raw_terms.extend(
            match.group(0)
            for match in re.finditer(
                r"\b[A-Z][A-Za-z0-9.+_-]*(?:\s+[A-Z][A-Za-z0-9.+_-]*){1,6}\b",
                prompt,
            )
        )
        raw_terms.extend(
            match.group(0)
            for match in re.finditer(
                r"\b[a-zA-Z_][\w.-]*\.(?:py|js|ts|tsx|jsx|md|json|csv|txt|html|css)\b",
                prompt,
            )
        )

        normalized_terms = []
        seen = set()
        for term in raw_terms:
            normalized = self._normalize_observation_text(str(term).strip())
            if len(normalized) < 3 or normalized in seen:
                continue
            seen.add(normalized)
            normalized_terms.append(normalized)
        return normalized_terms

    def _observation_steps_in_prompt(
        self,
        index: dict[int, dict[str, Any]],
        prompt: str,
    ) -> list[int]:
        normalized_prompt = self._normalize_observation_text(prompt)
        matches = []
        for step, record in index.items():
            observation = record.get("observation")
            if not observation:
                continue
            normalized_observation = self._normalize_observation_text(str(observation))
            if normalized_observation and normalized_observation in normalized_prompt:
                matches.append(step)
        return sorted(matches)

    def _observation_transition_in_prompt(
        self,
        index: dict[int, dict[str, Any]],
        prompt: str,
    ) -> dict[str, Any] | None:
        patterns = [
            r'observation:\s*"(?P<start>.*?)"\s+and\s+the\s+observation:\s*"(?P<end>.*?)"',
            (
                r'observation\s+is:\s*"(?P<start>.*?)"\s+into\s+the\s+state\s+where\s+'
                r'the\s+observation\s+is:\s*"(?P<end>.*?)"'
            ),
        ]
        match = None
        for pattern in patterns:
            match = re.search(
                pattern,
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match is not None:
                break
        if match is None:
            return None
        start_obs = self._normalize_observation_text(match.group("start"))
        end_obs = self._normalize_observation_text(match.group("end"))
        if not start_obs or not end_obs:
            return None

        start_candidates: list[int] = []
        end_candidates: list[int] = []
        for step, record in index.items():
            observation = record.get("observation")
            if not observation:
                continue
            normalized = self._normalize_observation_text(str(observation))
            if normalized == start_obs:
                start_candidates.append(step)
            if normalized == end_obs:
                end_candidates.append(step)
        pairs = [
            (start, end)
            for start in sorted(start_candidates)
            for end in sorted(end_candidates)
            if start < end
        ]
        if not pairs:
            return None

        preferred = [pair for pair in pairs if 3 <= pair[1] - pair[0] <= 8]
        candidates = preferred or pairs
        start, end = min(
            candidates,
            key=lambda pair: (pair[0], pair[1] - pair[0]),
        )
        return {
            "start_step": start,
            "end_step": end,
            "start_candidates": sorted(start_candidates),
            "end_candidates": sorted(end_candidates),
        }

    def _normalize_observation_text(self, text: str) -> str:
        return " ".join(text.replace("|", " ").lower().split())

    async def answer(self, request: AdapterRequest) -> AdapterResponse:
        corpus_id, scenario_key = await self._ensure_corpus(
            benchmark=request.benchmark,
            config=request.config,
            scenario_id=request.scenario_id,
            seed_context=request.seed_context,
        )
        query_result = await self._request(
            "POST",
            f"/corpora/{corpus_id}/query",
            json={"question": self._question_for_turn(request), "max_results": self.max_results},
        )
        resume_result: dict[str, Any] = {}
        try:
            resume_result = await self._request(
                "GET",
                f"/corpora/{corpus_id}/resume",
                params={"max_gaps": 10, "max_activity": 20, "max_actions": 10},
            )
        except Exception:
            resume_result = {}
        sidecar = self._sidecar_facts(scenario_key, request.turn.prompt)
        return AdapterResponse(
            prediction=self._prediction_from_query(query_result),
            retrieved_context={
                "query": query_result,
                "resume": resume_result,
                "sidecar": sidecar,
            },
            metadata={
                "corpus_id": str(corpus_id),
                "epistemic_status": query_result.get("epistemic_status"),
                "tier_used": query_result.get("tier_used"),
                "hit_count": len(query_result.get("hits") or []),
                "chunk_hit_count": len(query_result.get("chunk_hits") or []),
                "gap_count": len(query_result.get("gaps") or []),
                "sidecar_fact_count": len(sidecar.get("mentioned_step_facts", [])),
                "frontier": resume_result.get("continuity_frontier", {}),
            },
        )

    async def observe(self, request: ObservationRequest) -> None:
        corpus_id, scenario_key = await self._ensure_corpus(
            benchmark=request.benchmark,
            config=request.config,
            scenario_id=request.scenario_id,
            seed_context=request.seed_context,
        )
        await self._ingest_document(
            corpus_id=corpus_id,
            content=self._render_observation(request),
            source_uri=f"{self._source_prefix(scenario_key)}/turn/{request.turn.turn_index}",
            document_type="memory_observation",
            independence_group=scenario_key,
        )
        if self.observe_environment_feedback:
            self._environment_feedback_by_key.setdefault(scenario_key, []).append(
                {
                    "turn_index": request.turn.turn_index,
                    "prompt": request.turn.prompt[:1000],
                    "prediction": request.prediction,
                    "observed_outcome": request.actual_answer,
                    "source": "post_turn_environment_feedback",
                }
            )
        try:
            await self._request(
                "POST",
                f"/corpora/{corpus_id}/activity/feedback",
                json={
                    "question": request.turn.prompt,
                    "step": f"memorybench_turn_{request.turn.turn_index}",
                    "outcome": "applied",
                    "actor": "memory-evaluation-harness",
                    "source_event_type": "memory_observation",
                    "notes": json.dumps(
                        {
                            "suite": request.benchmark,
                            "config": request.config,
                            "case_id": request.scenario_id,
                            "turn_index": request.turn.turn_index,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                },
            )
        except Exception:
            return None
