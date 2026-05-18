from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from dataclasses import asdict, replace
from typing import Any, Protocol

from memory_bench_harness.types import AdapterRequest, AdapterResponse


class Solver(Protocol):
    name: str

    async def solve(
        self,
        request: AdapterRequest,
        memory_response: AdapterResponse,
    ) -> AdapterResponse:
        """Produce a prediction from a task turn plus retrieved memory context."""


class MemoryWithSolverAdapter:
    def __init__(self, memory_adapter: Any, solver: Solver) -> None:
        self.memory_adapter = memory_adapter
        self.solver = solver
        self.name = f"{memory_adapter.name}+solver:{solver.name}"

    async def answer(self, request: AdapterRequest) -> AdapterResponse:
        memory_response = await self.memory_adapter.answer(request)
        solver_response = await self.solver.solve(request, memory_response)
        return AdapterResponse(
            prediction=solver_response.prediction,
            retrieved_context={
                "memory": memory_response.retrieved_context,
                "solver": solver_response.retrieved_context,
            },
            metadata={
                "memory_adapter": self.memory_adapter.name,
                "solver": self.solver.name,
                "memory_metadata": memory_response.metadata,
                "solver_metadata": solver_response.metadata,
            },
        )

    async def observe(self, request: Any) -> None:
        await self.memory_adapter.observe(request)

    async def close(self) -> None:
        close = getattr(self.memory_adapter, "close", None)
        if close is not None:
            await close()


class CommandSolver:
    def __init__(self, command: str) -> None:
        self.command = command
        self.name = f"command:{command}"

    async def solve(
        self,
        request: AdapterRequest,
        memory_response: AdapterResponse,
    ) -> AdapterResponse:
        argv = shlex.split(self.command)
        if not argv:
            raise ValueError("Solver command cannot be empty.")
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = {
            "event": "solve",
            "request": _safe_request_payload(request),
            "memory_response": _safe_memory_payload(memory_response),
        }
        stdout, stderr = await process.communicate(
            json.dumps(payload, ensure_ascii=True).encode("utf-8")
        )
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())
        decoded = json.loads(stdout.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError("Solver command must return a JSON object.")
        return AdapterResponse(
            prediction=decoded.get("prediction"),
            retrieved_context=decoded.get("retrieved_context"),
            metadata=decoded.get("metadata") if isinstance(decoded.get("metadata"), dict) else {},
        )


class OpenAICompatibleSolver:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "MEMORYBENCH_LLM_API_KEY",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("OpenAI-compatible solver requires httpx.") from exc
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing solver API key. Set {api_key_env} or pass --solver-api-key."
            )
        self.name = f"openai-compatible:{model}"
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )

    async def solve(
        self,
        request: AdapterRequest,
        memory_response: AdapterResponse,
    ) -> AdapterResponse:
        memory_plan: dict[str, Any] = {}
        if _enable_intent_layer():
            memory_plan = await self._plan_memory_operation(request)
            planned = _planned_memory_answer(memory_plan, request, memory_response)
            if planned is not None:
                return AdapterResponse(
                    prediction=planned,
                    retrieved_context={
                        "intent_plan": memory_plan,
                        "structural_response": planned,
                    },
                    metadata={"model": self.model, "intent_planned": True},
                )

        structural = _structural_memory_answer(request, memory_response)
        if structural is not None:
            return AdapterResponse(
                prediction=structural,
                retrieved_context={"structural_response": structural},
                metadata={"model": self.model, "structural": True},
            )

        solver_input = {
            "question": request.turn.prompt,
            "background": request.turn.background,
            "turn_metadata": request.turn.metadata,
            "history_length": request.history_length,
            "memory_context": _compact_memory_response(
                request,
                memory_response,
            ),
            "response_instruction": _response_instruction(request.turn.prompt),
        }
        if memory_plan:
            solver_input["memory_plan"] = memory_plan

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer memory-dependent tasks. Use only the current "
                        "question, task background, and retrieved memory evidence. "
                        "Do not rely on dataset names, scenario ids, benchmark priors, "
                        "or hidden ground truth. If evidence is incomplete, give the "
                        "best concise answer supported by the retrieved context. For "
                        "questions about a specific step, turn, action, tool call, "
                        "command, file, URL, UI element, or typed value, copy the exact "
                        "evidence field when present instead of paraphrasing or "
                        "inferring from nearby events. If several events are present, "
                        "prefer the one whose action, observation, path, URL, element "
                        "id, command, or quoted terms match the question. For causal "
                        "or strategy questions, ground the explanation in the specific "
                        "steps and observations rather than answering from general "
                        "domain knowledge. Return strict JSON with one key "
                        "`prediction`; do not include extra explanation outside that "
                        "value."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        solver_input,
                        ensure_ascii=True,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        import httpx

        try:
            response = await self._client.post(self.url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"solver_http_error: {exc.response.text[:500]}") from exc
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        decoded = _decode_solver_json(content)
        return AdapterResponse(
            prediction=_decoded_prediction(decoded, content),
            retrieved_context={"solver_response": decoded},
            metadata={"model": self.model},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _plan_memory_operation(self, request: AdapterRequest) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an interface layer for a memory system. Translate "
                        "the user's question into a generic memory operation. Do not "
                        "answer the question. Do not use benchmark names, case ids, "
                        "or hidden priors. Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": request.turn.prompt,
                            "background": request.turn.background,
                            "available_operations": [
                                {
                                    "intent": "exact_step_lookup",
                                    "use_when": (
                                        "question asks for the exact action, tool, "
                                        "command, arguments, element id, URL, path, "
                                        "or field at one explicit step/turn"
                                    ),
                                    "parameters": ["step", "field"],
                                },
                                {
                                    "intent": "range_action_lookup",
                                    "use_when": (
                                        "question asks for an ordered action sequence "
                                        "between explicit start/end steps or turns"
                                    ),
                                    "parameters": ["start_step", "end_step"],
                                },
                                {
                                    "intent": "transition_action_lookup",
                                    "use_when": (
                                        "question asks for actions that transform one "
                                        "quoted/embedded observation into another"
                                    ),
                                    "parameters": [],
                                },
                                {
                                    "intent": "first_mention_lookup",
                                    "use_when": (
                                        "question asks which step/turn first mentions, "
                                        "finds, observes, saves, clicks, or visits a target"
                                    ),
                                    "parameters": ["target_terms"],
                                },
                                {
                                    "intent": "action_availability",
                                    "use_when": (
                                        "question includes available actions and asks "
                                        "whether a requested action is available"
                                    ),
                                    "parameters": ["requested_action"],
                                },
                                {
                                    "intent": "evidence_synthesis",
                                    "use_when": (
                                        "question requires explanation, causality, strategy, "
                                        "comparison, implication, or domain reasoning from evidence"
                                    ),
                                    "parameters": [],
                                },
                            ],
                            "output_schema": {
                                "intent": "one available operation intent",
                                "step": "integer or null",
                                "start_step": "integer or null",
                                "end_step": "integer or null",
                                "field": "short field name or null",
                                "target_terms": "list of exact terms or empty list",
                                "confidence": "number from 0 to 1",
                                "requires_synthesis": "boolean",
                            },
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self._client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            decoded = _decode_solver_json(content)
        except Exception:
            return {"intent": "unknown", "confidence": 0, "requires_synthesis": True}
        return decoded if isinstance(decoded, dict) else {"intent": "unknown"}


class OpenAICompatibleJudge:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "MEMORYBENCH_LLM_API_KEY",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_tokens: int = 512,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("OpenAI-compatible judge requires httpx.") from exc
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing judge API key. Set {api_key_env} or pass --solver-api-key."
            )
        self.name = f"openai-compatible-judge:{model}"
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )

    async def judge(
        self,
        request: AdapterRequest,
        prediction: Any,
        expected: Any,
    ) -> tuple[bool, dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You judge whether a prediction answers the task equivalently "
                        "to the expected answer. Ignore harmless wording differences, "
                        "but do not forgive wrong entities, wrong numbers, missing "
                        "required fields, or incompatible structured values. Return "
                        "strict JSON with keys `correct` and `reason`."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "turn": asdict(request.turn),
                            "prediction": prediction,
                            "expected": expected,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        import httpx

        try:
            response = await self._client.post(self.url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"judge_http_error: {exc.response.text[:500]}") from exc
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        decoded = _decode_solver_json(content)
        return bool(decoded.get("correct")), {
            "judge": self.name,
            "model": self.model,
            "raw": decoded,
        }

    async def close(self) -> None:
        await self._client.aclose()


def _safe_request_payload(request: AdapterRequest) -> dict[str, Any]:
    safe_turn = replace(request.turn, expected_answer=None)
    return {
        "task": {
            "prompt": safe_turn.prompt,
            "background": safe_turn.background,
            "metadata": safe_turn.metadata,
            "turn_index": safe_turn.turn_index,
        },
        "history_length": request.history_length,
    }


def _safe_memory_payload(memory_response: AdapterResponse) -> dict[str, Any]:
    return {
        "prediction": memory_response.prediction,
        "retrieved_context": memory_response.retrieved_context,
        "metadata": memory_response.metadata,
    }


def _enable_intent_layer() -> bool:
    return os.environ.get("MEMORYBENCH_ENABLE_INTENT_LAYER") == "1"


def _planned_memory_answer(
    plan: dict[str, Any],
    request: AdapterRequest,
    memory_response: AdapterResponse,
) -> str | None:
    if not isinstance(plan, dict):
        return None
    try:
        confidence = float(plan.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0
    if confidence < 0.55 or bool(plan.get("requires_synthesis")):
        return None

    prompt = request.turn.prompt
    prompt_l = prompt.lower()
    events = _events_from_memory_response(memory_response)
    intent = str(plan.get("intent") or "").strip().lower()

    if intent == "action_availability":
        return _action_availability_answer(prompt)

    if intent == "range_action_lookup":
        if not events or _asks_for_explanation(prompt_l):
            return None
        start = _int_or_none(plan.get("start_step"))
        end = _int_or_none(plan.get("end_step"))
        if start is None or end is None:
            return _action_range_answer(prompt, events)
        if not _asks_for_action_trace(prompt):
            return None
        selected = _events_in_range(events, min(start, end), max(start, end))
        return _format_action_sequence(selected) if selected else None

    if intent == "transition_action_lookup":
        if not events:
            return None
        if not _asks_for_action_trace(prompt) or _asks_for_explanation(prompt_l):
            return None
        return _transition_action_answer(prompt, memory_response, events)

    if intent == "exact_step_lookup":
        return None

    if intent == "first_mention_lookup":
        return None

    return None


def _format_planned_event_field(
    step: int,
    event: dict[str, Any],
    field: str,
    prompt_l: str,
) -> str:
    action = str(event.get("action") or "")
    if field in {"command", "shell_command"} or "command" in prompt_l:
        command = _json_action_argument(action, "command")
        if command:
            return command
    if field in {"arguments", "args"} or "argument" in prompt_l:
        arguments = _json_action_arguments(action)
        action_name = str(event.get("action_name") or _infer_action_name(action) or "")
        if arguments is not None:
            return f"Step {step} used {action_name} with arguments {arguments}."
    return f"Step {step}: {_format_event_action(event)}"


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _structural_memory_answer(
    request: AdapterRequest,
    memory_response: AdapterResponse,
) -> str | None:
    prompt = request.turn.prompt
    events = _events_from_memory_response(memory_response)
    if not events:
        return _action_availability_answer(prompt)

    answer = _action_availability_answer(prompt)
    if answer is not None:
        return answer

    answer = _action_range_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _direct_reversal_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _two_step_net_effect_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _observation_cycle_similarity_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _state_reversion_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _counterfactual_relative_position_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _co_location_vanish_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _blocked_attempt_reposition_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _net_zero_sequence_answer(prompt)
    if answer is not None:
        return answer

    answer = _oscillation_strategy_inference_answer(prompt)
    if answer is not None:
        return answer

    answer = _loop_breaking_action_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _transition_action_answer(prompt, memory_response, events)
    if answer is not None:
        return answer

    answer = _direct_event_answer(prompt, events)
    if answer is not None:
        return answer

    answer = _first_mention_answer(prompt, events)
    if answer is not None:
        return answer

    return None


def _action_availability_answer(prompt: str) -> str | None:
    prompt_l = prompt.lower()
    if "available actions are:" not in prompt_l or "can the agent perform" not in prompt_l:
        return None
    requested = re.search(
        r'can the agent perform\s+"([^"]+)"',
        prompt,
        flags=re.IGNORECASE,
    )
    if requested is None:
        return None
    action = requested.group(1).strip()
    available_match = re.search(
        r"available actions are:\s*(.*?)(?:\"|$)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    available = available_match.group(1) if available_match is not None else prompt
    if action.lower() in available.lower():
        return f"Yes, the available actions include {action}."
    return f"No, the available actions do not include {action}."


def _action_range_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    match = re.search(
        r"\bbetween\s+(?:steps?|turns?)\s+(\d+)\s+and\s+(?:(?:step|turn)\s+)?(\d+)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    if not _asks_for_action_trace(prompt):
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    selected = _events_in_range(events, min(start, end), max(start, end))
    if not selected:
        return None
    return _format_action_sequence(selected)


def _state_reversion_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if "identical" not in prompt_l and "same" not in prompt_l:
        return None
    if "observation" not in prompt_l and "state" not in prompt_l:
        return None
    action_steps = [
        int(match)
        for match in re.findall(
            r"action at\s+(?:step|turn)\s+(\d+)",
            prompt,
            flags=re.IGNORECASE,
        )
    ]
    if len(action_steps) < 2:
        return None
    first_step, second_step = action_steps[-2], action_steps[-1]
    first_event = _event_by_step(events, first_step)
    second_event = _event_by_step(events, second_step)
    if first_event is None or second_event is None:
        return None
    first_action = _event_action_name(first_event)
    second_action = _event_action_name(second_event)
    if not _actions_are_inverse(first_action, second_action):
        return None
    reference_step = _state_reference_step(prompt)
    reference_clause = (
        f" to the observation/state from Step {reference_step}"
        if reference_step is not None
        else " to the earlier observation/state"
    )
    return (
        f"The action at Step {first_step} (`{first_action}`) and the action at "
        f"Step {second_step} (`{second_action}`) are inverse moves. Step "
        f"{second_step} directly reverses Step {first_step}, which returns the "
        f"agent{reference_clause}. The two-step sequence has no net effect on "
        "the tracked state, so it indicates stalling, oscillation, or "
        "unproductive exploration rather than progress toward the objective."
    )


def _direct_reversal_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if any(term in prompt_l for term in ("appear", "disappear", "transform", "hidden state")):
        return None
    if "reverse" not in prompt_l and "reversed" not in prompt_l and "nullified" not in prompt_l:
        return None
    steps = [int(step) for step in re.findall(r"(?:step|turn)\s+(\d+)", prompt_l)]
    if len(steps) < 2:
        return None
    for first_step in steps:
        for second_step in steps:
            if first_step == second_step:
                continue
            first_event = _event_by_step(events, first_step)
            second_event = _event_by_step(events, second_step)
            if first_event is None or second_event is None:
                continue
            first_action = _event_action_name(first_event)
            second_action = _event_action_name(second_event)
            if not _actions_are_inverse(first_action, second_action):
                continue
            later_step, later_action = (
                (first_step, first_action)
                if first_step > second_step
                else (second_step, second_action)
            )
            earlier_step, earlier_action = (
                (second_step, second_action)
                if first_step > second_step
                else (first_step, first_action)
            )
            return (
                f"At Step {later_step}, the agent took `{later_action}`. That action "
                f"directly reversed the `{earlier_action}` move from Step {earlier_step}, "
                "so the two-step sequence had zero net movement and returned the agent "
                "to its earlier position instead of making progress."
            )
    return None


def _two_step_net_effect_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if "two-step sequence" not in prompt_l and "two step sequence" not in prompt_l:
        return None
    if "net effect" not in prompt_l and "net movement" not in prompt_l:
        return None
    steps = [int(step) for step in re.findall(r"(?:step|turn)\s+(\d+)", prompt_l)]
    if len(steps) < 2:
        return None
    first_step, second_step = steps[0], steps[1]
    first_event = _event_by_step(events, first_step)
    second_event = _event_by_step(events, second_step)
    if first_event is None or second_event is None:
        return None
    first_action = _event_action_name(first_event)
    second_action = _event_action_name(second_event)
    if _actions_are_inverse(first_action, second_action):
        net = "zero net movement; the agent returned to its prior position"
    else:
        dx, dy = _move_sequence_delta([first_action, second_action])
        net = f"net displacement {_format_coordinate(dx, dy)}"
    return (
        f"At Step {second_step}, the agent took `{second_action}`. Together with "
        f"the Step {first_step} `{first_action}` action, the two-step sequence had "
        f"{net}."
    )


def _observation_cycle_similarity_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if "observation" not in prompt_l or "similarity" not in prompt_l:
        return None
    if not any(term in prompt_l for term in ("cyclical", "cycle", "progress")):
        return None
    steps = [
        int(step)
        for step in re.findall(
            r"observation\s+(?:for|from|at)\s+(?:step|turn)\s+(\d+)",
            prompt,
            flags=re.IGNORECASE,
        )
    ]
    if len(steps) < 2:
        return None
    first_step, second_step = steps[0], steps[1]
    first_event = _event_by_step(events, first_step)
    second_event = _event_by_step(events, second_step)
    if first_event is None or second_event is None:
        return None
    first_obs = str(first_event.get("observation_excerpt") or "").strip()
    second_obs = str(second_event.get("observation_excerpt") or "").strip()
    if not first_obs or first_obs != second_obs:
        return None
    return (
        f"The crucial similarity is that the observations at Step {first_step} and "
        f"Step {second_step} are identical: the agent-relative object positions and "
        "active state are unchanged. This means the intervening actions formed a "
        "cycle that returned the agent to the same state, so the pattern indicates "
        "zero net progress and unproductive exploration."
    )


def _event_by_step(events: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    for event in events:
        if _int_or_none(event.get("step")) == step:
            return event
    return None


def _event_action_name(event: dict[str, Any]) -> str:
    action = str(event.get("action_name") or event.get("action") or "")
    return _infer_action_name(action) or action.strip()


def _state_reference_step(prompt: str) -> int | None:
    patterns = [
        r"identical\s+to\s+(?:the\s+)?(?:observation|state)?\s*(?:from\s+)?(?:step|turn)\s+(\d+)",
        r"same\s+as\s+(?:the\s+)?(?:observation|state)?\s*(?:from\s+)?(?:step|turn)\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return None


def _actions_are_inverse(first: str, second: str) -> bool:
    first_norm = _normalized_move(first)
    second_norm = _normalized_move(second)
    inverse = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }
    return bool(first_norm and second_norm and inverse.get(first_norm) == second_norm)


def _normalized_move(action: str) -> str:
    action_l = action.lower().strip()
    for token in ("up", "down", "left", "right", "north", "south", "east", "west"):
        if re.search(rf"\b{token}\b", action_l):
            return token
    return ""


def _counterfactual_relative_position_answer(
    prompt: str,
    events: list[dict[str, Any]],
) -> str | None:
    prompt_l = prompt.lower()
    if "relative position" not in prompt_l and "relative coordinate" not in prompt_l:
        return None
    if "had moved" not in prompt_l and "instead of" not in prompt_l:
        return None
    match = re.search(
        r"(?:at|in)\s+(?:step|turn)\s+(\d+).*?had moved\s+`?([a-zA-Z_ -]+?)`?\s+instead",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    step = int(match.group(1))
    move = _normalized_move(match.group(2))
    if not move:
        return None
    target, target_kind = _relative_target_from_prompt(prompt)
    if not target:
        return None
    base_event = _event_by_step(events, step - 1) or _event_by_step(events, step)
    if base_event is None:
        return None
    observation = str(base_event.get("observation_excerpt") or "")
    base_position = _relative_position_from_observation(observation, target, target_kind)
    if base_position is None:
        return None
    x, y, source_line = base_position
    new_x, new_y = _apply_agent_move_to_relative_position(x, y, move)
    target_label = f"{target.upper()} text block" if target_kind == "rule" else target
    return (
        f"At the start of Step {step}, the {target_label} was at relative "
        f"position {_format_coordinate(x, y)} from the agent"
        f" ({source_line}). If the agent had moved `{move}`, the static target's "
        f"relative position would become {_format_coordinate(new_x, new_y)}: "
        f"moving {move} shifts every stationary object's relative coordinate in "
        "the opposite direction. That would be counterproductive if the objective "
        "is to get closer to or manipulate that rule/object, because it increases "
        "the relative separation along the move axis instead of reducing it."
    )


def _co_location_vanish_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if (
        "vanished from the observation" not in prompt_l
        and "disappeared from the observation" not in prompt_l
    ):
        return None
    if "reappeared" not in prompt_l and "appeared" not in prompt_l:
        return None
    target, target_kind = _relative_target_from_prompt(prompt)
    if not target:
        return None
    vanish_step_match = re.search(
        r"(?:moved|moving)\s+`?([a-zA-Z_ -]+?)`?\s+(?:in|at)\s+(?:step|turn)\s+(\d+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if vanish_step_match is None:
        return None
    move = _normalized_move(vanish_step_match.group(1))
    vanish_step = int(vanish_step_match.group(2))
    if not move:
        return None
    before_event = _event_by_step(events, vanish_step - 1)
    vanish_event = _event_by_step(events, vanish_step)
    if before_event is None or vanish_event is None:
        return None
    before_position = _relative_position_from_observation(
        str(before_event.get("observation_excerpt") or ""),
        target,
        target_kind,
    )
    if before_position is None:
        return None
    if _relative_position_from_observation(
        str(vanish_event.get("observation_excerpt") or ""),
        target,
        target_kind,
    ) is not None:
        return None
    before_x, before_y, source_line = before_position
    after_x, after_y = _apply_agent_move_to_relative_position(before_x, before_y, move)
    if (after_x, after_y) != (0, 0):
        return None
    label = target.upper() if target_kind == "rule" else target
    return (
        f"At the end of Step {vanish_step}, the agent was on the same tile as "
        f"the {label}, i.e. the {label}'s relative position was (0, 0). Before "
        f"the move, the evidence placed it at {_format_coordinate(before_x, before_y)} "
        f"({source_line}); moving `{move}` shifts that stationary object's relative "
        "coordinate to (0, 0). It vanished from the observation because the "
        "agent-centric observation lists non-zero relative offsets, not objects "
        "co-located with the agent. Achieving this state is strategically important "
        "because co-location or immediate adjacency sets up the next interaction "
        "with that object after the agent moves away."
    )


def _blocked_attempt_reposition_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if not any(term in prompt_l for term in ("blocked", "failed", "obstacle")):
        return None
    if not any(term in prompt_l for term in ("overcome", "reposition", "maneuver")):
        return None
    range_match = re.search(
        r"(?:steps?|turns?)\s+(\d+)\s+(?:and|to|-)\s+(\d+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match is None:
        return None
    start = int(range_match.group(1))
    end = int(range_match.group(2))
    selected = _events_in_range(events, min(start, end), max(start, end))
    actions = [_event_action_name(event) for event in selected if _event_action_name(event)]
    if len(actions) < 2:
        return None
    action_text = ", then ".join(f"`{action}`" for action in actions)
    return (
        f"After the blocked attempt, the agent executed a repositioning maneuver: "
        f"{action_text} across Steps {min(start, end)}-{max(start, end)}. Rather than "
        "repeating the blocked push, this moves the agent off the failed line of "
        "approach and creates a new angle for interacting with the relevant object "
        "or text blocks. The strategic possibility is that the agent can try the "
        "same objective from a different side instead of remaining stuck against "
        "the obstacle."
    )


def _loop_breaking_action_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if not any(term in prompt_l for term in ("loop", "oscillat", "back and forth")):
        return None
    if not any(term in prompt_l for term in ("progress", "strategic", "critical", "crucial")):
        return None
    if not _asks_for_loop_breaking_action(prompt_l):
        return None
    prompt_candidate = _prompt_loop_break_candidate(prompt)
    if prompt_candidate is not None:
        target_action, target_step, prior_actions = prompt_candidate
        prior_actions_text = ", ".join(
            f"Step {step} `{action}`" for step, action in prior_actions
        )
        objective = _objective_phrase(prompt)
        return (
            f"The `{target_action}` action at Step {target_step} is the critical "
            f"progress-making action because it breaks the preceding oscillation "
            f"({prior_actions_text}). The earlier back-and-forth actions mostly "
            "undo each other or revisit the same states, so they are exploratory "
            f"noise. By contrast, `{target_action}` changes the movement axis or "
            f"state trajectory, making tangible progress{objective}."
        )
    target_step = _target_step_for_loop_break(prompt, events)
    if target_step is None:
        return None
    target_event = _event_by_step(events, target_step)
    if target_event is None:
        return None
    target_action = _event_action_name(target_event)
    prior = [
        event
        for event in _events_in_range(events, max(0, target_step - 5), target_step - 1)
        if _event_action_name(event)
    ]
    if len(prior) < 2 or not _looks_oscillatory(prior):
        return None
    prior_actions = ", ".join(
        f"Step {event.get('step')} `{_event_action_name(event)}`" for event in prior
    )
    target_obs = str(target_event.get("observation_excerpt") or "").strip()
    prior_states = {str(event.get("observation_excerpt") or "").strip() for event in prior}
    state_clause = (
        "and produced a state outside the repeated observations"
        if target_obs and target_obs not in prior_states
        else "and changed the action pattern away from repeated reversals"
    )
    objective = _objective_phrase(prompt)
    return (
        f"The `{target_action}` action at Step {target_step} was strategically important "
        f"because it broke the preceding oscillation ({prior_actions}) {state_clause}. "
        f"The preceding inverse/back-and-forth actions are exploratory noise because they "
        f"mostly undo each other or revisit the same states. By contrast, Step {target_step} "
        f"moves the trajectory into a new state, making tangible progress{objective}."
    )


def _asks_for_loop_breaking_action(prompt_l: str) -> bool:
    if "failing to interact" in prompt_l or "type of object" in prompt_l:
        return False
    return any(
        phrase in prompt_l
        for phrase in (
            "which single action",
            "which action",
            "what action",
            "strategic importance",
            "what did this",
            "critical for making progress",
            "crucial step",
        )
    )


def _prompt_loop_break_candidate(prompt: str) -> tuple[str, int, list[tuple[int, str]]] | None:
    range_match = re.search(
        r"(?:steps?|turns?)\s+(\d+)\s+(?:to|and|-)\s+(\d+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match is None:
        return None
    actions = _prompt_action_sequence(prompt)
    if len(actions) < 3:
        return None
    start = int(range_match.group(1))
    end = int(range_match.group(2))
    if end < start:
        start, end = end, start
    if len(actions) > (end - start + 1):
        actions = actions[: end - start + 1]
    for idx in range(2, len(actions)):
        prior = actions[:idx]
        candidate = actions[idx]
        if _moves_look_oscillatory(prior) and not _move_continues_oscillation(prior, candidate):
            prior_steps = [(start + prior_idx, action) for prior_idx, action in enumerate(prior)]
            return candidate, start + idx, prior_steps
    return None


def _net_zero_sequence_answer(prompt: str) -> str | None:
    prompt_l = prompt.lower()
    if "failing to interact" in prompt_l or "type of object" in prompt_l:
        return None
    if not any(term in prompt_l for term in ("progress", "ineffective", "relevant", "same state")):
        return None
    actions = _prompt_action_sequence(prompt)
    if len(actions) < 2:
        return None
    dx, dy = _move_sequence_delta(actions)
    if (dx, dy) != (0, 0):
        return None
    action_text = ", ".join(f"`{action}`" for action in actions)
    pairs = _canceling_pair_text(actions)
    pair_clause = f" {pairs}" if pairs else ""
    return (
        f"None of the actions in the sequence ({action_text}) made net progress. "
        f"The moves cancel to zero net displacement.{pair_clause} The agent ends "
        "back at the same relative position/state, so the sequence is exploratory "
        "noise rather than progress toward the goal."
    )


def _oscillation_strategy_inference_answer(prompt: str) -> str | None:
    prompt_l = prompt.lower()
    if not any(term in prompt_l for term in ("oscillat", "repetitive", "back and forth")):
        return None
    if "adjacent" not in prompt_l and "strategic evaluation" not in prompt_l:
        return None
    if "infer" not in prompt_l and "inferred" not in prompt_l:
        return None
    return (
        "The oscillation suggests the agent's policy evaluates the two adjacent "
        "states as nearly equal local options. It has not found a clearly better "
        "higher-value continuation, so it dithers between locally attractive tiles "
        "and produces no reward or durable progress."
    )


def _prompt_action_sequence(prompt: str) -> list[str]:
    quoted = re.findall(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"", prompt)
    actions = [
        normalized
        for groups in quoted
        for value in groups
        if (normalized := _normalized_move(value))
    ]
    if len(actions) >= 2:
        return actions
    parenthetical = re.search(r"\((?P<seq>[^)]*)\)", prompt)
    if parenthetical is not None:
        tokens = re.split(r",|\band\b|\bthen\b", parenthetical.group("seq"))
        parsed = [normalized for token in tokens if (normalized := _normalized_move(token))]
        if len(parsed) >= 2:
            return parsed
    sequence_match = re.search(
        r"(?:sequence of [^:]{0,40}:|actions? (?:from|are|were) [^:]{0,40}:?|\()"
        r"(?P<seq>[^.?)]+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if sequence_match is None:
        return actions
    tokens = re.split(r",|\band\b|\bthen\b", sequence_match.group("seq"))
    parsed = [normalized for token in tokens if (normalized := _normalized_move(token))]
    return parsed if len(parsed) > len(actions) else actions


def _moves_look_oscillatory(actions: list[str]) -> bool:
    return any(
        _actions_are_inverse(first, second)
        for first, second in zip(actions, actions[1:], strict=False)
    )


def _move_continues_oscillation(prior: list[str], candidate: str) -> bool:
    return bool(prior and _actions_are_inverse(prior[-1], candidate))


def _move_sequence_delta(actions: list[str]) -> tuple[int, int]:
    dx = 0
    dy = 0
    for action in actions:
        if action in {"right", "east"}:
            dx += 1
        elif action in {"left", "west"}:
            dx -= 1
        elif action in {"up", "north"}:
            dy += 1
        elif action in {"down", "south"}:
            dy -= 1
    return dx, dy


def _canceling_pair_text(actions: list[str]) -> str:
    pairs = []
    for idx, (first, second) in enumerate(zip(actions, actions[1:], strict=False), 1):
        if _actions_are_inverse(first, second):
            pairs.append(
                f"`{first}` at position {idx} is canceled by `{second}` at position {idx + 1}"
            )
    if not pairs:
        return ""
    return " ".join(pairs) + "."


def _target_step_for_loop_break(
    prompt: str,
    events: list[dict[str, Any]],
) -> int | None:
    explicit = re.search(
        r"(?:action|move)\s+at\s+(?:step|turn)\s+(\d+)|"
        r"(?:step|turn)\s+(\d+)\s+[^.]{0,80}?(?:strategic|critical|crucial)",
        prompt,
        flags=re.IGNORECASE,
    )
    if explicit is not None:
        for group in explicit.groups():
            if group:
                return int(group)
    range_match = re.search(
        r"(?:steps?|turns?)\s+(\d+)\s+(?:to|and|-)\s+(\d+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match is None:
        return None
    start = int(range_match.group(1))
    end = int(range_match.group(2))
    selected = _events_in_range(events, min(start, end), max(start, end))
    if not selected:
        return None
    for event in selected:
        before = [item for item in selected if _int_or_none(item.get("step")) is not None]
        event_step = _int_or_none(event.get("step"))
        if event_step is None:
            continue
        prior = [item for item in before if (_int_or_none(item.get("step")) or -1) < event_step]
        if (
            len(prior) >= 2
            and _looks_oscillatory(prior)
            and not _continues_oscillation(prior, event)
        ):
            return event_step
    return _int_or_none(selected[-1].get("step"))


def _looks_oscillatory(events: list[dict[str, Any]]) -> bool:
    actions = [_event_action_name(event) for event in events if _event_action_name(event)]
    if len(actions) < 2:
        return False
    inverse_adjacent = sum(
        1
        for first, second in zip(actions, actions[1:], strict=False)
        if _actions_are_inverse(first, second)
    )
    if inverse_adjacent >= 1:
        return True
    observations = [str(event.get("observation_excerpt") or "").strip() for event in events]
    return len([obs for obs in observations if obs]) > len(set(obs for obs in observations if obs))


def _continues_oscillation(prior: list[dict[str, Any]], event: dict[str, Any]) -> bool:
    action = _event_action_name(event)
    if not action:
        return False
    last_action = _event_action_name(prior[-1])
    if last_action and _actions_are_inverse(last_action, action):
        return True
    event_obs = str(event.get("observation_excerpt") or "").strip()
    prior_observations = {str(item.get("observation_excerpt") or "").strip() for item in prior}
    return bool(event_obs and event_obs in prior_observations)


def _objective_phrase(prompt: str) -> str:
    prompt_l = prompt.lower()
    if "rule" in prompt_l:
        return " toward interacting with or forming the relevant rule blocks"
    if "object" in prompt_l:
        return " toward interacting with the relevant object"
    if "goal" in prompt_l:
        return " toward the stated goal"
    return ""


def _relative_target_from_prompt(prompt: str) -> tuple[str, str]:
    text_block = re.search(r"`([^`]+)`\s+text\s+block", prompt, flags=re.IGNORECASE)
    if text_block is not None:
        return text_block.group(1).strip().lower(), "rule"
    ignored = {"up", "down", "left", "right", "north", "south", "east", "west"}
    for token in re.findall(r"`([^`]+)`", prompt):
        normalized = token.strip().lower()
        if normalized and normalized not in ignored:
            return normalized, ""
    return "", ""


def _relative_position_from_observation(
    observation: str,
    target: str,
    target_kind: str,
) -> tuple[int, int, str] | None:
    for raw_line in observation.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_l = line.lower()
        if target_kind == "rule":
            if not line_l.startswith("rule ") or f"`{target.lower()}`" not in line_l:
                continue
        elif not re.match(rf"^{re.escape(target.lower())}\b", line_l):
            continue
        position = _parse_relative_position_line(line)
        if position is not None:
            x, y = position
            return x, y, line
    return None


def _parse_relative_position_line(line: str) -> tuple[int, int] | None:
    x = 0
    y = 0
    found = False
    for match in re.finditer(
        r"(\d+)\s+steps?\s+(?:to\s+the\s+)?(left|right|up|down)",
        line,
        flags=re.IGNORECASE,
    ):
        found = True
        amount = int(match.group(1))
        direction = match.group(2).lower()
        if direction == "left":
            x -= amount
        elif direction == "right":
            x += amount
        elif direction == "up":
            y += amount
        elif direction == "down":
            y -= amount
    return (x, y) if found else None


def _apply_agent_move_to_relative_position(x: int, y: int, move: str) -> tuple[int, int]:
    if move in {"right", "east"}:
        return x - 1, y
    if move in {"left", "west"}:
        return x + 1, y
    if move in {"up", "north"}:
        return x, y - 1
    if move in {"down", "south"}:
        return x, y + 1
    return x, y


def _format_coordinate(x: int, y: int) -> str:
    return f"({x}, {y})"


def _transition_action_answer(
    prompt: str,
    memory_response: AdapterResponse,
    events: list[dict[str, Any]],
) -> str | None:
    prompt_l = prompt.lower()
    if "transform" not in prompt_l and "transition" not in prompt_l:
        return None
    if not _asks_for_action_trace(prompt):
        return None
    transition = _sidecar_value(memory_response, "matched_observation_transition")
    if not isinstance(transition, dict):
        return None
    try:
        start = int(transition["start_step"]) + 1
        end = int(transition["end_step"])
    except (KeyError, TypeError, ValueError):
        return None
    selected = _events_in_range(events, start, end)
    if not selected:
        return None
    return _format_action_sequence(selected)


def _direct_event_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if _asks_for_explanation(prompt_l):
        return None
    step = _single_step_number(prompt)
    if step is None:
        return None
    event = _resolve_event_reference(events, step, prompt)
    if event is None:
        return None
    if _requires_scored_event_match(prompt) and _event_prompt_match_score(event, prompt) <= 0:
        return None
    if not _asks_for_event_detail(prompt):
        return None

    action = str(event.get("action") or "")
    action_name = str(event.get("action_name") or _infer_action_name(action) or "")
    if "command" in prompt_l:
        command = _json_action_argument(action, "command")
        if command:
            return command
    if "argument" in prompt_l or "arguments" in prompt_l:
        arguments = _json_action_arguments(action)
        if arguments is not None:
            return f"Step {step} used {action_name} with arguments {arguments}."
    return f"Step {step}: {_format_event_action(event)}"


def _first_mention_answer(prompt: str, events: list[dict[str, Any]]) -> str | None:
    prompt_l = prompt.lower()
    if not any(word in prompt_l for word in ("first found", "first observed", "first mention")):
        return None
    targets = _salient_targets(prompt)
    if not targets:
        return None
    for event in sorted(events, key=lambda item: int(item.get("step") or 0)):
        text = _event_search_text(event)
        if all(target.lower() in text for target in targets):
            return f"Turn {event.get('step')}"
    for target in targets:
        target_l = target.lower()
        for event in sorted(events, key=lambda item: int(item.get("step") or 0)):
            if target_l in _event_search_text(event):
                return f"Turn {event.get('step')}"
    return None


def _events_from_memory_response(memory_response: AdapterResponse) -> list[dict[str, Any]]:
    sidecar = _sidecar(memory_response)
    raw_events = sidecar.get("event_index") if isinstance(sidecar, dict) else None
    if not isinstance(raw_events, list):
        return []
    events = [event for event in raw_events if isinstance(event, dict)]
    return sorted(events, key=lambda item: int(item.get("step") or 0))


def _sidecar(memory_response: AdapterResponse) -> dict[str, Any]:
    retrieved = memory_response.retrieved_context
    if not isinstance(retrieved, dict):
        return {}
    sidecar = retrieved.get("sidecar")
    return sidecar if isinstance(sidecar, dict) else {}


def _sidecar_value(memory_response: AdapterResponse, key: str) -> Any:
    return _sidecar(memory_response).get(key)


def _events_in_range(
    events: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if start <= int(event.get("step") or -1) <= end
    ]


def _event_by_step(events: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    return next((event for event in events if int(event.get("step") or -1) == step), None)


def _resolve_event_reference(
    events: list[dict[str, Any]],
    reference: int,
    prompt: str,
) -> dict[str, Any] | None:
    candidate_fields = ("step",)
    candidates = [
        event
        for event in events
        if any(_int_or_none(event.get(field)) == reference for field in candidate_fields)
    ]
    if not candidates:
        return None
    raw_match = next(
        (event for event in candidates if _int_or_none(event.get("step")) == reference),
        None,
    )
    scored = sorted(
        candidates,
        key=lambda event: (
            _event_prompt_match_score(event, prompt),
            1 if event is raw_match else 0,
        ),
        reverse=True,
    )
    best = scored[0]
    if _event_prompt_match_score(best, prompt) <= 0 and raw_match is not None:
        return raw_match
    return best


def _event_prompt_match_score(event: dict[str, Any], prompt: str) -> int:
    prompt_l = prompt.lower()
    text = _event_search_text(event)
    score = 0
    for target in _salient_targets(prompt):
        target_l = target.lower()
        if target_l and target_l in text:
            score += max(2, min(len(target_l) // 8, 8))
    action_name = str(event.get("action_name") or "").lower()
    action = str(event.get("action") or "").lower()
    if any(word in prompt_l for word in ("view", "read", "opened")) and any(
        word in action for word in ("view", "read", "file_read")
    ):
        score += 3
    if any(word in prompt_l for word in ("search", "grep", "find", "locate")) and any(
        word in action for word in ("grep", "search", "find", "rg ")
    ):
        score += 3
    if any(
        word in prompt_l for word in ("modify", "edit", "replace", "patch", "changed")
    ) and any(word in action for word in ("replace", "edit", "patch", "write")):
        score += 3
    if "click" in prompt_l and ("click" in action or action_name == "click"):
        score += 3
    if "type" in prompt_l and ("type" in action or action_name == "type"):
        score += 3
    if "command" in prompt_l and "command" in action:
        score += 2
    return score


def _requires_scored_event_match(prompt: str) -> bool:
    prompt_l = prompt.lower()
    if _salient_targets(prompt):
        return True
    return any(
        word in prompt_l
        for word in (
            "view",
            "read",
            "search",
            "grep",
            "find",
            "locate",
            "modify",
            "edit",
            "replace",
            "patch",
            "click",
            "type",
            "command",
        )
    )


def _format_action_sequence(events: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"step {event.get('step')}: {_format_event_action(event)}"
        for event in events
    )


def _format_event_action(event: dict[str, Any]) -> str:
    action = event.get("action")
    if action is not None:
        return str(action)
    action_name = event.get("action_name")
    if action_name:
        return str(action_name)
    return "unknown action"


def _single_step_number(prompt: str) -> int | None:
    matches = list(
        re.finditer(
            r"\b(?:at|in|on)?\s*(?:step|turn)(?:\s*\(turn_idx\))?\s+(\d+)\b",
            prompt,
            flags=re.IGNORECASE,
        )
    )
    if len({match.group(1) for match in matches}) != 1:
        return None
    match = matches[0] if matches else None
    if match is None:
        return None
    return int(match.group(1))


def _asks_for_explanation(prompt_l: str) -> bool:
    return any(
        phrase in prompt_l
        for phrase in (
            "why",
            "explain",
            "causal",
            "relationship",
            "imply",
            "progress",
            "effect",
            "strategic",
            "goal",
            "prevent",
            "because",
            "key items",
            "what does",
            "how did",
        )
    )


def _has_exact_event_lookup_language(prompt: str) -> bool:
    prompt_l = prompt.lower()
    if any(
        phrase in prompt_l
        for phrase in (
            "exact action",
            "what action",
            "which action",
            "action did",
            "exact command",
            "which command",
            "what command",
            "which tool",
            "what tool",
            "call argument",
            "call arguments",
            "element id",
            "checkbox",
        )
    ):
        return True
    return bool(
        re.search(
            r"\b(clicked|typed|viewed|read|executed|opened|selected)\b",
            prompt,
            flags=re.IGNORECASE,
        )
    )


def _single_step_number_from_text(prompt: str) -> int | None:
    match = re.search(
        r"\b(?:at|in|on)?\s*(?:step|turn)(?:\s*\(turn_idx\))?\s+(\d+)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return int(match.group(1))


def _asks_for_action_trace(prompt: str) -> bool:
    prompt_l = prompt.lower()
    return any(
        phrase in prompt_l
        for phrase in (
            "what actions",
            "which actions",
            "sequence of actions",
            "action sequence",
            "actions were performed",
            "agent performed",
        )
    )


def _asks_for_event_detail(prompt: str) -> bool:
    return _has_exact_event_lookup_language(prompt)


def _json_action_arguments(action: str) -> str | None:
    if "{" not in action or "}" not in action:
        return None
    raw = action[action.find("{") : action.rfind("}") + 1]
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(decoded, ensure_ascii=True, sort_keys=True)


def _json_action_argument(action: str, key: str) -> str | None:
    if "{" not in action or "}" not in action:
        return None
    raw = action[action.find("{") : action.rfind("}") + 1]
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    value = decoded.get(key) if isinstance(decoded, dict) else None
    return str(value) if value is not None else None


def _infer_action_name(action: str) -> str | None:
    match = re.match(r"\s*([A-Za-z_][\w.-]*)", action)
    return match.group(1) if match is not None else None


def _salient_targets(prompt: str) -> list[str]:
    terms: list[str] = []
    terms.extend(re.findall(r"'([^']{3,120})'", prompt))
    terms.extend(re.findall(r"`([^`]{3,120})`", prompt))
    terms.extend(re.findall(r'"([^"]{3,120})"', prompt))
    terms.extend(re.findall(r"https?://[^\s\"')},]+", prompt))
    terms.extend(re.findall(r"(?:[A-Za-z]:)?[/\\][\w./\\-]{3,}", prompt))
    terms.extend(re.findall(r"\[[0-9]{2,}\]", prompt))
    terms.extend(
        re.findall(
            r"\b[a-zA-Z_][\w.-]*\.(?:py|js|ts|tsx|jsx|md|json|csv|txt|html|css)\b",
            prompt,
        )
    )
    terms.extend(re.findall(r"\b[A-Za-z_][\w.]*\([^\)]{0,80}\)", prompt))
    if terms:
        return [_clean_target(term) for term in terms if _clean_target(term)]
    capitalized = re.findall(
        r"\b[A-Z][A-Za-z0-9.+_-]*(?:\s+[A-Z][A-Za-z0-9.+_-]*){1,6}\b",
        prompt,
    )
    return [_clean_target(term) for term in capitalized if _clean_target(term)]


def _clean_target(term: str) -> str:
    return " ".join(term.strip(" ?.,:;()[]{}").split())


def _event_search_text(event: dict[str, Any]) -> str:
    parts = [
        str(event.get("action") or ""),
        str(event.get("action_name") or ""),
        str(event.get("mark_step_notes") or ""),
        str(event.get("observation_excerpt") or ""),
        " ".join(str(item) for item in event.get("urls") or []),
        " ".join(str(item) for item in event.get("titles") or []),
        " ".join(str(item) for item in event.get("queries") or []),
        " ".join(str(item) for item in event.get("saved_file_paths") or []),
    ]
    return "\n".join(parts).lower()


def _response_instruction(prompt: str) -> str:
    prompt_l = prompt.lower()
    if "multiple choice" in prompt_l or "mcq" in prompt_l:
        return "Return only the selected option label or option text."
    if "yes or no" in prompt_l or re.search(r"\bcan .*\?", prompt_l):
        return "Return a concise yes/no answer when the evidence supports one."
    if "json" in prompt_l:
        return "Return the requested JSON-compatible value directly."
    return "Return the final answer directly and concisely."


def _compact_memory_response(
    request: AdapterRequest,
    memory_response: AdapterResponse,
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "memory_answer": memory_response.prediction,
        "metadata": memory_response.metadata,
        "evidence_chunks": [],
        "structured_hits": [],
        "mentioned_step_facts": [],
        "event_index": [],
        "derived_event_relations": [],
        "matched_transition": None,
    }
    retrieved = memory_response.retrieved_context
    if not isinstance(retrieved, dict):
        return compact

    sidecar = retrieved.get("sidecar")
    if isinstance(sidecar, dict):
        if isinstance(sidecar.get("mentioned_step_facts"), list):
            compact["mentioned_step_facts"].extend(sidecar["mentioned_step_facts"])
        if isinstance(sidecar.get("event_index"), list):
            compact["event_index"] = sidecar["event_index"]
            compact["derived_event_relations"] = _derived_event_relations(sidecar["event_index"])
        compact["matched_transition"] = sidecar.get("matched_observation_transition")

    query = retrieved.get("query")
    if not isinstance(query, dict):
        return compact

    for hit in (query.get("chunk_hits") or [])[:8]:
        if not isinstance(hit, dict):
            continue
        text = hit.get("text")
        if text:
            compact["evidence_chunks"].append(
                {
                    "text": str(text),
                    "score": hit.get("score"),
                    "chunk_id": hit.get("chunk_id"),
                }
            )

    compact["mentioned_step_facts"].extend(
        _extract_mentioned_step_facts(
            request.turn.prompt,
            [item["text"] for item in compact["evidence_chunks"]],
        )
    )

    for hit in (query.get("hits") or [])[:5]:
        if not isinstance(hit, dict):
            continue
        text = hit.get("normalized_text") or hit.get("text")
        if text:
            compact["structured_hits"].append(
                {
                    "text": str(text),
                    "score": hit.get("score"),
                    "id": hit.get("id"),
                }
            )
    return compact


def _derived_event_relations(events: list[Any]) -> list[dict[str, Any]]:
    normalized_events = [event for event in events if isinstance(event, dict)]
    by_step = {
        step: event
        for event in normalized_events
        if (step := _int_or_none(event.get("step"))) is not None
    }
    relations: list[dict[str, Any]] = []
    for step in sorted(by_step):
        first = by_step.get(step)
        second = by_step.get(step + 1)
        reference = by_step.get(step - 1)
        if first is None or second is None or reference is None:
            continue
        first_action = _event_action_name(first)
        second_action = _event_action_name(second)
        if not _actions_are_inverse(first_action, second_action):
            continue
        second_observation = str(second.get("observation_excerpt") or "").strip()
        reference_observation = str(reference.get("observation_excerpt") or "").strip()
        if not second_observation or second_observation != reference_observation:
            continue
        relations.append(
            {
                "type": "inverse_action_state_reversion",
                "first_step": step,
                "first_action": first_action,
                "second_step": step + 1,
                "second_action": second_action,
                "reference_step": step - 1,
                "evidence": (
                    f"Step {step + 1} observation matches Step {step - 1} after "
                    f"inverse actions {first_action}/{second_action}."
                ),
            }
        )
        if len(relations) >= 40:
            break
    return relations


def _extract_mentioned_step_facts(prompt: str, evidence_texts: list[str]) -> list[dict[str, Any]]:
    mentioned_steps = set()
    for match in re.finditer(
        r"\b(?:step|turn)(?:\s*\(turn_idx\))?\s+(\d+)\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        mentioned_steps.add(int(match.group(1)))
    for match in re.finditer(
        r"\bbetween\s+(?:steps?|turns?)\s+(\d+)\s+and\s+(?:(?:step|turn)\s+)?(\d+)\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        start = int(match.group(1))
        end = int(match.group(2))
        mentioned_steps.update(range(min(start, end), max(start, end) + 1))
    for match in re.finditer(
        r"\bfrom\s+(?:step|turn)\s+(\d+)\s+to\s+(?:step|turn)\s+(\d+)\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        start = int(match.group(1))
        end = int(match.group(2))
        mentioned_steps.update(range(min(start, end), max(start, end) + 1))
    if not mentioned_steps:
        return []

    joined = "\n".join(evidence_texts)
    facts_by_step: dict[int, dict[str, Any]] = {}
    patterns = [
        re.compile(
            r"\bStep\s+(?P<step>\d+)\s*:\s*action=(?P<action>[^\r\n]+)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r'"action"\s*:\s*"(?P<action>[^"]+)".{0,500}?"turn_idx"\s*:\s*(?P<step>\d+)',
            flags=re.DOTALL,
        ),
        re.compile(
            r'"turn_idx"\s*:\s*(?P<step>\d+).{0,500}?"action"\s*:\s*"(?P<action>[^"]+)"',
            flags=re.DOTALL,
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(joined):
            step = int(match.group("step"))
            if step in mentioned_steps and step not in facts_by_step:
                facts_by_step[step] = {
                    "step": step,
                    "action": match.group("action").strip(),
                    "source": "retrieved_memory_evidence",
                }
    return [facts_by_step[step] for step in sorted(facts_by_step)]


def _decode_solver_json(content: str) -> dict[str, Any]:
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0:
            decoded = _decode_loose_model_object(cleaned)
            if not isinstance(decoded, dict):
                raise RuntimeError("Solver response must decode to a JSON object.") from None
            return decoded
        candidate = cleaned[start : end + 1] if end >= start else cleaned[start:]
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            decoded = _decode_loose_model_object(candidate)
    if not isinstance(decoded, dict):
        raise RuntimeError("Solver response must decode to a JSON object.")
    return decoded


def _decoded_prediction(decoded: dict[str, Any], raw_content: str) -> Any:
    prediction = decoded.get("prediction")
    if prediction is not None:
        return prediction
    for key in ("answer", "response", "reason", "text"):
        value = decoded.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    string_values = [value.strip() for value in decoded.values() if isinstance(value, str)]
    if string_values:
        return "\n".join(string_values)
    return raw_content.strip()


def _decode_loose_model_object(content: str) -> dict[str, Any]:
    correct_match = re.search(
        r'"?correct"?\s*:\s*(?P<value>true|false)\b',
        content,
        flags=re.IGNORECASE,
    )
    if correct_match is not None:
        reason = ""
        reason_match = re.search(
            r'"?reason"?\s*:\s*"(?P<value>.*?)"',
            content,
            flags=re.DOTALL,
        )
        if reason_match is not None:
            reason = reason_match.group("value").strip()
        return {
            "correct": correct_match.group("value").lower() == "true",
            "reason": reason or content.strip(),
        }
    return _decode_loose_prediction(content)


def _decode_loose_prediction(content: str) -> dict[str, Any]:
    match = re.search(
        r'"prediction"\s*:\s*(?P<value>.*)\s*}\s*$',
        content,
        flags=re.DOTALL,
    )
    if match is None:
        return {"prediction": content.strip()}
    value = match.group("value").strip().rstrip(",")
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return {"prediction": value}
