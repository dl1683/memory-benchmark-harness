# Adapter contract

Adapters make memory systems comparable.

Each scenario turn has two event types:

- `answer`: answer the current turn using whatever the memory system already knows.
- `observe`: write the completed turn and ground-truth feedback into memory for future turns.

The answer event intentionally does not include raw prior turns. Otherwise an
adapter could pass by reading the prompt payload instead of using durable memory.

## Answer request

```json
{
  "event": "answer",
  "benchmark": "memoryarena",
  "config": "group_travel_planner",
  "scenario_id": "group_travel_planner:0",
  "seed_context": {},
  "turn": {
    "turn_index": 1,
    "prompt": "current prompt",
    "expected_answer": null,
    "background": null
  },
  "history_length": 1,
  "allow_oracle": false
}
```

## Response

```json
{
  "prediction": "adapter answer",
  "retrieved_context": "memory evidence or context used",
  "metadata": {
    "memory_reads": 2,
    "memory_writes": 1
  }
}
```

## Observe request

```json
{
  "event": "observe",
  "benchmark": "memoryarena",
  "config": "group_travel_planner",
  "scenario_id": "group_travel_planner:0",
  "seed_context": {},
  "turn": {
    "turn_index": 1,
    "prompt": "completed prompt",
    "expected_answer": "ground truth answer",
    "background": null
  },
  "prediction": "adapter answer from the answer event",
  "actual_answer": "ground truth answer"
}
```

## Rules

- Real adapters receive `expected_answer: null` during `answer`.
- Real adapters should only see ground truth during `observe`, after they already answered.
- `retrieved_context` should contain the memory evidence used to answer.
- `metadata` should include memory reads, memory writes, token counts, latency,
  or any backend-specific continuity metrics when available.
- Adapters should be deterministic when temperature/model settings allow it.

## Command adapter behavior

The harness starts a fresh process for each event:

```powershell
python your_adapter.py
```

It writes the request JSON to stdin and reads the response JSON from stdout.
