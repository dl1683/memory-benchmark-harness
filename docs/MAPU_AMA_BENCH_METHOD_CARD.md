# MapU AMA-Bench method card

Status date: 2026-05-18.

This document records the public-safe evidence for the MapU AMA-Bench run and
the intended official leaderboard submission. It does not include API keys,
private credentials, downloaded datasets, or bulky raw result artifacts.

## Summary

MapU was evaluated as an agent memory system on AMA-Bench using the official
208-episode protocol. The run produced 2,496 judged answers and an official
macro accuracy of `0.626577255143205`.

Live leaderboard comparison from the harness:

- AMA-Bench memory-agent leaderboard: would rank `#1`.
- Current memory-agent leader at audit time: `AMA-agent`, `0.557925`.
- AMA-Bench model-only leaderboard: would rank `#3`.
- Current model-only leader at audit time: `gpt 5.2`, `0.6982833333333334`.

The correct submission category is `Agent`, not `Model`, because the evaluated
system is MapU plus a solver/judge interface, not a standalone base model.

## Run configuration

- Memory backend: MapU REST API at `http://127.0.0.1:8000`.
- Solver: Gemini 3.1 Flash-Lite.
- Judge: Gemini 3.1 Flash-Lite.
- Benchmark: AMA-Bench official 208-episode open-ended QA protocol.
- Questions per episode: 12.
- Total judged answers: 2,496.
- Chunk size: 8 episodes.
- Intra-chunk concurrency: 4 workers.
- Chunk-level parallelism: supported by `-ChunkWorkers` in
  `scripts/run_mapu_ama_official_chunked.ps1`.

Reproduction command:

```powershell
$env:GEMINI_API_KEY = "<key>"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_mapu_ama_official_chunked.ps1 `
  -RunId gemini_official_full_20260518 `
  -ChunkSize 8 `
  -Concurrency 4 `
  -Solver gemini `
  -SolverModel gemini-3.1-flash-lite `
  -Judge gemini `
  -JudgeModel gemini-3.1-flash-lite `
  -SolverMaxTokens 2048 `
  -JudgeMaxTokens 512 `
  -Resume
```

The script clears oracle, benchmark-cue, and experimental intent-layer
environment flags before running the benchmark.

## Scorecard

Official macro accuracy: `0.626577255143205`.

Local flat accuracy: `0.6338141025641025`.

Domain averages:

| Domain | Score |
| --- | ---: |
| GAME | 0.8035 |
| WEB | 0.6814 |
| TEXT2SQL | 0.6580 |
| OPENWORLD_QA | 0.6288 |
| EMBODIED_AI | 0.5109 |
| SOFTWARE | 0.4768 |

Capability breakdown:

| Domain | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| TEXT2SQL | 0.8296 | 0.7712 | 0.7761 | 0.2549 |
| SOFTWARE | 0.3632 | 0.4800 | 0.3836 | 0.6806 |
| WEB | 0.6240 | 0.7957 | 0.6667 | 0.6393 |
| GAME | 0.7750 | 0.8000 | 0.8222 | 0.8167 |
| EMBODIED_AI | 0.5246 | 0.8000 | 0.5667 | 0.1525 |
| OPENWORLD_QA | 0.5714 | 0.6632 | 0.5140 | 0.7667 |

## Completion audit evidence

The final completion audit verified:

- MapU API health endpoint returned `status: ok`, `version: 0.1.0`.
- Official-shape submission exists locally with 208 rows.
- Each row has 12 question UUIDs, 12 answers, and 12 judge scores.
- Total submitted answers: 2,496.
- Export warnings: none.
- Live AMA-Bench leaderboard comparison cleared the memory-agent leaderboard.

The local official submission artifact is intentionally ignored by Git:

```text
results\gemini_official_full_20260518.merged.submission.jsonl
```

## Anti-overfitting policy

This harness treats benchmark scores as regression evidence for general memory
quality, not as permission to specialize to public benchmark examples.

Default-path restrictions:

- No answer tables.
- No expected-answer leakage into adapter answer calls.
- No scenario-specific branches.
- No deterministic helpers keyed on episode IDs, scenario names, or exact
  benchmark prompts.
- No benchmark-cue or oracle environment flags in official runs.
- Structural memory improvements must be generic event, relation, retrieval, or
  synthesis improvements that are useful outside AMA-Bench.

The MapU adapter stores seed context and observed turns through the memory API,
then answers from retrieved memory evidence and generic structural relations.

## Official publication checklist

To submit this run to the official AMA-Bench Hugging Face leaderboard, use the
local merged submission JSONL and the following metadata:

- Submission type: `Agent`.
- Suggested display name: `MapU Memory Agent`.
- Score to expect from official validator: `0.626577255143205`.
- Required fields still needed from the project owner: public organization/name,
  contact email, and public project URL.

Do not publish API keys, local `.env` files, or private run logs.
