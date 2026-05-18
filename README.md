# Memory Benchmark Harness

A reusable harness for testing agent-memory systems against public memory
benchmarks.

The goal is simple: memory systems should be evaluated on whether they improve
multi-session agent behavior, not whether they can store notes.

## Current MapU AMA-Bench result

As of 2026-05-18, the harness produced a clean official-protocol AMA-Bench run
for MapU as an agent-memory system:

- Official macro accuracy: `0.626577255143205`.
- Coverage: 208 AMA-Bench episodes, 2,496 judged answers, 0 export warnings.
- Live leaderboard comparison: would rank `#1` on the AMA-Bench memory-agent
  leaderboard, above `AMA-agent` at `0.557925`.
- Model-only comparison: would rank `#3`; this is not claimed as a raw model
  leaderboard result because MapU is an agent memory system.
- Solver and judge: Gemini 3.1 Flash-Lite.

See [MapU AMA-Bench method card](docs/MAPU_AMA_BENCH_METHOD_CARD.md) and
[live benchmark status](MAPU_LIVE_BENCHMARK_STATUS.md) for the run command,
anti-overfitting policy, domain breakdown, and official submission checklist.

## What this runs now

- MemoryArena: multi-session agentic memory tasks from Hugging Face.
- AMA-Bench: long-horizon agent trajectory memory QA from Hugging Face.

## What this tracks as external/heavy benchmarks

- AgentMemoryBench: strong fit, but requires Docker-backed task servers.
- MemoryAgentBench: relevant incremental-memory benchmark, heavier integration.
- LongMemEval / LoCoMo: useful long-term conversational-memory baselines.

## Install

```powershell
uv venv
uv pip install -e ".[all]"
```

## Catalog available benchmarks

```powershell
uv run memorybench catalog
```

## Export benchmark scenarios

```powershell
uv run memorybench export --benchmark memoryarena --out data/memoryarena.jsonl
uv run memorybench export --benchmark ama_bench --out data/ama_bench.sample.jsonl --limit 5
```

## Run a smoke benchmark

The built-in `null` adapter intentionally returns empty predictions. It is a
baseline plumbing check, not a memory system.

```powershell
uv run memorybench run --benchmarks memoryarena ama_bench --adapter null --limit 3 --out results/null_run.json
```

The built-in `oracle` adapter returns the ground-truth answer. It is only a
ceiling and scorer sanity check.

```powershell
uv run memorybench run --benchmarks memoryarena ama_bench --adapter oracle --allow-oracle --limit 3 --out results/oracle_run.json
```

## Run MapU as the memory adapter

Start the MapU REST API separately, then run:

```powershell
uv run memorybench run `
  --benchmarks memoryarena ama_bench `
  --adapter mapu `
  --mapu-base-url http://127.0.0.1:8000 `
  --limit 3 `
  --out results/mapu_run.json
```

The MapU adapter creates one isolated MapU corpus per benchmark scenario. It
writes seed context and completed turns through MapU document ingestion, then
answers later turns by querying MapU.

For leaderboard-style answer quality, add a solver layer on top of MapU
retrieval. Local Ollama is the preferred first pass when available. The Liquid
shortcut uses Liquid's local Ollama/Hugging Face model path by default:

```powershell
ollama pull hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF
uv run memorybench run `
  --benchmarks memoryarena ama_bench `
  --adapter mapu `
  --mapu-base-url http://127.0.0.1:8000 `
  --solver liquid `
  --limit 3 `
  --max-turns-per-scenario 2 `
  --out results/mapu_liquid_solver_run.json
```

You can also use any installed Ollama model directly:

```powershell
uv run memorybench run `
  --benchmarks memoryarena ama_bench `
  --adapter mapu `
  --mapu-base-url http://127.0.0.1:8000 `
  --solver ollama `
  --solver-model phi4:latest `
  --limit 3 `
  --out results/mapu_ollama_solver_run.json
```

Gemini 3.1 Flash-Lite is available as a hosted fallback through the Gemini
OpenAI-compatible endpoint:

```powershell
$env:GEMINI_API_KEY = "<key>"
uv run memorybench run `
  --benchmarks memoryarena ama_bench `
  --adapter mapu `
  --mapu-base-url http://127.0.0.1:8000 `
  --solver gemini `
  --judge gemini `
  --limit 3 `
  --out results/mapu_solver_run.json
```

This keeps the comparison honest: MapU is the memory backend; the solver is the
task-answering model. Exact match is still reported, but `--judge gemini` adds
semantic correctness for free-form benchmark answers where literal string match
is too brittle for quick quality checks.

The solver prompt does not receive full benchmark seed memory directly. Large
memory payloads such as AMA-Bench trajectories are written to the memory adapter
and must come back through adapter retrieval.

## Plug in a memory system

Use the command adapter when the memory system lives outside this repo:

```powershell
uv run memorybench run `
  --benchmarks memoryarena ama_bench `
  --adapter-command "python examples/adapter_command_echo.py" `
  --limit 3 `
  --out results/command_run.json
```

The command receives one JSON request on stdin per benchmark event. `answer`
events must print one JSON response on stdout. `observe` events should persist
the completed turn and may print `{}`.

See [ADAPTER_CONTRACT.md](docs/ADAPTER_CONTRACT.md).

## Evaluation stance

This harness measures memory as an agent substrate:

- later-session answer quality
- evidence/context returned by the memory system
- latency and error rate
- whether prior turns are actually reused
- benchmark-level and task-type-level scores

It is intentionally benchmark-agnostic and memory-system-agnostic.
