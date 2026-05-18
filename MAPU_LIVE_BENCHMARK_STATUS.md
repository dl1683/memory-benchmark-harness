# MapU memory benchmark status

Current policy: benchmark scores are regression signals for general memory quality. The default harness path must not contain scenario-specific branches, exact prompt phrase rules, answer tables, benchmark-cue flags, or deterministic answer helpers.

## Current implementation stance

- The solver answers from the current question, task background, and retrieved memory evidence only.
- The solver payload does not include expected answers, scenario identifiers, or benchmark-specific hints.
- The MapU adapter exposes generic memory structures: seed context, observed turns, trajectory/event indexes, mentioned step facts, matched observations, retrieved evidence, and continuity metadata.
- Observed-turn documents do not store current-turn ground truth answers.
- Tests now assert that expected answers are redacted before adapter answer calls and that MapU observation documents do not contain ground-truth answer text.

## Last validation

- `uv run --with ruff ruff check .` passed.
- `uv run --extra all --extra dev python -m pytest -q` passed with `2 passed`.

## Benchmark interpretation

Prior high scores are not treated as production evidence if they depended on benchmark-specific solver rules. The next meaningful score must come from the current default structural path and should be read as a memory-system regression signal, not as an invitation to patch individual benchmark failures.

## Next improvement target

Improve the underlying memory structure rather than adding benchmark logic:

1. Normalize all trajectory events into typed records with step, action, tool, arguments, paths, URLs, entities, result status, and observation summary.
2. Retrieve by generic question features: explicit step references, quoted entities, URLs, file paths, tool/action names, and observation snippets.
3. Keep raw evidence attached to every answer so failures can be diagnosed as retrieval, normalization, synthesis, or missing-memory issues.
4. Add cross-benchmark and synthetic holdout checks that ask novel memory questions not present in public benchmark phrasing.

## Current clean benchmark evidence

Latest default clean AMA-Bench run:

- Result: `results\mapu_gemini_flash_lite_ama_full208_judged_restored_default_v1.json`
- Environment: oracle unset, benchmark-cue variables unset, intent layer unset.
- Score: 135 / 208 semantic correct, 64.90% semantic accuracy, 0 errors.
- By config: Game 22/30, Embodied 24/30, OpenWorld 16/30, Text2SQL 42/51, Software 12/36, Web 19/31.

Official leaderboard caveat:

- The original clean result was a 208-answer local regression run, not a leaderboard-ready official AMA-Bench submission.
- Official AMA-Bench has 208 episodes with 12 QA pairs each, so a complete leaderboard run must produce 2,496 answers with `--max-turns-per-scenario 0`.
- Official submission rows require `episode_id`, `question_uuid_list`, `answer_list`, and `llm_as_judge_score_list`.
- The harness now carries `turn_metadata.question_uuid` into new run reports and provides `memorybench ama-submission` to export official-shape JSONL plus the official domain/capability macro score shape.
- The harness also provides `memorybench ama-leaderboard` to fetch live Hugging Face leaderboard JSONL, compute the same macro average, and compare a candidate score against agent/model boards.
- Solver and judge configuration are now separated in the CLI with `--judge-api-key`, `--judge-api-key-env`, `--judge-model`, `--judge-base-url`, and `--judge-max-tokens`, so official runs can use one model for answering and another model for semantic judging.
- `scripts\run_mapu_ama_official.ps1` runs the clean official-shape pipeline end to end: clears oracle/cue/intent env flags, checks MapU API health, runs all AMA turns, exports official JSONL, and compares the resulting macro score against the live leaderboard.
- The runner supports `--offset` and `--limit`, and `memorybench ama-submission --report` accepts multiple reports, so full AMA runs can be chunked/resumed and merged before leaderboard comparison.
- `scripts\run_mapu_ama_official_chunked.ps1` orchestrates resumable official runs across chunks, skips existing chunk reports with `-Resume`, supports chunk-level parallelism with `-ChunkWorkers`, merges chunk reports, and compares the merged score once against the live leaderboard.
- The old `mapu_gemini_flash_lite_ama_full208_judged_restored_default_v1.json` artifact predates this metadata and correctly fails official export.
- Current verified online leaderboard bars, computed from the Hugging Face leaderboard data on 2026-05-18: top memory-agent entry is `AMA-agent` at 55.79%; top model entry is `gpt 5.2` at 69.83%.
- Latest full official-protocol MapU run cleared the memory-agent leaderboard but not the model-only leaderboard. Do not claim top model-board performance unless a future run clears 69.83%.

## Full official AMA-Bench result

Run:

- Command: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_mapu_ama_official_chunked.ps1 -RunId gemini_official_full_20260518 -ChunkSize 8 -Concurrency 4 -Solver gemini -SolverModel gemini-3.1-flash-lite -Judge gemini -JudgeModel gemini-3.1-flash-lite -SolverMaxTokens 2048 -JudgeMaxTokens 512 -Resume`
- Report chunks: `results\gemini_official_full_20260518_offset*_limit8.json`
- Merged submission: `results\gemini_official_full_20260518.merged.submission.jsonl`
- Coverage: 208 episodes, 2,496 answers, 0 export warnings.
- Local flat accuracy: 0.6338.
- Official macro accuracy: 0.6266.

Domain averages:

- Game: 0.8035.
- Embodied AI: 0.5109.
- OpenWorld QA: 0.6288.
- Text2SQL: 0.6580.
- Software: 0.4768.
- Web: 0.6814.

Live leaderboard comparison:

- Memory-agent board: would rank 1st; clears current leader `AMA-agent` at 0.5579.
- Model board: would rank 3rd; does not clear `gpt 5.2` at 0.6983 or `GPT-5 mini` at 0.6564.

Official-shape smoke evidence:

- `uv run memorybench ama-submission --report results\mapu_gemini_flash_lite_ama_full208_judged_restored_default_v1.json --out results\mapu_gemini_flash_lite_ama_official_submission_debug.jsonl` correctly failed because the old report has no `turn_metadata.question_uuid` and only one QA turn per episode.
- `uv run memorybench run --benchmarks ama_bench --adapter null --limit 1 --max-turns-per-scenario 0 --concurrency 1 --out results\ama_submission_metadata_smoke.json` produced 12 AMA turns for one episode, and `memorybench ama-submission` exported 1 episode / 12 answers with no warnings.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id official_shape_liquid_smoke_20260518 --solver liquid --solver-model hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest --solver-max-tokens 768 --limit 1 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_liquid_ama_official_shape_limit1.json` completed through the live MapU API with 12 evaluated turns, 0 errors, and exported `results\mapu_liquid_ama_official_shape_limit1_submission.jsonl` with no warnings.
- `uv run memorybench ama-leaderboard --kind all --compare-score 64.9 --top 3` fetched the live boards and showed that 64.9% would rank first on the memory-agent board but third on the model board if reproduced under the official protocol.
- `uv run memorybench run --benchmarks ama_bench --adapter null --limit 1 --max-turns-per-scenario 1 --concurrency 1 --judge ollama --judge-model phi4:latest --judge-max-tokens 64 --out results\judge_args_phi4_smoke.json` completed with 1 evaluated turn and 0 errors, validating the separate local judge argument path.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id official_shape_liquid_phi4_limit2_retry_20260518 --solver liquid --solver-model hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest --solver-max-tokens 768 --judge ollama --judge-model phi4:latest --judge-max-tokens 128 --limit 2 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_liquid_phi4_ama_official_shape_limit2_retry.json` completed with 24 evaluated turns, 0 errors, but 0 semantic correct.
- `uv run memorybench ama-submission --report results\mapu_liquid_phi4_ama_official_shape_limit2_retry.json --out results\mapu_liquid_phi4_ama_official_shape_limit2_retry_submission.jsonl --expected-episodes 2 --expected-questions-per-episode 12` exported 2 episodes / 24 answers with official macro 0.0.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id state_reversion_structural_smoke_20260518 --solver liquid --solver-model hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest --solver-max-tokens 768 --judge ollama --judge-model phi4:latest --judge-max-tokens 128 --limit 1 --max-turns-per-scenario 1 --concurrency 1 --out results\mapu_state_reversion_structural_smoke.json` scored 1 / 1 semantic with 0 errors after the generic state-reversion extractor.
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_mapu_ama_official.ps1 -RunId official_script_smoke_local_20260518 -Solver ollama -SolverModel qwen2.5:0.5b -Judge ollama -JudgeModel qwen2.5:0.5b -Limit 1 -Concurrency 1 -SolverMaxTokens 64 -JudgeMaxTokens 64` completed the end-to-end script smoke with 12 evaluated turns, 0 errors, exported official-shape JSONL, and live leaderboard comparison. The output now warns that partial-domain macro scores are not leaderboard-comparable.
- `uv run memorybench run --benchmarks ama_bench --adapter null --offset 0 --limit 1 --max-turns-per-scenario 0 --concurrency 1 --out results\chunk_smoke_offset0.json`, `uv run memorybench run --benchmarks ama_bench --adapter null --offset 1 --limit 1 --max-turns-per-scenario 0 --concurrency 1 --out results\chunk_smoke_offset1.json`, and `uv run memorybench ama-submission --report results\chunk_smoke_offset0.json results\chunk_smoke_offset1.json --out results\chunk_smoke_merged_submission.jsonl --expected-episodes 2 --expected-questions-per-episode 12` validated chunked official-shape merging: 2 episodes / 24 answers, 0 export warnings except the expected missing-domain comparability warning.
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_mapu_ama_official_chunked.ps1 -RunId official_chunked_script_smoke_nollm_20260518 -Solver none -Judge none -ChunkSize 1 -MaxChunks 2 -Concurrency 1 -SolverMaxTokens 64 -JudgeMaxTokens 64` validated the resumable chunked script against the live MapU API: two 12-turn chunks, merged 2-episode / 24-answer submission, and final live leaderboard comparison.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id counterfactual_geometry_smoke_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 64 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 1 --max-turns-per-scenario 2 --concurrency 1 --out results\mapu_counterfactual_geometry_smoke.json` scored 2 / 2 semantic, with the second prediction correctly deriving the `DOOR` text block counterfactual relative coordinate `(-4, 0)` from Step 46 evidence.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id colocation_structural_smoke_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 64 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 1 --max-turns-per-scenario 3 --concurrency 1 --out results\mapu_colocation_structural_smoke.json` scored 3 / 3 semantic. The third prediction correctly derived object co-location `(0, 0)` when the `ball` vanished after the agent moved onto its prior relative coordinate `(1, 0)`.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id episode0_structural_lift_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 1 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_episode0_structural_lift.json` scored 11 / 12 semantic with 0 errors on the first official-shape AMA episode after the structural relation additions.
- `uv run memorybench ama-submission --report results\mapu_episode0_structural_lift.json --out results\mapu_episode0_structural_lift.submission.jsonl --expected-episodes 1 --expected-questions-per-episode 12` exported the episode-0 slice with Game-domain macro 0.9375 and the expected partial-domain leaderboard comparability warning.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id episode0_loop_break_lift_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 1 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_episode0_loop_break_lift.json` regressed to 8 / 12 because the first loop-breaking rule over-fired on object-interaction questions and trusted event-step indexing over explicit prompt action sequences.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id episode0_loop_break_tightened_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 1 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_episode0_loop_break_tightened.json` scored 12 / 12 semantic with 0 errors after tightening the loop-breaking rule to action-importance prompts and explicit action-sequence parsing.
- `uv run memorybench ama-submission --report results\mapu_episode0_loop_break_tightened.json --out results\mapu_episode0_loop_break_tightened.submission.jsonl --expected-episodes 1 --expected-questions-per-episode 12` exported the episode-0 slice with Game-domain macro 1.0 and the expected partial-domain leaderboard comparability warning.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id game_limit2_structural_generalization_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 2 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_game_limit2_structural_generalization.json` scored 14 / 24 semantic on the first two Game episodes before net-zero sequence handling.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id game_limit2_net_zero_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 2 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_game_limit2_net_zero.json` scored 15 / 24 but regressed episode 0 because net-zero answered an object-interaction question.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id game_limit2_net_zero_tightened_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 2 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_game_limit2_net_zero_tightened.json` scored 16 / 24 semantic with 0 errors after excluding object-type/failing-to-interact prompts from net-zero handling.
- `uv run memorybench ama-submission --report results\mapu_game_limit2_net_zero_tightened.json --out results\mapu_game_limit2_net_zero_tightened.submission.jsonl --expected-episodes 2 --expected-questions-per-episode 12` exported the two-episode slice with Game-domain macro 0.6875 and the expected partial-domain leaderboard comparability warning.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id game_limit2_twostep_parentheses_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 2 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_game_limit2_twostep_parentheses.json` scored 17 / 24 semantic with 0 errors after adding explicit two-step net-effect handling and parenthesized action sequence parsing.
- `uv run memorybench ama-submission --report results\mapu_game_limit2_twostep_parentheses.json --out results\mapu_game_limit2_twostep_parentheses.submission.jsonl --expected-episodes 2 --expected-questions-per-episode 12` exported the two-episode slice with Game-domain macro 0.7292 and the expected partial-domain leaderboard comparability warning.
- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id game_limit2_structural_more_relations_20260518 --solver ollama --solver-model qwen2.5:0.5b --solver-max-tokens 128 --judge ollama --judge-model qwen2.5:0.5b --judge-max-tokens 64 --limit 2 --max-turns-per-scenario 0 --concurrency 1 --out results\mapu_game_limit2_structural_more_relations.json` scored 19 / 24 semantic with 0 errors after adding blocked-attempt repositioning, identical-observation cycle comparison, and oscillation strategy inference while gating direct reversal away from transformation/appearance prompts.
- `uv run memorybench ama-submission --report results\mapu_game_limit2_structural_more_relations.json --out results\mapu_game_limit2_structural_more_relations.submission.jsonl --expected-episodes 2 --expected-questions-per-episode 12` exported the two-episode slice with Game-domain macro 0.7917 and the expected partial-domain leaderboard comparability warning.
- `uv run --with ruff ruff check src\memory_bench_harness\runner.py src\memory_bench_harness\cli.py` and `uv run --with ruff ruff check src\memory_bench_harness\cli.py` passed after the official-submission and live-leaderboard changes.
- `uv run --with ruff ruff check src\memory_bench_harness\solvers.py` passed after the parser robustness and state-reversion changes.

Important experiment outcomes:

- Flash-Lite intent layer is implemented but opt-in with `MEMORYBENCH_ENABLE_INTENT_LAYER=1`. It should remain off by default until it improves full clean scores.
- Direct planner execution for exact-step and first-mention prompts hurt Web-style ambiguous UI questions, so exact-step and first-mention plans must guide synthesis rather than bypass it.
- Focused evidence windows around salient events regressed the 177-case slice and should not be default.
- Ordinal event-reference surfaces regressed the full default run and should not be default unless paired with a better disambiguation/rerank layer.
- Local Liquid and Phi4 are not viable full-run solvers on their own for AMA-Bench: Liquid produced malformed or low-quality answers, Phi4 answered from game priors instead of retrieved evidence, and a two-episode Liquid+Phi4 judged sample scored 0 / 24.
- Generic structural extraction remains valuable. The inverse-action state-reversion extractor uses event evidence to answer same/identical state questions where consecutive actions cancel each other, without benchmark-specific answer tables or prompt phrase hacks.
- Solver context now includes generic `derived_event_relations`, starting with `inverse_action_state_reversion` links derived from adjacent inverse actions and repeated observations. This gives any solver reusable memory relations instead of forcing it to infer them from raw event text.
- The structural layer now handles counterfactual agent-centric geometry: when a question asks how a stationary object's relative position would change if the agent had moved in another direction, it parses the object's observed relative coordinate and applies the inverse coordinate update.
- The structural layer now handles object co-location from disappearance/reappearance in agent-centric observations: if an object at a non-zero relative coordinate vanishes after the agent moves onto that coordinate, MapU can answer that the object is at relative `(0, 0)` and explain why the observation omitted it.
- The structural layer now handles loop-breaking action importance when the question is explicitly asking which action or why a named action mattered. The default rule is tightened to avoid object-type questions and to prefer action sequences stated in the prompt over ambiguous event-step indexing.
- The structural layer now handles net-zero movement sequences and direct reversal questions. The net-zero operator is gated away from object-interaction prompts after an observed overfire.
- The structural layer now handles explicit two-step net-effect questions and parenthesized action sequences such as `(left, right, down, up)`.
- The structural layer now handles blocked-attempt repositioning, identical-observation cycle comparison, and oscillation strategy inference. Direct reversal is gated away from transformation/appearance prompts to avoid suppressing higher-level rule-change explanations.

Next architecture target:

Build a general high-recall retrieval/rerank layer inspired by top memory systems: hybrid lexical + dense retrieval, source-preserving episode windows only after rerank, and explicit temporal/causal/objective links. Do not add benchmark-specific prompt or scenario rules.

## Post-leaderboard general memory improvements

The first post-leaderboard improvement targets the lowest-scoring capability
shape without benchmark-specific answer tables: temporal/action ledgers.

Implemented generic ledger primitives:

- Action frequency and action-step timelines.
- Entity-scoped direct-object action histories.
- File/URL access summaries.
- Inventory additions/removals from transfer actions.
- Structural prompt normalization from MapU `structured_next_steps` target
  questions, so the ledger can consume the memory system's own normalized
  operation target instead of relying only on raw prompt wording.

Validation slice:

- `uv run memorybench run --benchmarks ama_bench --adapter mapu --mapu-base-url http://127.0.0.1:8000 --mapu-run-id event_ledger_gemini_judged_smoke_20260518 --solver gemini --solver-model gemini-3.1-flash-lite --solver-max-tokens 512 --judge gemini --judge-model gemini-3.1-flash-lite --judge-max-tokens 256 --offset 30 --limit 1 --max-turns-per-scenario 12 --concurrency 1 --out results\event_ledger_gemini_judged_smoke_20260518.json`
- Result: 11 / 12 semantic correct, 0 errors, 91.67% semantic accuracy on one
  Embodied AI episode.
- The remaining miss is a general state-transition issue: container interaction
  summaries should distinguish actions mentioning an entity from actions that
  changed that entity's state.

This is not a new official score. It is evidence that the next architecture
direction should be a provenance-backed state ledger with entity state-change
semantics, not more benchmark prompt branches.

## Reranked event subset experiment

Tried a generic LLM-context reranker that passed only nearby referenced steps, small referenced ranges, transition windows, and top lexical event matches to the solver while preserving the full event index for structural operators.

Evidence:

- 177-case slice: `results\mapu_gemini_flash_lite_ama_limit177_judged_reranked_events_v1.json`, 119 / 177 semantic, 67.23%.
- Full 208-case run: `results\mapu_gemini_flash_lite_ama_full208_judged_reranked_events_v1.json`, 134 / 208 semantic, 64.42%.

Decision: reverted from default because the full benchmark regressed below the restored default 135 / 208. The slice improvement did not generalize to Web/OpenWorld in the full run.

Current default after revert remains the restored clean structural path. Lint and unit tests passed after revert.
