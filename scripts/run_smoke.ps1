uv run --extra all memorybench catalog
uv run --extra all memorybench run --benchmarks memoryarena ama_bench --adapter null --limit 1 --out results/null_smoke.json
uv run --extra all memorybench run --benchmarks memoryarena ama_bench --adapter oracle --allow-oracle --limit 1 --out results/oracle_smoke.json
