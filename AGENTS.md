# AGENTS.md (VertiRF)

## Execution Rules

1. Plan before implementation.
2. Every step must define verifiable acceptance criteria.
3. If a step fails, diagnose root cause first, then retry.
4. Use at most two subagents.
5. Keep stage isolation:
   - `station` module: station retrieval and station metadata logic only.
   - `catalog` module: event catalog retrieval/filtering only.
   - `waveform` module: waveform processing, decon/corr/stack only.

## Coding Rules

1. Keep decon as a single fast implementation in `core/decon.py` (no multiple decon mode branches).
2. Validate serial/parallel consistency whenever decon internals change.
3. Keep zero-phase filters in `filters/zero_phase.py` only.
4. Add tests for every behavior change.
5. Document CLI changes in `README.md` and `tasks.md`.
6. For corr/stack:
   - keep smoothing bandwidth and post-filter options explicit;
   - keep stack peak window parameters explicit and test-covered.

## Native Backend Rule

1. C/C++ backend is optional and environment-dependent.
2. Always emit `assets/native_backend_status.json` with:
   - compiler/toolchain detection result;
   - whether native backend is available;
   - fallback backend decision.

## Validation Gates

1. `ruff check src tests scripts examples`
2. `python -m pytest -q`
3. `python scripts/method_parallel_benchmark.py --out method_parallel_benchmark_summary.json`
4. `python -m vertirf.agent.server --self-test`
