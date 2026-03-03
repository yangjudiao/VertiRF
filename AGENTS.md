# AGENTS.md (VertiRF)

## Execution Rules

1. Plan before implementation.
2. Every step must define verifiable acceptance criteria.
3. If a step fails, diagnose root cause first, then retry.
4. Use at most two subagents.
5. Keep stage isolation:
   - `station` module: station retrieval and station metadata logic only.
   - `catalog` module: event catalog retrieval/filtering only.
   - `waveform` module: waveform processing/decon/filtering only.

## Coding Rules

1. Preserve reproducible baseline behavior in `core/decon.py`.
2. Add new performance paths without silently changing baseline outputs.
3. Keep zero-phase filters in `filters/zero_phase.py` only.
4. Add tests for every behavior change.
5. Document CLI changes in `README.md` and `tasks.md`.

## Validation Gates

1. `python -m pytest -q`
2. `python scripts/benchmark.py --out benchmark_summary.json`
3. `python -m vertirf.agent.server --self-test`
