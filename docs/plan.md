# Decon Prompt22-Compatible Fast Optimization Plan

## Meta
- Date: 2026-03-03
- Trigger: User requires decon to stay consistent with prompt22-style results while maximizing speed and keeping a single decon implementation.
- Scope: `src/vertirf/core/decon.py` and related tests/benchmark scripts in VertiRF only.

## Step 1: Implement Prompt22-Compatible Single Fast Engine
- Status: Completed (2026-03-03)
- Actions:
  - Refactor decon core to preserve prompt22 decon semantics:
    - same lag search window logic;
    - same amplitude update formula;
    - same phase-shift formula;
    - same stop criterion behavior.
  - Keep single decon implementation entry (`run_batch_decon`) with no mode split.
  - Apply equivalent speedups only (precomputed FFT terms, incremental residual update without changing math).
- Acceptance Criteria (AC):
  - AC-1.1: No baseline/optimized/ultra mode branches remain in decon API.
  - AC-1.2: Core loop semantics are preserved by construction and code review.
  - AC-1.3: Decon still supports serial and thread-parallel execution.
- Execution Result:
  - Updated `src/vertirf/core/decon.py` to prompt22-compatible semantics:
    - lag search defaults to prompt22 window (`[0, maxlag)` when `allow_negative_impulse=False`);
    - amplitude update keeps `amp = rw[idx] / dt`;
    - final phase shift keeps prompt22 frequency-domain formula.
  - Kept single decon entry `run_batch_decon(...)` (no mode branches).
  - Applied equivalent-only optimizations:
    - precompute source-filter FFT terms;
    - precompute single-impulse prediction kernel;
    - incremental residual update with circular shifted AXPY.

## Step 2: Add Consistency Tests Against Legacy-Compatible Reference
- Status: Completed (2026-03-03)
- Actions:
  - Add tests that compare the new decon engine against an in-test legacy-compatible reference implementation.
  - Validate numerical equivalence with strict tolerances.
  - Validate serial vs parallel numerical consistency.
- Acceptance Criteria (AC):
  - AC-2.1: New tests fail if semantic drift appears.
  - AC-2.2: `pytest` remains green.
- Execution Result:
  - Reworked `tests/test_decon.py`:
    - added strict legacy-reference equivalence test (`mae/max_abs/iterations`);
    - kept serial-vs-parallel numerical consistency gate;
    - added search-domain behavior test for `allow_negative_impulse`.
  - Test gate:
    - `python -m pytest -q` -> `11 passed`.

## Step 3: Add Efficiency + Consistency Benchmark Artifact
- Status: Completed (2026-03-03)
- Actions:
  - Add benchmark script that runs:
    - legacy-compatible reference path;
    - new optimized single fast path.
  - Output JSON with consistency metrics (`mae/max_abs/flatten_corrcoef`) and speedup.
- Acceptance Criteria (AC):
  - AC-3.1: Benchmark JSON is generated successfully.
  - AC-3.2: Reported speedup > 1.0x with near-machine-precision consistency.
- Execution Result:
  - Added benchmark script:
    - `scripts/benchmark_decon_legacy_equiv.py`
  - Medium benchmark artifact:
    - `benchmark_decon_legacy_equiv_medium.json`
    - `speedup_vs_legacy = 2.98597x`
    - consistency: `mae=3.78e-17`, `max_abs=1.44e-15`, `flatten_corrcoef≈1.0`
  - Large benchmark artifact:
    - `benchmark_decon_legacy_equiv_large.json`
    - `speedup_vs_legacy = 2.86895x`
    - consistency: `mae=2.55e-17`, `max_abs=1.22e-15`, `flatten_corrcoef=1.0`

## Step 4: Run Gates, Commit, and Push
- Status: Completed (2026-03-03)
- Actions:
  - Run lint/tests/benchmark/self-test as applicable.
  - Commit and push to `origin/main`.
- Acceptance Criteria (AC):
  - AC-4.1: `ruff` and `pytest` pass.
  - AC-4.2: Benchmark artifact exists and shows both consistency and speedup.
  - AC-4.3: Remote contains the new commit.
- Execution Result:
  - `ruff check src tests scripts examples` passed.
  - `python -m pytest -q` passed (`11 passed`).
  - `python -m vertirf.agent.server --self-test` passed.
  - Commit and push status: completed (see git log for commit hash).

## Task: Corr Prompt22-Compatible Convergence + Equivalent Speedup

## Step C1: Align Corr Semantics To Prompt22 While Keeping Single Engine
- Status: Completed (2026-03-03)
- Actions:
  - Refactor corr path in `core/methods.py` to prompt22-compatible retrieval:
    - spectral cross-correlation (`D * conj(B)`),
    - optional denominator division with source-power smoothing and water-level,
    - zero-phase bandpass multiply,
    - output shift and max-abs row normalization.
  - Keep `run_batch_method` single corr engine (no baseline/optimized branch split).
  - Preserve `jobs` parallel-call capability.
- Acceptance Criteria (AC):
  - AC-C1.1: Corr has one algorithmic path regardless of mode flag.
  - AC-C1.2: Corr config supports prompt22-equivalent controls.
  - AC-C1.3: `jobs>1` remains callable and stable.
- Execution Result:
  - Refactored corr in `src/vertirf/core/methods.py` to prompt22-compatible spectral retrieval:
    - `rf_spec = D * conj(B)`
    - optional divide by smoothed source-power denominator with water-level floor
    - multiply zero-phase filter response
    - time shift and max-abs row normalization
  - Corr remains single-engine regardless of mode flag.
  - Added prompt22-relevant corr controls into `MethodConfig`:
    - `corr_divide_denom`
    - `corr_water_level`
    - `corr_shift_sec`
  - Updated CLI/agent wiring:
    - `src/vertirf/cli.py`
    - `src/vertirf/agent/server.py`

## Step C2: Consistency + Parallel Tests
- Status: Completed (2026-03-03)
- Actions:
  - Add strict legacy-reference corr equivalence tests.
  - Keep/extend corr serial-vs-parallel consistency test.
  - Add decon/corr parallel-call smoke validation.
- Acceptance Criteria (AC):
  - AC-C2.1: Corr legacy-equivalence tests pass with near-machine-precision diff.
  - AC-C2.2: Corr/decon serial-vs-parallel consistency gates pass.
- Execution Result:
  - Reworked `tests/test_corr_stack.py`:
    - strict corr legacy-reference equivalence test
    - corr serial-vs-parallel strict consistency test
    - corr mode-flag convergence test (`baseline` vs `optimized`)
    - retained stack alignment test
  - Decon parallel consistency remains covered in `tests/test_decon.py`.
  - Gate snapshot:
    - `python -m pytest -q tests/test_corr_stack.py tests/test_agent.py tests/test_decon.py` -> `10 passed`.

## Step C3: Efficiency Benchmark + Real-Data Consistency Check
- Status: Completed (2026-03-03)
- Actions:
  - Add corr benchmark script comparing legacy-reference corr vs optimized single engine.
  - Produce JSON artifacts for medium/large workloads.
  - Run prompt22-real sample consistency check against reference implementation.
- Acceptance Criteria (AC):
  - AC-C3.1: Benchmark artifacts include consistency and speedup fields.
  - AC-C3.2: Speedup > 1.0x with strict consistency metrics.
- Execution Result:
  - Added corr benchmark script:
    - `scripts/benchmark_corr_legacy_equiv.py`
  - Medium benchmark artifact:
    - `benchmark_corr_legacy_equiv_medium.json`
    - `speedup_vs_legacy = 1.0640x`
    - consistency: `mae=0`, `max_abs=0`, `flatten_corrcoef=1.0`
  - Large benchmark artifact:
    - `benchmark_corr_legacy_equiv_large.json`
    - `speedup_vs_legacy = 1.1425x`
    - consistency: `mae=0`, `max_abs=0`, `flatten_corrcoef≈1.0`
  - Prompt22 real-data sample consistency artifact (30 events):
    - `assets/corr_prompt22_real_consistency_20260303.json`
    - `pair_count=1800`, `mae_mean=0`, `max_abs_max=0`

## Step C4: Gates + GitHub Push
- Status: Completed (2026-03-03)
- Actions:
  - Run `ruff`, `pytest`, method benchmark, agent self-test.
  - Commit and push to `origin/main`.
- Acceptance Criteria (AC):
  - AC-C4.1: all gates pass.
  - AC-C4.2: remote includes new commit hash.
- Execution Result:
  - `ruff check src tests scripts examples` passed.
  - `python -m pytest -q` passed (`11 passed`).
  - `python -m vertirf.agent.server --self-test` passed.
  - `python scripts/method_parallel_benchmark.py --out method_parallel_benchmark_summary.json` executed successfully, confirming decon/corr/stack parallel-call capability.
  - Commit and push: completed (see git log for commit hash).
