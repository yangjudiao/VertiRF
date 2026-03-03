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

## Task: Stack Prompt22-Compatible Convergence + Equivalent Speedup

## Step S1: Align Stack Semantics To Prompt22
- Status: Completed (2026-03-04)
- Actions:
  - Converge stack peak-window alignment reference from implicit center-based indexing to prompt22-compatible zero-reference indexing (configurable).
  - Keep single stack engine in `core/methods.py`; no alternate stack algorithm branches.
  - Wire optional stack zero-reference index through CLI/agent config.
- Acceptance Criteria (AC):
  - AC-S1.1: Stack output can exactly match prompt22 stack reference semantics when zero-reference index is provided.
  - AC-S1.2: Existing stack call sites without zero-reference index keep backward-compatible behavior.
- Execution Result:
  - `MethodConfig` 新增 `stack_zero_index`（可选）。
  - `core/methods.py` 的 stack 路径已对齐 prompt22 语义：
    - 峰值窗口按 `(sample_index - stack_zero_index) * dt` 解释（`stack_zero_index` 为空时保持原中心参考语义）。
    - 对齐目标索引由 `stack_zero_index` 控制（默认回退到 `n//2`，兼容旧调用）。
  - CLI/agent 参数打通：
    - `src/vertirf/cli.py` 新增 `--stack-zero-index`
    - `src/vertirf/agent/server.py` 新增 `stack_zero_index` 参数解析
  - 真实数据 stack-only 全量对比（185 events）结果：
    - `0.1-0.5`: `verdict=一致`, `mae=3.82e-10`, `max_abs=1.30e-08`
    - `0.1-0.35`: `verdict=一致`, `mae=3.96e-10`, `max_abs=1.44e-08`
    - 报告：`D:/works_2/seismic_data_retrieval_1/data/prompt22_real_vertirf_stackonly_parallel_20260304/stack_only_comparison_report.json`

## Step S2: Optimize Stack Efficiency and Parallel Throughput
- Status: Completed (2026-03-04)
- Actions:
  - Replace per-trace stack loop with batch-vectorized filter + peak detect + shift path.
  - Keep `jobs` parallel-call capability via chunked processing and stable output ordering.
  - Add strict legacy-equivalence + serial/parallel consistency tests.
- Acceptance Criteria (AC):
  - AC-S2.1: `tests/test_corr_stack.py` includes stack strict-equivalence and serial-vs-parallel checks.
  - AC-S2.2: stack benchmark reports `speedup_vs_legacy > 1.0x` with near-machine-precision consistency.
- Execution Result:
  - stack 核心实现优化：
    - 预计算滤波响应（每批一次），避免旧实现“每条 trace 重建一次频响”的重复开销。
    - 保留 `jobs` 并行调用能力；串并输出保持一致。
  - 测试增强（`tests/test_corr_stack.py`）：
    - 新增 `stack` 严格 legacy 等价测试（含 `stack_zero_index`）
    - 新增 `stack` 串并一致性严格测试
  - 新增基准脚本：
    - `scripts/benchmark_stack_legacy_equiv.py`
  - 基准结果：
    - `benchmark_stack_legacy_equiv_medium.json`: `speedup_vs_legacy=5.2763x`
    - `benchmark_stack_legacy_equiv_large.json`: `speedup_vs_legacy=7.7951x`
    - 一致性均为机器精度级等价（`mae=0`, `max_abs=0`, `flatten_corrcoef≈1.0`）

## Step S3: Gates, Real-data Full Stack Validation, GitHub Push
- Status: Completed (2026-03-04)
- Actions:
  - Run lint/tests/parallel benchmark/agent self-test.
  - Run real-data stack-only full validation against prompt22 reference bands.
  - Commit and push as the single canonical stack implementation.
- Acceptance Criteria (AC):
  - AC-S3.1: validation gates pass.
  - AC-S3.2: real-data 185-event stack comparison verdict is `一致` for both bands.
  - AC-S3.3: remote branch includes the new commit.
- Execution Result:
  - 闸门全部通过：
    - `ruff check src tests scripts examples`
    - `python -m pytest -q` (`13 passed`)
    - `python scripts/method_parallel_benchmark.py --out method_parallel_benchmark_summary.json`
    - `python -m vertirf.agent.server --self-test`
  - 本地提交已完成：
    - `86749c7 stack: align prompt22 semantics and optimize single-engine path`
    - `b1d96a4 docs: record stack gates and push retry status`
  - 远端推送状态：
    - 初次推送出现网络抖动（`connection reset / could not connect to server`），重试后已成功推送到 `origin/main`。
    - 远端范围：`2f4d145..b1d96a4`。
