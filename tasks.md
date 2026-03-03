# VertiRF Tasks

## Phase 1: Bootstrap

- [x] Create standalone project directory and git repository.
- [x] Create open-source baseline files (README, LICENSE, AGENTS, architecture, tasks).
- Acceptance criteria:
  - Repository is runnable locally.
  - Documentation files exist and are readable.

## Phase 2: Core Decon Algorithm

- [x] Implement single fast time-iteration decon engine.
- [x] Optimize iteration update path with prepared reusable state and incremental residual update.
- [x] Implement impulse negative-time toggle.
- Acceptance criteria:
  - `python -m py_compile` passes for all source files.
  - CLI exposes one decon execution path (no baseline/optimized/ultra branch confusion).

## Phase 3: Filter Extensions

- [x] Implement zero-phase Gaussian filter.
- [x] Implement zero-phase Butterworth bandpass filter.
- [x] Implement zero-phase Raised-Cosine bandpass filter.
- [x] Implement zero-phase Tukey bandpass filter.
- Acceptance criteria:
  - All filter types callable from CLI.
  - Tests cover all filter modes.

## Phase 4: Corr/Stack Method Integration (Prompt27)

- [x] Add unified method dispatcher for `decon/corr/stack`.
- [x] Add corr smoothing bandwidth parameter and corr post-filter selection.
- [x] Converge corr to single fast engine (remove baseline/optimized algorithm split).
- [x] Add stack peak-window parameterization.
- [x] Add serial/parallel execution support for corr and stack.
- Acceptance criteria:
  - CLI supports `--method {decon,corr,stack}`.
  - Corr and stack run with `jobs=1` and `jobs>1`.

## Phase 5: Validation and Benchmark

- [x] Add synthetic correctness tests for corr and stack.
- [x] Keep decon tests green.
- [x] Run method-level serial-vs-parallel benchmark for three methods.
- Acceptance criteria:
  - `pytest` passes.
  - `method_parallel_benchmark_summary.json` exists and includes speedup fields for all methods.

## Phase 6: Native Backend Status

- [x] Detect C/C++ toolchain availability and write status artifact.
- [x] Provide fallback decision trace when toolchain unavailable.
- Acceptance criteria:
  - `assets/native_backend_status.json` exists and is human-readable.

## Phase 7: Real Case Visualization

- [x] Generate one real-case three-method wiggle figure.
- [x] Update README homepage section with figure and reproducible command.
- Acceptance criteria:
  - `assets/real_case_three_methods_wiggle.png` exists.
  - README references and explains the figure.

## Phase 8: Agent Interface and CI

- [x] Implement JSON-RPC style agent server.
- [x] Add self-test mode.
- [x] Add GitHub Actions CI workflow.
- [x] Add style check gate (`ruff`).
- [x] Add minimal MCP client example.
- Acceptance criteria:
  - `python -m vertirf.agent.server --self-test` passes.
  - CI workflow runs style + tests + benchmark smoke.

## Phase 9: Engineering Benchmark Reproducibility

- [x] Add engineering dataset build script from event NPZ sources.
- [x] Add reproducible benchmark runner on saved dataset.
- [x] Document data build and replay commands in README.
- Acceptance criteria:
  - `engineering_dataset.npz` is generated with metadata summary.
  - `repro_report.json` includes runtime speedup and serial/parallel consistency.
