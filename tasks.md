# VertiRF Tasks

## Phase 1: Bootstrap

- [x] Create standalone project directory and git repository.
- [x] Create open-source baseline files (README, LICENSE, AGENTS, architecture, tasks).
- Acceptance criteria:
  - Repository is runnable locally.
  - Documentation files exist and are readable.

## Phase 2: Core Algorithm

- [x] Implement baseline time-iteration decon.
- [x] Implement optimized path with prepared reusable state.
- [x] Implement impulse negative-time toggle.
- Acceptance criteria:
  - `python -m py_compile` passes for all source files.
  - CLI supports baseline/optimized and impulse toggle options.

## Phase 3: Filter Extensions

- [x] Implement zero-phase Gaussian filter.
- [x] Implement zero-phase Butterworth bandpass filter.
- [x] Implement zero-phase Raised-Cosine bandpass filter.
- [x] Implement zero-phase Tukey bandpass filter.
- Acceptance criteria:
  - All filter types callable from CLI.
  - Tests cover all filter modes.

## Phase 4: Validation and Benchmark

- [x] Add closed-loop correctness tests.
- [x] Add impulse-toggle behavior tests.
- [x] Run benchmark and generate summary report.
- Acceptance criteria:
  - `pytest` passes.
  - `benchmark_summary.json` includes baseline and optimized runtime and speedup.

## Phase 5: Agent Interface

- [x] Implement JSON-RPC style agent server.
- [x] Add self-test mode.
- Acceptance criteria:
  - `python -m vertirf.agent.server --self-test` passes.

## Phase 6: Documentation Closure

- [x] Complete bilingual README with usage and overview diagram.
- [x] Keep architecture/tasks/AGENTS aligned with implemented code.
- Acceptance criteria:
  - A new user can run quickstart, tests, and benchmark without hidden steps.

## Phase 7: CI and Example Clients

- [x] Add GitHub Actions CI workflow.
- [x] Add style check gate (`ruff`).
- [x] Add minimal MCP client example.
- Acceptance criteria:
  - CI workflow runs style check + tests + benchmark smoke.
  - Example client can call `ping` and `run_decon_synthetic`.

## Phase 8: Engineering Benchmark Reproducibility

- [x] Add engineering dataset build script from event NPZ sources.
- [x] Add reproducible benchmark runner on saved dataset.
- [x] Document data build and replay commands in README.
- Acceptance criteria:
  - `engineering_dataset.npz` is generated with metadata summary.
  - `repro_report.json` includes runtime speedup and baseline/optimized consistency.
