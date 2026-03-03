# VertiRF Architecture

## 1. Goals

- Provide a standalone, reproducible implementation of time-iteration deconvolution.
- Support multiple zero-phase filters with a unified interface.
- Provide performance optimization by reusable preparation and batch parallel execution.
- Provide AI-agent-callable interface for scripted orchestration.
- Provide reproducible engineering benchmark dataset and replay workflow.

## 2. Non-Goals

- No GUI in this phase.
- No remote service deployment in this phase.
- No direct coupling with station or catalog retrieval pipelines.

## 3. Stage Isolation

- `station`: station retrieval and station metadata boundary.
- `catalog`: event retrieval and filtering boundary.
- `waveform`: signal processing and decon boundary.

## 4. Core Components

- `filters.zero_phase`
  - Build frequency-domain amplitude responses for all supported filters.
- `core.decon`
  - Baseline decon function.
  - Optimized decon function with prepared reusable state.
  - Batch runner with optional parallel execution.
- `waveform.synthetic`
  - Synthetic wavelet/response generation and convolution helpers.
- `agent.server`
  - JSON-RPC style command dispatcher for AI agents.
- `cli`
  - Human and automation entry point.
- `scripts.build_engineering_dataset`
  - Build fixed benchmark dataset from event NPZ corpus.
- `scripts.run_engineering_repro`
  - Replay baseline/optimized benchmark on fixed dataset.

## 5. Data Flow

1. Input traces + source wavelet.
2. Zero-phase filter response construction.
3. Time-iteration deconvolution (baseline or optimized).
4. Optional batch parallel execution.
5. Validation output (metrics, benchmark, test logs).

## 6. Performance Strategy

- Reuse filter and wavelet FFT state across traces.
- Keep baseline path untouched for regression comparison.
- Use thread-level parallelism for independent traces.

## 7. Verification Strategy

- Closed-loop recovery tests:
  - source * response -> observed -> decon -> recovered response/fit.
- Behavior tests:
  - `allow_negative_impulse` on/off.
  - filter type coverage.
- Benchmark:
  - baseline vs optimized runtime and speedup.
- Repro benchmark:
  - fixed dataset replay for reproducible performance tracking.

## 8. CI Gates

GitHub Actions workflow enforces:
- Style check (`ruff`).
- Unit tests (`pytest`).
- Benchmark smoke run and artifact export (`benchmark_summary.json`).
