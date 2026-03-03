# VertiRF Architecture

## 1. Goals

- Provide a standalone, reproducible implementation of three VRF methods:
  - decon (time-iteration)
  - corr (cross-correlation based)
  - stack (peak-window aligned stacking)
- Support multiple zero-phase filters with a unified interface.
- Provide serial/parallel execution for all three methods.
- Provide AI-agent-callable interface for scripted orchestration.
- Provide reproducible engineering benchmark dataset and replay workflow.

## 2. Non-Goals

- No GUI in this phase.
- No remote service deployment in this phase.
- No direct coupling with station or catalog retrieval pipelines.

## 3. Stage Isolation

- `station`: station retrieval and station metadata boundary.
- `catalog`: event retrieval and filtering boundary.
- `waveform`: signal processing and decon/corr/stack boundary.

## 4. Core Components

- `filters.zero_phase`
  - Build frequency-domain amplitude responses for all supported filters.
- `core.decon`
  - Single fast decon implementation (incremental residual update).
- `core.methods`
  - Unified method dispatcher: `decon/corr/stack`.
  - Corr smoothing/post-filter control.
  - Stack peak-window control.
- `waveform.synthetic`
  - Synthetic wavelet/response generation and convolution helpers.
- `agent.server`
  - JSON-RPC style command dispatcher for AI agents.
- `cli`
  - Human and automation entry point.
- `scripts.method_parallel_benchmark`
  - Serial-vs-parallel speedup benchmark for all methods.
- `scripts.generate_real_case_three_methods_wiggle`
  - Real-case visualization for decon/corr/stack.
- `scripts.check_native_backend`
  - Native (C/C++) backend environment diagnosis.

## 5. Data Flow

1. Input traces + source wavelet.
2. Zero-phase pre-filtering.
3. Method execution (`decon` or `corr` or `stack`).
4. Optional post-filtering (corr).
5. Optional batch parallel execution.
6. Validation output (metrics, benchmark, figures, status artifacts).

## 6. Performance Strategy

- Reuse FFT/filter preparation in method paths.
- Keep decon as a single fast path and validate serial/parallel numerical consistency.
- Use thread-level parallelism for independent traces.
- Emit per-method speed/consistency report against serial execution.

## 7. Native Backend Strategy (C/C++)

- Detect compiler toolchain availability (`cl/g++/gcc/cmake`).
- If unavailable, produce explicit status artifact and use optimized NumPy fallback.
- If available in future runs, attach build script and plug-in adapter without changing high-level APIs.

## 8. Verification Strategy

- Decon correctness:
  - closed-loop synthetic recovery checks.
- Corr correctness:
  - synthetic alignment/shape sanity checks with smoothing+post-filter coverage.
- Stack correctness:
  - synthetic peak-window alignment checks.
- Performance:
  - serial vs parallel benchmark on decon/corr/stack.
- Documentation:
  - README contains real-case three-method visualization.

## 9. CI Gates

GitHub Actions workflow enforces:
- Style check (`ruff`).
- Unit tests (`pytest`).
- Benchmark smoke run and artifact export (`benchmark_summary.json`).
