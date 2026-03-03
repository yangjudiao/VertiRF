# VertiRF

English

VertiRF is a standalone open-source project for time-iteration deconvolution in vertical receiver function workflows. It keeps a reproducible baseline implementation and provides an optimized execution path with zero-phase filter extensibility and batch parallelism.

Chinese

VertiRF 是一个独立开源项目，用于垂向接收函数流程中的 time-iteration 反卷积。项目同时保留可复现的 baseline 实现，并提供优化执行路径，支持多种零相位滤波器与批量并行。

## Features | 功能特性

- Baseline time-iteration decon implementation compatible with existing Gaussian workflow.
- Optimized batch mode with reusable FFT preparation and thread-level parallel execution.
- Zero-phase filter options:
  - `gaussian`
  - `butterworth_bandpass`
  - `raised_cosine_bandpass`
  - `tukey_bandpass`
- Configurable impulse picking behavior:
  - `allow_negative_impulse=true`: full-lag search
  - `allow_negative_impulse=false`: non-negative lag only
- AI-agent-callable JSON-RPC interface (`vertirf.agent.server`).
- Engineering benchmark dataset builder and reproducible benchmark runner.

## Project Layout | 目录结构

```text
VertiRF/
  src/vertirf/
    station/
    catalog/
    waveform/
    filters/
    core/
    agent/
    cli.py
  tests/
  scripts/
  examples/
  assets/
  .github/workflows/
  architecture.md
  tasks.md
  AGENTS.md
```

## Quick Start | 快速开始

```bash
cd D:\works_2\VertiRF
python -m pip install -e .
```

Run synthetic batch decon (optimized):

```bash
python -m vertirf.cli run-synthetic \
  --mode optimized \
  --filter-type butterworth_bandpass \
  --low-hz 0.1 --high-hz 0.8 \
  --allow-negative-impulse true \
  --jobs 4 \
  --traces 64 --samples 1024
```

Run benchmark:

```bash
python scripts/benchmark.py --out benchmark_summary.json --jobs 4
```

Run tests:

```bash
python -m pytest -q
```

Run style check:

```bash
ruff check src tests scripts examples
```

Agent self-test:

```bash
python -m vertirf.agent.server --self-test
```

## MCP Client Example | MCP 客户端示例

Minimal JSON-RPC client example:

```bash
python examples/mcp_client_example.py
```

This script starts `vertirf.agent.server`, sends `ping` and `run_decon_synthetic` requests, and prints responses.

## Engineering Benchmark Dataset | 工程基准数据集

Build dataset from existing event NPZ files:

```bash
python scripts/build_engineering_dataset.py \
  --input-dir D:\works_2\seismic_data_retrieval_1\data\prompt19\p14_like_lowpass_t200\convolved_npz \
  --out data/engineering_benchmark/engineering_dataset.npz \
  --events 12 --stations 20 --component z --seed 20260303
```

Run reproducible engineering benchmark:

```bash
python scripts/run_engineering_repro.py \
  --dataset data/engineering_benchmark/engineering_dataset.npz \
  --out data/engineering_benchmark/repro_report.json \
  --jobs 4 --repeat 2 --filter-type butterworth_bandpass --allow-negative-impulse
```

## CI | 持续集成

GitHub Actions workflow: `.github/workflows/ci.yml`

Pipeline includes:
- `ruff` style check
- `pytest`
- benchmark smoke run and artifact upload (`benchmark_summary.json`)

## Pipeline Overview | 流程示意

![VertiRF overview](assets/overview_pipeline.png)

## Documentation | 文档

- [architecture.md](architecture.md): requirements and technical architecture.
- [tasks.md](tasks.md): staged development tasks and acceptance criteria.
- [AGENTS.md](AGENTS.md): AI agent execution rules for this project.
