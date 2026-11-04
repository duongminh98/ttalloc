# ttalloc

Code for the bachelor thesis *Adaptive allocation of inference compute for large reasoning models*
(Dương Tuấn Minh).

The problem: there are $N$ questions and one shared token budget $B$. While a reasoning model is thinking, the
allocator decides at each decision point which questions keep reasoning and which ones stop and answer. The goal is
to maximise the number of correct answers without ever spending more than $B$ tokens.

The main model is Qwen3-1.7B, served with vLLM. Three methods are implemented:

| Id | Method | Role |
|---|---|---|
| B2 | DEER early exit with a per-question allowance calibrated to the budget | Baseline |
| M2 | Adaptive token price: a learned estimator predicts the accuracy gain per token of continuing; the price clears the shared budget | Proposed method |
| M3 | M2 with the estimator replaced by empirical accuracy curves from other seeds (deliberate look-ahead) | Diagnostic upper reference |

The other methods in the plan (B0, B1, B3–B5, M1, O1) are deferred to phase P5.

## Documentation

The project documents are written in Vietnamese.

- Specification: [docs/PLAN.md](docs/PLAN.md). Method: [docs/Bao-cao-Phuong-phap-chi-tiet.md](docs/Bao-cao-Phuong-phap-chi-tiet.md).
- Progress: [docs/progress.md](docs/progress.md). Design decisions (ADRs): [docs/decisions.md](docs/decisions.md).
- Runbook for every GPU and CPU step, P0–P4: [ops/RUNBOOK.md](ops/RUNBOOK.md). Rules for coding agents: [AGENTS.md](AGENTS.md).

## Layout

```
src/ttalloc/
  engine/      vLLM generator, scripted generator for tests, segment cutting at reasoning markers, LiveSource
  logging/     trace logger (resumable), Parquet store, log validation, probe/answer studies
  features/    next-event logic on logged traces, canonical decision points, features, empirical curves
  estimator/   training data, LightGBM / logistic estimator with isotonic calibration, rho and h*, evaluation
  controller/  budget pool and guard, the Runner shared by simulation and live runs
  policies/    B2 (DEER) and M2/M3 (adaptive price)
  sim/         LoggedSource, precomputed tables, rho providers, simulation grid, B2 calibration, tuning
  online/      live runs on vLLM, timing
  eval/        statistics, metrics, decision gates G1–G3, report tables and figures
  baseline/    full generation, official DEER repo, throughput, segmented-vs-full equivalence test
configs/       base defaults, models, datasets, prompts, experiments (g1, tune_val, e2_main, e4/*, e5_online)
ops/           Kaggle and Vast.ai tooling, Hugging Face sync, runbook
tests/         unit and integration tests (CPU), GPU tests (tests/gpu)
```

## Setup and tests (no GPU needed)

Requires [uv](https://docs.astral.sh/uv/) and git.

```bash
uv sync                          # Python 3.12, core and dev dependencies
uv run pytest                    # CPU tests (gpu, slow and net markers are skipped by default)
uv run ruff check src tests ops && uv run mypy src
```

On Windows, set `UV_LINK_MODE=copy` if uv cannot create hard links.

## Main commands

All commands accept `--exp <experiment yaml>` and `--set key=value` overrides.

| Step | Command | Runs on |
|---|---|---|
| Splits | `ttalloc splits build`, `ttalloc splits verify` | CPU (network) |
| Baselines (P1) | `ttalloc baseline full / deer / throughput`, `ttalloc equivalence` | GPU |
| Logging (P1–P3) | `ttalloc log --split val --seeds 0-7`, `ttalloc validate-logs` | GPU |
| Studies | `ttalloc study answer-mode / probe-noise` (GPU), `ttalloc study curves` (CPU) | |
| Estimator (P2) | `ttalloc estimator train`, `ttalloc estimator eval --split val` | CPU |
| Simulation (P2–P3) | `ttalloc table build`, `ttalloc sim run`, `ttalloc tune` | CPU |
| Test-set discipline (P3) | `ttalloc lock --exp ...`, `ttalloc eval-test --exp configs/locked/... --approved-by <name>` | CPU or GPU |
| Live runs (P4) | `ttalloc online run` | GPU |
| Reports and gates | `ttalloc report`, `ttalloc gate g1 / g2 / g3` | CPU |

Test splits (`test_*`) are only touched through `eval-test` with a locked config and a named approver. The single
exception is the baseline reproduction in P1, which also requires `--approved-by`.

## Running on GPUs

- **Kaggle (T4, free):** development and GPU tests.
  `uv run --group ops python ops/kaggle/push.py <script> --no-wait` pushes the code as a private dataset and runs
  `ops/kaggle/<script>.py`. `--watch-only --fail-fast` then follows the log while the kernel runs.
- **Vast.ai (RTX 3090, bf16):** baseline reproduction, data logging and live runs. `ops/vast/vast.py` searches,
  creates, watches and destroys instances over the Vast REST API (no SSH needed). Each instance runs a job script
  from `ops/vast/jobs/`. On failure the instance waits for fixed code instead of shutting down.
- **Data exchange:** code, logged traces, estimators and run outputs move between machines through a private
  Hugging Face dataset (`ops/hf_sync.py`).

Credentials (`KAGGLE_API_TOKEN`, `HF_TOKEN`, `TTALLOC_HF_REPO`, `VAST_API_KEY`) are read from user environment
variables and are never stored in the repository.

## Status

P0–P4 code and CPU tests are complete. No GPU step has been run yet; [docs/progress.md](docs/progress.md) tracks each
step.
