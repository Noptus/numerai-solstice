# Numerai Solstice

[![Research checks](https://github.com/Noptus/numerai-solstice/actions/workflows/tests.yml/badge.svg)](https://github.com/Noptus/numerai-solstice/actions/workflows/tests.yml)

Reproducible **Numerai v6** research on a local CPU: purged walk-forward validation, official neutral correlation, fixed-benchmark contribution, and a recorded model-selection process.

AI-assisted implementation. The repository makes its data, tests, search budget and evaluation boundaries inspectable. Historical tournament scores do not establish live investment performance.

## Measured research

| Study | Mean historical NCORR | Interpretation |
| --- | ---: | --- |
| [First frozen study](reports/README.md) | 0.01689 | Above its small-feature control (0.00410); below the supplied public benchmark |
| [Full-feature follow-up](reports/WIDE_V2.md) | 0.01837 | Exploratory reuse of the same historical period; above its control (0.00120) |
| Supplied public benchmark | 0.02784 | Higher than both custom selections |

The follow-up's point-estimate gain over the first study is small and its interval includes zero. These results support a reproducible research case study; they do not establish a live edge. Both reports retain losing candidates, all calibration trials and uncertainty.

The [rolling/capacity study](reports/ROLLING_INTERIM.md) is saved as an **incomplete experiment**, with all completed checkpoints retained. On its first 53-date development window, the fixed-budget candidate scored **0.03304**, below the public benchmark at **0.03802**. These are different dates from the table above. No final selection or later evaluation was performed; this is not a benchmark win.

A trusted model bundle produced **7,153 dated predictions in 0.391 seconds** inside the inference function, including file/hash checks. This is a local timing on the reference Mac; process startup is excluded. Raw data and model binaries stay outside Git.

## Run it

Python 3.12+, macOS or Linux, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The default experiment uses six CPU threads, one model at a time and at most 750 training observations per date. The measured reference machine has 32 GiB RAM. Raw data and model artifacts stay outside Git.

```bash
git clone https://github.com/Noptus/numerai-solstice.git
cd numerai-solstice
uv sync --frozen --extra dev --no-editable
uv run --frozen --no-editable solstice plan --config configs/wide.toml
uv run --frozen --no-editable pytest

# Public datasets; no account credentials required. Pins the round on first use.
uv run --frozen --no-editable solstice download --include-live

# All stages, with integrity-checked checkpoints for resuming the same run.
uv run --frozen --no-editable solstice run --output runs/mac-wide-v6
```

Download size, model fit times, historical scores and uncertainty are recorded in the [reference experiment](reports/README.md). Budget several hours for a new machine; the CLI logs each fit and stage. Stop and resume the same command without discarding completed fits. Changing code, dependencies, configuration or snapshot requires a fresh run directory. Source-aware uv cache keys rebuild the local wheel after packaged research code changes.

## Research design

```mermaid
flowchart LR
    A[Pinned public snapshot] --> B[Three purged walk-forward folds]
    B --> C[Shortlist from CV only]
    C --> D[2024 calibration: blend and neutralization]
    D --> E[Freeze selection and trial log]
    E --> F[Reserved historical holdout]
    F --> G[Dated prediction export]
```

| Component | What is compared or enforced |
| --- | --- |
| Baselines | Small-feature LightGBM and medium-feature ridge regression |
| Tree candidates | Regularization, extra trees, slower learning and recency weighting |
| Target diversity | Explicit Jupiter, factor-feature-neutral and Camille 60-day targets |
| Era boosting | Recorded in-sample reweighting ablation; no assumed benefit |
| Sampling | Deterministic ticker/date hashes, independent of labels |
| Validation | Earlier dates only; 90 business days purged before every outer and inner evaluation boundary |
| Early stopping | Separate purged block inside training; outer labels never select tree counts |
| Ensemble | CV shortlist; nonnegative greedy blend and exposure neutralization tuned on calibration only |
| Scoring | NCORR and contribution to `lgbm_jupiter_60`, using supplied v6 neutralizers |
| Evidence | Per-date scores, every calibration trial, SHA-256 data/model hashes and dependency versions |

The purge covers the explicit 60-day targets, the documented lag, and a conservative buffer. It uses weekdays rather than a bespoke exchange calendar. Scores use every eligible stock in each evaluation date; only training is sampled. Numerai's `data_type=test` rows are excluded.

The reserved holdout is historical public data, not a claim that nobody has previously examined it. Opening it consumes it. A better score after changing the design needs fresh prospective evidence. Block-bootstrap intervals use 16-date blocks to reflect overlapping outcomes approximately; they are not a guarantee of independence.

## Inspect the implementation

| Path | Purpose |
| --- | --- |
| [`configs/wide.toml`](configs/wide.toml) | Seven-candidate full-feature follow-up, with explicit reused-evaluation status |
| [`research/data.py`](research/data.py) | Pinned downloads, SHA-256 integrity, schema and key validation |
| [`research/validation.py`](research/validation.py) | Date purging, deterministic sampling and weighting |
| [`research/models.py`](research/models.py) | Training-only preprocessing, inner stopping and model fits |
| [`research/scoring.py`](research/scoring.py) | Official-metric parity, reusable neutralizer factorization and uncertainty |
| [`research/runner.py`](research/runner.py) | Checkpoints, staged selection and holdout access journal |
| [`tests/`](tests/) | Numerical parity, isolation failures and real synthetic-data CPU integration |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Technique sources, assumptions and next experiments |

The loader retains medium-set columns for exposure postprocessing, even in a small-only or full-feature custom plan. Evaluation stocks remain complete; training keys are preselected before feature batches are materialized.

Individual stages are available as `solstice train --stage cv`, `score --stage cv`, `train --stage calibration`, `score --stage calibration`, `select`, and `finalize`. Use the same `--config`, `--data` and `--output` throughout. Running `finalize` again returns its saved result rather than evaluating the holdout again.

A prediction export refits the frozen selection on eligible history from the pinned round:

```bash
uv run --frozen --no-editable python -m scripts.export_live \
  --run runs/mac-wide-v6 --output runs/mac-wide-v6/live-export
```

After that export, inference can use a separate live-only snapshot with no training labels or refit:

```bash
uv run --frozen --no-editable python -m scripts.download_live --data data/next-live
uv run --frozen --no-editable python -m scripts.predict_bundle \
  --bundle runs/mac-wide-v6/live-export --data data/next-live \
  --output runs/next-live-predictions.csv
```

Use a fresh data directory for each new round; existing snapshots stay pinned. Inference verifies model, source, dependency and live-file hashes and preserves every ticker/date key.

Bundles require the recorded source and dependencies. The first study is preserved by `v0.6.0-reference`, and the full-feature follow-up by `v0.6.1-research`; use the matching tagged environment for each dated bundle. The current source produces new bundles.

The export writes dated ticker/date predictions and an integrity manifest. It does not upload, stake or trade. A historical snapshot is not a current-round submission. Check current Numerai requirements before independently arranging deployment. Pickled models are local trusted artifacts; do not load checkpoints from untrusted sources.

## Compatibility and reuse

The v6 path uses `date` / `numerai_ticker`, explicit `target_jupiter_60`, numeric feature sets and supplied neutralizer matrices. Text `feature_country` is excluded. Restored continuous features may contain missing values: ridge fits its imputer/scaler on training only; trees use native missing-value handling.

The original `solstice.py`, `config.py` and `validation.py` remain as a **legacy v5.2 research implementation**, documented in [the legacy guide](docs/LEGACY_V5.md). Its Spearman diagnostics are not v6 NCORR. Public v5 training downloads were retired; that path needs cached compatible data and its older optional dependencies. Use the `solstice` console command or `python -m research` for v6.

No standalone source license has been granted in this repository. Dataset and dependency licenses remain with their owners.
