# Numerai v6 reference experiment

The later [full-feature follow-up](WIDE_V2.md) reuses this historical period and is labeled exploratory.

Completed 2026-10-22 on **Apple M5**, 32 GiB RAM, using **6 CPU threads**. Public snapshot round **1381**; **9** predeclared candidates and **3** purged walk-forward folds. AI-assisted implementation.

## Reserved historical evaluation

**2025-06-06 through 2026-07-17**, 59 evaluation dates. All eligible stocks are scored on the same dates. Scores are not trading returns.

| Model | Mean NCORR | 95% interval for mean NCORR | Mean fixed-benchmark contribution |
| --- | ---: | --- | ---: |
| Frozen selection | 0.016886 | [0.005735, 0.028112] | -0.000909 |
| Simple small-feature baseline | 0.004105 | [-0.009156, 0.017000] | -0.005546 |
| Public `lgbm_jupiter_60` | 0.027842 | [0.017444, 0.039531] | 0.000000 |

Against the simple baseline, the paired mean NCORR difference is **+0.012782** (95% block-bootstrap interval [+0.007213, +0.017587]). The historical paired gain is positive under this approximate interval.

Against the public benchmark, the paired mean NCORR difference is **-0.010956** (95% block-bootstrap interval [-0.017846, -0.004693]). The selection underperforms under this approximate interval.

![Historical NCORR paths](holdout.svg)

Intervals use 16-date circular blocks and 2,000 draws. They approximate serial dependence from overlapping 60-day outcomes; they do not correct for every research decision or establish a live edge. Contribution is measured against the fixed public benchmark, whose contribution to itself is zero, rather than Numerai's changing meta-model.

## Selection fixed before holdout access

CV ranking: mean NCORR minus 0.25 × between-date standard deviation. The shortlist was **era-boost-ablation, balanced-medium, slow-medium, factor-target, ridge-medium**. Calibration used 2024-01-05 through 2024-12-27 with **65 recorded trials**; the reserved evaluation starts only after a further maturity gap.

Selected weights: **`balanced-medium` 22.2%, `ridge-medium` 44.4%, `slow-medium` 33.3%**. Feature-neutralization proportion: **0.50**, against the top 50 contemporaneously exposed medium-set numeric features, chosen without targets.

The [selection](reference-v6-1381/selection.json), [complete calibration trial log](reference-v6-1381/calibration_trials.json) and [holdout access journal](reference-v6-1381/holdout_access.json) record this boundary. The historical holdout is now consumed. Changing the model after reading these results requires fresh prospective evidence; public historical data is not guaranteed never to have informed earlier work.

## All walk-forward candidates

These are the complete CV results, including losing candidates. Tree counts were selected using purged inner stopping blocks, then the models were refitted on eligible training observations. The baseline is our small-feature control, not an exact reproduction of Numerai's 200,000-row example.

| Candidate | Mean NCORR | Date-score std | Mean benchmark contribution | CV fit minutes |
| --- | ---: | ---: | ---: | ---: |
| `era-boost-ablation` | 0.028805 | 0.017797 | 0.002953 | 4.48 |
| `balanced-medium` | 0.028380 | 0.016408 | 0.003527 | 2.58 |
| `slow-medium` | 0.027385 | 0.017774 | 0.001167 | 4.47 |
| `factor-target` | 0.025689 | 0.020928 | 0.006860 | 2.75 |
| `ridge-medium` | 0.017028 | 0.015214 | -0.002069 | 5.54 |
| `extra-medium` | 0.015154 | 0.015016 | -0.002107 | 2.98 |
| `camille-target` | 0.013732 | 0.018471 | 0.001126 | 2.89 |
| `recent-medium` | 0.012726 | 0.017891 | 0.000215 | 1.03 |
| `baseline-small` | 0.006201 | 0.018021 | -0.004145 | 0.35 |

## Measured compute and data

- Research wall time: **55.5 minutes**, including sampling, prediction, scoring and calibration; excludes download and later export/reporting.
- Model fits: **37**, totaling **49.3 fit minutes**. [Fit metadata](reference-v6-1381/model_fits.json) includes row counts, dates, tree counts, seeds and hashes.
- Largest fit: **1,728,000** training observations; deterministic cap of **1,500** per date. Evaluation stocks were not sampled.
- Snapshot files: **7.69 GiB**, including supplied neutralizers and the dated live preview.
- Source, exact dependency versions, complete configuration and dataset hashes: [frozen plan](reference-v6-1381/plan.json). Training source commit: `6fa1500222fbb9fcde72775bdc3898de476d8a5d`. Use the source hashes recorded with this experiment.
- Official-metric parity on 6,092 real stocks and 300 neutralizer columns: maximum absolute difference **3.24e-10**, recorded in the parity check.

The frozen selection was also refitted for the pinned **2026-10-20** snapshot: **7,153** ticker/date predictions, with model and CSV hashes in [`live_export.json`](reference-v6-1381/live_export.json). This dated local export was not uploaded, staked or traded.

Local bundle inference also produced 7,153 predictions in **0.364 seconds** inside the inference function, including file and hash checks. [Timing and provenance](reference-v6-1381/inference.json). Process startup is outside that timing.

## Reproduce and inspect

For exact reproduction, use the `v0.6.0-reference` tag, which includes the matching research source and helpers. From that repository root, after `uv sync --frozen --extra dev --no-editable`:

```bash
# Offline consistency checks; no dataset download or training.
uv run --frozen --no-editable python -m scripts.verify_report reports/reference-v6-1381

# Restore the exact recorded public round, while it remains available.
uv run --frozen --no-editable python -m scripts.download_reference \
  reports/reference-v6-1381 --data data/reference-v6-1381
uv run --frozen --no-editable solstice run \
  --data data/reference-v6-1381 --output runs/reproduction
```

Small floating-point differences across hardware and libraries are possible. A changed provider file fails the reference hash check rather than silently becoming the same experiment. [The evidence manifest](reference-v6-1381/evidence_manifest.json) verifies the portable score files; this is internal consistency, not an independent replication.

To regenerate the plot, install the optional `reports` dependency group, then run `python -m scripts.plot_report reports/reference-v6-1381 --output reports/holdout.svg`. The [research protocol](../docs/RESEARCH.md) lists technique sources and the untested follow-up experiments.
