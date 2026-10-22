# Numerai Solstice

[![Research checks](https://github.com/Noptus/numerai-solstice/actions/workflows/tests.yml/badge.svg)](https://github.com/Noptus/numerai-solstice/actions/workflows/tests.yml)

**AI-assisted research implementation.** Offline tests cover era isolation, cross-sectional transforms and training orchestration; model training and live performance have not been validated by this cleanup.

**Compatibility checked October 22, 2026:** this code targets v5.2. Numerai retired public v5.0-v5.3 training/validation downloads on October 16. Training requires previously cached compatible data. A v6 migration needs explicit column, target, feature and benchmark changes plus retraining; changing the version string is insufficient. See [Numerai data documentation](https://docs.numer.ai/numerai-tournament/data).

A Python research pipeline for the Numerai tournament, combining tree-based models, a transformer, and configurable prediction neutralization.

The public implementation is in [solstice.py](solstice.py), with experiment settings in [config.py](config.py). This repository demonstrates model-pipeline design. It does not establish a top-1% ranking, guaranteed profitability, or an independently validated improvement over the tournament benchmark.

## What is implemented

- Benchmark-aware LightGBM with era-focused retraining.
- A multi-target LightGBM blend and a residual-target model.
- CatBoost and an FT-Transformer component.
- Per-component rank normalization and weighted ensembling.
- Feature and benchmark neutralization, evaluation, checkpointing and optional submission.

These are hypotheses to compare, not guarantees of diversification. Training on a residual target does not by itself make predictions orthogonal to the meta-model. Different model families can have correlated errors, and stronger neutralization does not necessarily improve MMC.

## Start here

| File | Responsibility |
| --- | --- |
| [solstice.py](solstice.py) | Data loading, component training, prediction, calibration and submission |
| [config.py](config.py) | Features, targets, model parameters and ensemble weights |
| [requirements.txt](requirements.txt) | Python dependencies |
| [.env.example](.env.example) | Credential configuration template; use local values, never committed secrets |

## Offline research checks

No data downloads or ML framework install are needed for these checks:

```bash
python -m pip install -r requirements-checks.txt
python -m unittest discover -s tests -v
python -m py_compile solstice.py config.py validation.py
python solstice.py --help
```

The tests verify chronological order, disjoint roles, embargo gaps and failure on insufficient history. Cross-sectional ranks and neutralization are computed within each era; metrics reject nonfinite inputs and empty/constant evaluation groups and undefined mean/std ratios. A synthetic orchestration test runs the real training control flow with mocked models and checks that early stopping, weight calibration and evaluation receive different era labels; another rejects legacy checkpoints. They do not certify the entire model pipeline or establish profitable performance.

## Local setup with cached v5.2 data

```bash
git clone https://github.com/Noptus/numerai-solstice.git
cd numerai-solstice
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python solstice.py --help
```

The CLI checks the available dataset catalog before downloading missing files. Unavailable v5.2 files produce an explicit compatibility error before any partial download. Each command requests only the files it needs. Use the offline checks above for a clean-clone review. Full training also requires substantial memory and time. No training or submission was run during the cleanup.

```bash
# Train without submitting predictions
python solstice.py --train

# Evaluate an existing checkpoint
python solstice.py --evaluate
```

Submission is a separate, externally visible action. Only after reviewing the saved model and configuring your own account:

```bash
python solstice.py --submit
```

## Evaluation boundaries

The training path now separates validation eras into early stopping, a four-era gap, 95 calibration eras, a second four-era gap, and the final 95 evaluation eras. The transformer sees only the early-stopping partition; ensemble weights use only calibration eras. Partition labels, data version and target are saved with the checkpoint. Evaluation rejects legacy checkpoints without a split.

Four eras are a configurable gap for the v5 20-day experiment, not a universal leakage guarantee. Check actual era spacing and overlapping labels before extending it. Existing historical validation data was already used by earlier versions: a new split cannot make it untouched again. New claims require a frozen protocol and genuinely unseen future data. The reported neutralization variants are exploratory diagnostics; selecting one after viewing evaluation metrics consumes that evaluation set.
The displayed correlation is a **Spearman proxy**, not official Numerai CORR or MMC. The `sharpe` field is the mean/std ratio of per-era correlations, not an annualized returns Sharpe. `max_dd` measures drawdown in cumulative correlations, not monetary P&L. Benchmark comparisons use the same proxy.

For a reproducible result, record the source commit, data version, exact era splits, configuration, model hashes, per-era metrics and dated live round history. Compare against a simple baseline and include uncertainty and costs. The README deliberately does not quote a rank or hardware-performance estimate without a matching artifact.

No scheduled training or submission is enabled merely by cloning this repository. No paid compute is required by these documentation changes.

## References

- [Numerai scoring documentation](https://docs.numer.ai/numerai-tournament/scoring)
- [Numerai feature-neutralization discussion](https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955)
- [FT-Transformer paper](https://arxiv.org/abs/2106.11959)

## Reuse

This checkout does not currently include a standalone license file. Confirm reuse terms before redistribution. Tournament data and third-party dependencies retain their own terms.
