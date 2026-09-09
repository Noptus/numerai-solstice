# Numerai Solstice

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

## Local setup

```bash
git clone https://github.com/Noptus/numerai-solstice.git
cd numerai-solstice
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python solstice.py --help
```

Training and evaluation download tournament data and can require substantial memory and time. Review the data version, target configuration and machine requirements first.

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

The current training path calibrates ensemble weights on the last 95 validation eras and then reports evaluation on those same eras. Those figures are tuning diagnostics, not an untouched holdout estimate. A credible comparison needs a separate chronological holdout that remains sealed through feature, model and weight selection.

For a reproducible result, record the source commit, data version, exact era splits, configuration, model hashes, per-era metrics and dated live round history. Compare against a simple baseline and include uncertainty and costs. The README deliberately does not quote a rank or hardware-performance estimate without a matching artifact.

No scheduled training or submission is enabled merely by cloning this repository. No paid compute is required by these documentation changes.

## References

- [Numerai scoring documentation](https://docs.numer.ai/numerai-tournament/scoring)
- [Numerai feature-neutralization discussion](https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955)
- [FT-Transformer paper](https://arxiv.org/abs/2106.11959)

## Reuse

This checkout does not currently include a standalone license file. Confirm reuse terms before redistribution. Tournament data and third-party dependencies retain their own terms.
