# numerai-solstice 🌅

**An open-source Numerai tournament pipeline targeting top 1% performance.**

Built by [Raphael Caillon](https://numer.ai/solstice) — applied ML engineer, Numerai participant since 2021.

## Strategy

Solstice targets **maximum MMC (Meta-Model Contribution)** through:

1. **Benchmark-aware modeling** — uses Numerai's benchmark model predictions as input features, directly optimizing BMC
2. **Multi-target ensemble** — trains on 7 targets (ender, cyrusd, victor, alpha, ralph, teager2b, xerxes) for prediction diversity
3. **Era-boosted training** — iteratively retrains on worst-performing eras for temporal robustness
4. **Architecture diversity** — LightGBM (deep + shallow) + XGBoost + FT-Transformer for orthogonal signal
5. **Optimal neutralization** — feature neutralization at p=0.5, plus benchmark-aware orthogonalization
6. **Residual modeling** — dedicated model predicting target residuals after benchmark, maximizing BMC directly

## Why This Works

The Numerai scoring formula (2026):
```
score = 0.75 * corr20 + 2.25 * mmc20
```

MMC is weighted **3x more** than CORR. MMC measures signal that is **orthogonal to the meta-model** while still correlating with the target. Our pipeline is designed end-to-end for this objective.

## Quick Start

```bash
# Clone and install
git clone https://github.com/Noptus/numerai-solstice.git
cd numerai-solstice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Numerai API keys

# Train and submit
python solstice.py --train --submit

# Submit only (uses saved model)
python solstice.py --submit
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SOLSTICE PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data Layer (v5.2)                                          │
│  ├── Features: medium set (780) + benchmark models (8)      │
│  ├── Targets: ender_20 (primary) + 6 auxiliary              │
│  └── Walk-forward CV: 156-era windows, 8-era purge          │
│                                                             │
│  Model Layer (10+ diverse models)                           │
│  ├── 3x Deep LightGBM on ender_20 (multi-seed)             │
│  ├── 6x Deep LightGBM on auxiliary targets                  │
│  ├── 1x Shallow LightGBM (architecture diversity)           │
│  ├── 1x XGBoost (different tree structure)                  │
│  ├── 1x FT-Transformer (neural net orthogonality)           │
│  └── 1x Residual model (predicts target - benchmark)        │
│                                                             │
│  Ensemble Layer                                             │
│  ├── Era-weighted aggregation (EWA)                         │
│  ├── HRP (Hierarchical Risk Parity) weighting              │
│  └── Ridge meta-learner on validation                       │
│                                                             │
│  Post-Processing                                            │
│  ├── Feature neutralization (p=0.5)                         │
│  ├── Benchmark orthogonalization                            │
│  ├── Gaussianization + rank normalization                   │
│  └── Era-wise standardization                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Techniques

### 1. Benchmark Models as Features

The single biggest edge: Numerai provides predictions from their benchmark model (`v52_lgbm_ender20`). By using these as input features, our model learns to predict **what the benchmark misses** — directly optimizing BMC.

```python
# Join benchmark predictions to training data
train = train.join(benchmark_predictions, how="inner")
feature_cols = regular_features + benchmark_columns
model.fit(train[feature_cols], train[target])
```

### 2. Era Boosting

Iteratively focus training on the hardest time periods:

```python
for iteration in range(n_iterations):
    era_scores = evaluate_per_era(model, X, y, eras)
    worst_eras = era_scores.nsmallest(int(len(era_scores) * 0.5)).index
    model = retrain_on_worst(model, X[eras.isin(worst_eras)], ...)
```

### 3. Residual Modeling for MMC

Train a dedicated model to predict what the benchmark gets wrong:

```python
residuals = target - benchmark_predictions
residual_model.fit(X, residuals)
# Final: benchmark + residual_model(x) → high CORR + high MMC
```

### 4. Feature Neutralization

Remove predictable feature exposure to maximize unique signal:

```python
def neutralize(predictions, features, proportion=0.5):
    exposures = np.column_stack([features, np.ones(len(features))])
    correction = exposures @ np.linalg.lstsq(exposures, predictions)[0]
    return predictions - proportion * correction
```

### 5. FT-Transformer for Diversity

Neural networks produce fundamentally different predictions than tree models, increasing ensemble orthogonality to the meta-model:

```python
model = FTTransformer(
    d_numerical=788,  # features + benchmarks
    n_layers=3, d_token=192, n_heads=8,
    attention_dropout=0.2, ffn_dropout=0.2,
    activation="reglu", d_out=1,
)
```

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `NEUTRALIZE_PROP` | 0.5 | Optimal Sharpe from bias-variance sweep |
| `N_ESTIMATORS` | 10000 | Practical for laptop, competitive |
| `MAX_DEPTH` | 8 | Deep enough for non-linear interactions |
| `COLSAMPLE_BYTREE` | 0.1 | Numerai community-validated |
| `MIN_DATA_IN_LEAF` | 5000 | Era stability |
| `ERA_BOOST_PROPORTION` | 0.5 | Fraction of worst eras to retrain |
| `FEATURE_SET` | medium | 780 features, benchmark models add 8 more |

## Performance

*Live performance tracking: [numer.ai/solstice](https://numer.ai/solstice)*

Pipeline validated on walk-forward CV with 8-era purge gap (20D targets).

## Automated Submission

The pipeline includes a launchd-compatible auto-submit script:

```bash
# Install the cron (macOS)
cp com.numerai.solstice.autosubmit.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.numerai.solstice.autosubmit.plist
```

Submissions fire at 14:35, 15:35, 16:35 CEST daily (redundant attempts for reliability).

## Research References

- [Numerai Docs: Scoring](https://docs.numer.ai/numerai-tournament/scoring)
- [Numerai Docs: MMC](https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc)
- [Era Boosted Models](https://forum.numer.ai/t/era-boosted-models/189)
- [Feature Neutralization](https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955)
- [Signal Miner + GFlowNets](https://forum.numer.ai/t/gflownets-for-signal-miner-a-new-way-to-find-diverse-high-performing-models/7966)
- [AI for ML (jefferythewind)](https://forum.numer.ai/t/ai-for-ml-by-jefferythewind/8245)
- [V5.2 Faith II Release](https://forum.numer.ai/t/new-target-for-payouts-and-data-v5-2-faith-ii/8209)
- [FT-Transformer (arXiv:2106.11959)](https://arxiv.org/abs/2106.11959)

## License

MIT
