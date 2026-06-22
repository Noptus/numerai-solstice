# numerai-solstice 🌅

**An open-source Numerai tournament pipeline targeting top 1% MMC performance.**

Built by [Raphael Caillon](https://numer.ai/solstice) — applied ML engineer, Numerai participant since 2021.

## Strategy

Solstice v4 targets **maximum MMC (Meta-Model Contribution)** through a 5-component ensemble with two-stage neutralization:

1. **Benchmark-Aware Era-Boosted LGB** (35%) — uses benchmark predictions as features + iteratively retrains on worst eras for temporal robustness
2. **Multi-Target Blend** (20%) — trains on 6 auxiliary targets and rank-averages for prediction diversity
3. **Residual Model** (20%) — predicts `target - benchmark` directly, orthogonal to the meta-model by construction
4. **CatBoost** (15%) — symmetric/oblivious trees with uncorrelated errors relative to LGB
5. **FT-Transformer** (10%) — attention-based model capturing feature interactions that trees miss

## Why This Works

The Numerai scoring formula (2026):
```
score = 0.75 × corr20 + 2.25 × mmc20
```

MMC is weighted **3× more** than CORR. MMC measures signal that is **orthogonal to the meta-model** while still correlating with the target. Every component in our pipeline is designed to maximize this orthogonality.

## Quick Start

```bash
# Clone and install
git clone https://github.com/Noptus/numerai-solstice.git
cd numerai-solstice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Numerai API keys and model IDs

# Full pipeline: train + submit
python solstice.py --train --submit

# Submit only (uses saved model)
python solstice.py --submit

# Evaluate on validation
python solstice.py --evaluate
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                     │
│  • v5.2 train + validation + live + benchmark_models            │
│  • Medium features (780) + benchmarks (~8) = ~788 columns       │
│  • Float32, column-pruned parquet reads                         │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  COMPONENT MODELS                                               │
│                                                                 │
│  A. Era-Boosted LGB (3 seeds, benchmark-aware)         [35%]   │
│  B. Multi-Target Blend (6 targets × 1 model)           [20%]   │
│  C. Residual Model (2 seeds, features-only)            [20%]   │
│  D. CatBoost (2 seeds, oblivious trees)                [15%]   │
│  E. FT-Transformer (1 model, last 300 eras)            [10%]   │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  ENSEMBLE LAYER                                                 │
│  • Per-component rank normalization                             │
│  • Weighted average (calibrated on validation Sharpe)           │
│  • Gaussianization                                              │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING (two-stage neutralization)                     │
│  Stage 1: Feature neutralization (top-50 exposed, prop=P)       │
│  Stage 2: Benchmark orthogonalization (optional per slot)       │
│  → Re-rank to uniform [0,1]                                    │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  SUBMISSION (up to 5 slots)                                     │
│  Each slot uses different neutralization strength:              │
│  • p=0.25 → high CORR, low MMC                                 │
│  • p=0.50 → balanced                                           │
│  • p=0.75 → high MMC, moderate CORR                            │
│  • p=1.00 → max MMC                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Key Techniques

### 1. Benchmark Models as Features

Numerai provides benchmark model predictions (`v52_lgbm_ender20`). By including these as input features, the model learns to predict **what the benchmark misses** — directly optimizing BMC/MMC.

```python
# Features = raw features + benchmark predictions
all_feature_cols = medium_features + benchmark_columns
model.fit(train[all_feature_cols], train[target])
```

### 2. Era Boosting

Iteratively retrain on the worst-performing eras to smooth the Sharpe ratio across market regimes:

```python
for iteration in range(15):
    era_scores = evaluate_per_era(model, X, y, eras)
    worst_eras = era_scores[era_scores <= era_scores.quantile(0.5)].index
    model = retrain_incrementally(model, X[worst_eras], y[worst_eras])
```

### 3. Residual Modeling

A dedicated model predicts `target - benchmark`, producing signal orthogonal to the meta-model by construction:

```python
residual_target = target - benchmark_predictions
residual_model.fit(X_features_only, residual_target)
# → Orthogonal to meta → high MMC
```

### 4. Two-Stage Neutralization

```python
def two_stage_neutralize(preds, features, benchmarks, feat_prop, bench_prop):
    # Stage 1: Remove exposure to top-50 correlated features
    neutralized = preds - feat_prop * Ridge().fit(top_features, preds).predict(top_features)
    # Stage 2: Orthogonalize against benchmark predictions
    neutralized = neutralized - bench_prop * Ridge().fit(benchmarks, neutralized).predict(benchmarks)
    return rank_normalize(neutralized)
```

### 5. Architecture Diversity

The ensemble combines fundamentally different model families:
- **LightGBM**: Asymmetric leaf-wise trees (fast, accurate)
- **CatBoost**: Symmetric oblivious trees (different bias, uncorrelated errors)
- **FT-Transformer**: Attention-based (captures interactions trees miss)

This maximizes ensemble diversity → lower correlation with the meta-model.

## Configuration

All parameters live in `config.py`:

- **COMPONENT_WEIGHTS** — Prior ensemble weights (calibrated on validation)
- **LGB_DEEP** — 10k trees, depth 8, 0.1 colsample (community-validated)
- **CATBOOST_PARAMS** — 3k iterations, depth 6, oblivious trees
- **FT_TRANSFORMER_CONFIG** — 3 layers, 192 d_token, 8 heads
- **ERA_BOOST_ITERATIONS** — 15 iterations on worst 50% eras
- **SLOTS** — Up to 5 submission slots with differentiated neutralization

## Multi-Slot Strategy

The same base ensemble is submitted to multiple slots with different post-processing:

- **p=0.25**: Minimal neutralization → highest CORR, lowest MMC
- **p=0.50**: Balanced → good CORR + good MMC
- **p=0.75**: Aggressive → high MMC, moderate CORR
- **p=1.00**: Maximum neutralization → highest MMC, lowest CORR

This creates a diversified portfolio across the CORR/MMC frontier.

## Runtime

- ~100–130 min on Apple M-series (16GB RAM)
- ~4GB peak memory (float32, column-pruned, sequential training)
- Models are saved after training; submission-only runs take ~2 min

## Performance

*Live performance tracking: [numer.ai/solstice](https://numer.ai/solstice)*

## Research References

- [Numerai Docs: Scoring](https://docs.numer.ai/numerai-tournament/scoring)
- [Numerai Docs: MMC](https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc)
- [Era Boosted Models](https://forum.numer.ai/t/era-boosted-models/189)
- [Feature Neutralization](https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955)
- [FT-Transformer (arXiv:2106.11959)](https://arxiv.org/abs/2106.11959)
- [V5.2 Faith II Release](https://forum.numer.ai/t/new-target-for-payouts-and-data-v5-2-faith-ii/8209)

## License

MIT
