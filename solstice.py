#!/usr/bin/env python3
"""
SOLSTICE v4 — Numerai Tournament Pipeline
==========================================
Open-source 5-component ensemble pipeline targeting top 1% MMC performance.

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────────┐
  │  DATA LAYER                                                     │
  │  • v5.2 train + validation + live + benchmark_models            │
  │  • Medium features (780) + benchmarks (~8) = ~788 columns       │
  │  • Float32 everywhere, column-pruned parquet reads              │
  └─────────────────────────────────────────────────────────────────┘
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  COMPONENT MODELS (trained sequentially)                        │
  │                                                                 │
  │  A. Benchmark-Aware Era-Boosted LGB (3 seeds)         [35%]    │
  │     → 10k trees + medium + benchmarks + 15-iter era boost       │
  │     → learns what the meta-model misses                         │
  │                                                                 │
  │  B. Multi-Target Blend LGB (6 targets × 1 model)     [20%]    │
  │     → rank-average across targets for diversity                 │
  │     → different targets ≈ different signal facets               │
  │                                                                 │
  │  C. Residual Model LGB (2 seeds)                      [20%]    │
  │     → target = ender - benchmark_ender                          │
  │     → orthogonal to meta BY CONSTRUCTION                        │
  │                                                                 │
  │  D. CatBoost (2 seeds)                                [15%]    │
  │     → symmetric/oblivious trees = different inductive bias      │
  │     → uncorrelated errors with LGB                              │
  │                                                                 │
  │  E. FT-Transformer (1 model, last 300 eras)           [10%]    │
  │     → attention-based, orthogonal to ALL tree models            │
  │     → captures feature interactions trees miss                  │
  └─────────────────────────────────────────────────────────────────┘
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ENSEMBLE LAYER                                                 │
  │  • Per-component rank normalization [0,1]                       │
  │  • Weighted average (weights calibrated from validation sharpe) │
  │  • Gaussianization of final blend                               │
  └─────────────────────────────────────────────────────────────────┘
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  POST-PROCESSING                                                │
  │  • Feature neutralization (top-50 exposed features, prop=P)     │
  │  • Benchmark orthogonalization (optional per slot)              │
  │  • Re-gaussianization → uniform rank                            │
  └─────────────────────────────────────────────────────────────────┘
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  SUBMISSION (up to 5 slots, differentiated neutralization)      │
  │  Each slot applies different neutralization strength:           │
  │  • p=0.25 → high CORR, low MMC                                 │
  │  • p=0.50 → balanced                                           │
  │  • p=0.75 → high MMC                                           │
  │  • p=1.00 → max MMC, sacrifice CORR                            │
  └─────────────────────────────────────────────────────────────────┘

RUNTIME: ~100-130 min on M-series Mac (sequential training)
MEMORY:  ~4GB peak (float32, column-pruned, sequential models)

Usage:
    python solstice.py --train --submit    # Full pipeline
    python solstice.py --submit            # Submit with saved model
    python solstice.py --train             # Train only (no submit)
    python solstice.py --evaluate          # Evaluate saved model on validation
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, norm
from sklearn.linear_model import Ridge

import config


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════


def log(msg, level=0):
    """Timestamped logging with indentation."""
    indent = "  " * level
    print(f"[{time.strftime('%H:%M:%S')}] {indent}{msg}", flush=True)


def gaussianize(arr):
    """Rank → uniform → inverse normal CDF (Numerai standard)."""
    ranked = rankdata(arr, method="average") / (len(arr) + 1)
    return norm.ppf(ranked)


def per_era_metrics(preds, targets, eras):
    """Compute per-era Spearman correlations and derived metrics."""
    era_corrs = []
    for era in sorted(set(eras)):
        mask = eras == era
        if mask.sum() < 50:
            continue
        c, _ = spearmanr(preds[mask], targets[mask])
        if not np.isnan(c):
            era_corrs.append(c)
    era_corrs = np.array(era_corrs)
    return {
        "mean_corr": era_corrs.mean(),
        "std_corr": era_corrs.std(),
        "sharpe": era_corrs.mean() / (era_corrs.std() + 1e-8),
        "consistency": (era_corrs > 0).mean(),
        "max_dd": np.min(np.cumsum(era_corrs - era_corrs.mean())),
        "n_eras": len(era_corrs),
    }


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════


def download_data():
    """Download v5.2 datasets including benchmark models."""
    import numerapi

    napi = numerapi.NumerAPI()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "train": f"{config.DATA_VERSION}/train.parquet",
        "validation": f"{config.DATA_VERSION}/validation.parquet",
        "live": f"{config.DATA_VERSION}/live.parquet",
        "features": f"{config.DATA_VERSION}/features.json",
        "train_bench": f"{config.DATA_VERSION}/train_benchmark_models.parquet",
        "val_bench": f"{config.DATA_VERSION}/validation_benchmark_models.parquet",
        "live_bench": f"{config.DATA_VERSION}/live_benchmark_models.parquet",
    }

    for name, remote_path in files.items():
        local_path = config.DATA_DIR / Path(remote_path).name
        if not local_path.exists():
            log(f"Downloading {remote_path}...", 1)
            napi.download_dataset(remote_path, str(local_path))
        else:
            # Re-download live data if stale (>18h old)
            if "live" in name:
                age_h = (time.time() - local_path.stat().st_mtime) / 3600
                if age_h > 18:
                    log(f"Re-downloading stale {name} ({age_h:.0f}h old)...", 1)
                    napi.download_dataset(remote_path, str(local_path))
                    continue
            log(f"Already have {name}", 1)


def load_features():
    """Load feature metadata and return the configured feature list."""
    with open(config.DATA_DIR / "features.json") as f:
        metadata = json.load(f)
    features = metadata["feature_sets"][config.FEATURE_SET]
    log(f"Feature set '{config.FEATURE_SET}': {len(features)} features", 1)
    return features


def load_data(features, split="train"):
    """
    Load data with benchmark models joined.

    Returns:
        df: DataFrame with features + benchmarks + targets + era
        bench_cols: list of benchmark column names
    """
    targets = [config.PRIMARY_TARGET] + config.AUXILIARY_TARGETS
    cols = ["era"] + features + targets

    log(f"Loading {split} data (column-pruned)...", 1)
    if split == "live":
        cols = ["era"] + features  # live has no targets
    df = pd.read_parquet(config.DATA_DIR / f"{split}.parquet", columns=cols)
    df = df.astype({f: np.float32 for f in features if f in df.columns})

    # Join benchmark model predictions
    bench_path = config.DATA_DIR / f"{split}_benchmark_models.parquet"
    if bench_path.exists():
        bench = pd.read_parquet(bench_path)
        bench_cols = [c for c in bench.columns if c != "era"]
        df = df.join(bench[bench_cols], how="inner")
        log(f"  + {len(bench_cols)} benchmark columns → {df.shape[1]} total", 1)
    else:
        bench_cols = []
        log(f"  (no benchmark data for {split})", 1)

    log(f"  {split}: {len(df):,} rows × {df.shape[1]} cols", 1)
    return df, bench_cols


# ═══════════════════════════════════════════════════════════════════
# COMPONENT A: BENCHMARK-AWARE ERA-BOOSTED LGB
# ═══════════════════════════════════════════════════════════════════


def train_benchmark_era_boosted(train, all_feature_cols):
    """
    Core component: LGB trained on medium + benchmarks with era boosting.

    Era boosting iteratively retrains on the worst 50% of eras,
    smoothing the Sharpe ratio across market regimes.

    Args:
        train: Training DataFrame
        all_feature_cols: Feature columns (medium + benchmark)

    Returns:
        List of trained LGB models (one per seed)
    """
    import lightgbm as lgb

    X = train[all_feature_cols].values
    y = train[config.PRIMARY_TARGET].values
    eras = train["era"].values

    models = []
    for seed in config.SEEDS_ERA_BOOST:
        log(f"Seed {seed}: initial fit + {config.ERA_BOOST_ITERATIONS} era-boost iterations...", 2)
        t0 = time.time()

        # Initial fit
        params = {**config.LGB_DEEP, "random_state": seed, "n_estimators": config.ERA_BOOST_TREES_PER_STEP * 3}
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)

        # Era boosting iterations
        for iteration in range(config.ERA_BOOST_ITERATIONS):
            preds = model.predict(X)
            era_scores = {}
            for era in np.unique(eras):
                mask = eras == era
                if mask.sum() < 50:
                    continue
                c, _ = spearmanr(preds[mask], y[mask])
                era_scores[era] = c if not np.isnan(c) else 0

            # Select worst eras
            scores_series = pd.Series(era_scores)
            threshold = scores_series.quantile(config.ERA_BOOST_PROPORTION)
            worst_eras = scores_series[scores_series <= threshold].index
            boost_mask = np.isin(eras, worst_eras)

            if boost_mask.sum() < 1000:
                break

            # Retrain incrementally on worst eras
            boost_params = {
                **config.LGB_DEEP,
                "random_state": seed + iteration,
                "n_estimators": config.ERA_BOOST_TREES_PER_STEP,
            }
            booster = lgb.LGBMRegressor(**boost_params)
            booster.fit(X[boost_mask], y[boost_mask], init_model=model.booster_)
            model = booster

        models.append(model)
        log(f"Done ({time.time()-t0:.0f}s)", 3)

    return models


# ═══════════════════════════════════════════════════════════════════
# COMPONENT B: MULTI-TARGET BLEND
# ═══════════════════════════════════════════════════════════════════


def train_multi_target(train, all_feature_cols):
    """
    Train one model per auxiliary target.

    Different targets capture different signal facets. Rank-averaging
    predictions across targets adds diversity to the ensemble.

    Returns:
        Dict mapping target name → trained model
    """
    import lightgbm as lgb

    X = train[all_feature_cols].values
    models = {}

    for target in config.AUXILIARY_TARGETS:
        if target not in train.columns:
            log(f"SKIP {target} (not in data)", 2)
            continue
        log(f"Target: {target}...", 2)
        t0 = time.time()
        y = train[target].values
        model = lgb.LGBMRegressor(**{**config.LGB_MULTITARGET, "random_state": 42})
        model.fit(X, y)
        models[target] = model
        log(f"Done ({time.time()-t0:.0f}s)", 3)

    return models


# ═══════════════════════════════════════════════════════════════════
# COMPONENT C: RESIDUAL MODEL
# ═══════════════════════════════════════════════════════════════════


def train_residual(train, features):
    """
    Predict (target - benchmark) = pure orthogonal signal.

    Uses ONLY raw features (no benchmarks as input) to avoid leakage.
    The residual is orthogonal to the meta-model BY CONSTRUCTION,
    making it a direct contributor to MMC.

    Returns:
        List of trained LGB models (one per seed)
    """
    import lightgbm as lgb

    X = train[features].values
    residual_target = train[config.PRIMARY_TARGET].values - train[config.BENCHMARK_PRIMARY].values

    models = []
    for seed in config.SEEDS_RESIDUAL:
        log(f"Residual seed {seed}...", 2)
        t0 = time.time()
        model = lgb.LGBMRegressor(**{**config.LGB_RESIDUAL, "random_state": seed})
        model.fit(X, residual_target)
        models.append(model)
        log(f"Done ({time.time()-t0:.0f}s)", 3)

    return models


# ═══════════════════════════════════════════════════════════════════
# COMPONENT D: CATBOOST
# ═══════════════════════════════════════════════════════════════════


def train_catboost(train, all_feature_cols):
    """
    Symmetric/oblivious trees with fundamentally different inductive bias.

    CatBoost uses balanced tree splits, producing predictions with
    uncorrelated errors relative to LGB's asymmetric trees.

    Returns:
        List of trained CatBoost models (one per seed)
    """
    from catboost import CatBoostRegressor

    X = train[all_feature_cols].values
    y = train[config.PRIMARY_TARGET].values

    models = []
    for seed in config.SEEDS_CATBOOST:
        log(f"CatBoost seed {seed}...", 2)
        t0 = time.time()
        model = CatBoostRegressor(**{**config.CATBOOST_PARAMS, "random_seed": seed})
        model.fit(X, y)
        models.append(model)
        log(f"Done ({time.time()-t0:.0f}s)", 3)

    return models


# ═══════════════════════════════════════════════════════════════════
# COMPONENT E: FT-TRANSFORMER
# ═══════════════════════════════════════════════════════════════════


def train_ft_transformer(train, val, all_feature_cols):
    """
    Attention-based tabular model, orthogonal to ALL tree-based models.

    Since the meta-model is dominated by GBDTs, a transformer contributes
    unique signal that nobody else has → high MMC contribution.

    Trained on last 300 eras with early stopping on validation subset.

    Returns:
        Tuple of (model, feature_means, feature_stds)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from rtdl_revisiting_models import FTTransformer

    # Auto-detect best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log(f"Device: {device}", 2)

    cfg = config.FT_TRANSFORMER_CONFIG

    # Subsample to last N eras (memory + recency)
    all_eras = sorted(train["era"].unique())
    recent_eras = all_eras[-config.FT_TRAIN_ERAS:]
    train_sub = train[train["era"].isin(recent_eras)]

    X_train = train_sub[all_feature_cols].values.astype(np.float32)
    y_train = train_sub[config.PRIMARY_TARGET].values.astype(np.float32)

    # Validation subset for early stopping
    val_eras = sorted(val["era"].unique())[-config.FT_VAL_ERAS:]
    val_sub = val[val["era"].isin(val_eras)]
    X_val = val_sub[all_feature_cols].values.astype(np.float32)
    y_val = val_sub[config.PRIMARY_TARGET].values.astype(np.float32)

    # Standardize using training stats
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0) + 1e-8
    X_train = (X_train - means) / stds
    X_val_std = (X_val - means) / stds

    n_features = len(all_feature_cols)
    model = FTTransformer(
        n_cont_features=n_features,
        cat_cardinalities=[],
        d_out=1,
        n_blocks=cfg["n_layers"],
        d_block=cfg["d_token"],
        attention_n_heads=cfg["n_heads"],
        attention_dropout=cfg["attention_dropout"],
        ffn_d_hidden=int(cfg["d_token"] * cfg["ffn_factor"]),
        ffn_d_hidden_multiplier=None,
        ffn_dropout=cfg["ffn_dropout"],
        residual_dropout=cfg["residual_dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    t0 = time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b, None)
            loss = nn.MSELoss()(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation check every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_val_std, dtype=torch.float32).to(device)
                val_pred = model(val_tensor, None).cpu().numpy().flatten()
                val_loss = np.mean((val_pred - y_val) ** 2)
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg["patience"] // 5:
                log(f"Early stopping at epoch {epoch+1}", 3)
                break

        if (epoch + 1) % 10 == 0:
            log(f"Epoch {epoch+1}/{cfg['epochs']}, loss={epoch_loss/len(loader):.6f}", 3)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    log(f"Done ({time.time()-t0:.0f}s)", 3)
    return model, means, stds


# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE + POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════


def predict_all_components(models_dict, data_df, features, bench_cols, all_feature_cols):
    """
    Generate predictions from all 5 components.

    Returns:
        Dict mapping component name → prediction array
    """
    import torch

    X_full = data_df[all_feature_cols].values
    X_feat = data_df[features].values
    n = len(data_df)

    predictions = {}

    # A. Benchmark-aware era-boosted
    preds = np.zeros(n)
    for m in models_dict["benchmark_era_boost"]:
        preds += m.predict(X_full)
    preds /= len(models_dict["benchmark_era_boost"])
    predictions["benchmark_era_boost"] = preds

    # B. Multi-target blend (rank-average across targets)
    target_preds = []
    for target, m in models_dict["multi_target"].items():
        p = m.predict(X_full)
        target_preds.append(rankdata(p) / n)
    predictions["multi_target"] = np.mean(target_preds, axis=0)

    # C. Residual
    preds = np.zeros(n)
    for m in models_dict["residual"]:
        preds += m.predict(X_feat)
    preds /= len(models_dict["residual"])
    predictions["residual"] = preds

    # D. CatBoost
    preds = np.zeros(n)
    for m in models_dict["catboost"]:
        preds += m.predict(X_full)
    preds /= len(models_dict["catboost"])
    predictions["catboost"] = preds

    # E. FT-Transformer
    ft_model, means, stds = models_dict["ft_transformer"]
    X_std = (X_full.astype(np.float32) - means) / stds
    device = next(ft_model.parameters()).device
    ft_model.eval()
    ft_preds = []
    with torch.no_grad():
        for i in range(0, n, 4096):
            batch = torch.tensor(X_std[i:i+4096], dtype=torch.float32).to(device)
            pred = ft_model(batch, None)
            ft_preds.append(pred.cpu().numpy())
    predictions["ft_transformer"] = np.concatenate(ft_preds).flatten()

    return predictions


def weighted_rank_ensemble(predictions, weights):
    """
    Rank-normalize each component, compute weighted average, then gaussianize.

    This ensures each component contributes on the same scale regardless
    of raw prediction magnitude.
    """
    n = len(next(iter(predictions.values())))
    ensemble = np.zeros(n)
    for name, weight in weights.items():
        ranked = rankdata(predictions[name]) / (n + 1)  # Avoid exact 0 and 1
        ensemble += weight * ranked
    return gaussianize(ensemble)


def calibrate_weights(predictions, targets, eras, base_weights):
    """
    Calibrate ensemble weights on validation data.

    Optimizes for per-era Sharpe of the blend using a simple grid search
    around the prior weights.

    Returns:
        Dict of calibrated weights
    """
    log("Calibrating ensemble weights on validation...", 1)
    n = len(targets)

    # Rank-normalize all components
    ranked_preds = {}
    for name, preds in predictions.items():
        ranked_preds[name] = rankdata(preds) / (n + 1)

    # Grid search: test base weights and variants with each component boosted
    best_sharpe = -999
    best_weights = base_weights.copy()
    names = list(base_weights.keys())

    weight_variants = [base_weights]
    for boost_name in names:
        variant = base_weights.copy()
        variant[boost_name] *= 1.3
        total = sum(variant.values())
        variant = {k: v / total for k, v in variant.items()}
        weight_variants.append(variant)

    for weights in weight_variants:
        blend = np.zeros(n)
        for name, w in weights.items():
            blend += w * ranked_preds[name]
        metrics = per_era_metrics(blend, targets, eras)
        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_weights = weights

    log(f"Best Sharpe: {best_sharpe:.3f}", 2)
    log(f"Weights: {best_weights}", 2)
    return best_weights


def two_stage_neutralize(preds, features_df, bench_df, feat_prop, bench_prop):
    """
    Two-stage neutralization for MMC optimization.

    Stage 1: Remove exposure to top-50 most correlated features.
             This reduces crowded factor exposure.

    Stage 2: Orthogonalize against benchmark predictions.
             This reduces overlap with the meta-model.

    Args:
        preds: Raw ensemble predictions
        features_df: DataFrame of feature values
        bench_df: DataFrame of benchmark predictions
        feat_prop: Feature neutralization proportion (0 to 1)
        bench_prop: Benchmark orthogonalization proportion (0 to 1)

    Returns:
        Neutralized predictions, uniformly distributed in [0, 1]
    """
    ranked = rankdata(preds) / (len(preds) + 1)

    # Stage 1: Feature neutralization (top-50 most exposed features)
    if feat_prop > 0:
        n_neut = min(50, features_df.shape[1])
        exposures = np.abs(np.array([
            spearmanr(ranked, features_df.iloc[:, i].values)[0]
            for i in range(features_df.shape[1])
        ]))
        top_idx = np.argsort(exposures)[-n_neut:]
        neutralizers = features_df.iloc[:, top_idx].values.astype(np.float32)

        lr = Ridge(alpha=1.0).fit(neutralizers, ranked)
        correction = lr.predict(neutralizers)
        ranked = ranked - feat_prop * correction

    # Stage 2: Benchmark orthogonalization
    if bench_prop > 0 and bench_df is not None and bench_df.shape[1] > 0:
        bench_vals = bench_df.values.astype(np.float32)
        lr2 = Ridge(alpha=1.0).fit(bench_vals, ranked)
        correction2 = lr2.predict(bench_vals)
        ranked = ranked - bench_prop * correction2

    # Final uniform rank
    return rankdata(ranked) / (len(ranked) + 1)


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════


def evaluate_pipeline(predictions, val, features, bench_cols, all_feature_cols, weights):
    """Full evaluation: raw ensemble + all neutralization variants."""
    targets = val[config.PRIMARY_TARGET].values
    eras = val["era"].values
    unique_eras = sorted(val["era"].unique())
    eval_eras = unique_eras[-95:]  # Last 95 eras for robust evaluation
    eval_mask = np.isin(eras, eval_eras)

    # Raw ensemble
    ensemble = weighted_rank_ensemble(predictions, weights)
    raw_metrics = per_era_metrics(ensemble[eval_mask], targets[eval_mask], eras[eval_mask])
    log(f"RAW: corr={raw_metrics['mean_corr']:.5f}, sharpe={raw_metrics['sharpe']:.3f}, "
        f"consistency={raw_metrics['consistency']:.1%}", 2)

    # Neutralized variants
    features_df = val[features]
    bench_df = val[bench_cols]
    for prop in [0.25, 0.5, 0.75, 1.0]:
        neut_preds = two_stage_neutralize(
            ensemble[eval_mask],
            features_df.iloc[eval_mask],
            bench_df.iloc[eval_mask],
            prop, 0.3,
        )
        neut_metrics = per_era_metrics(neut_preds, targets[eval_mask], eras[eval_mask])
        log(f"NEUT(p={prop:.2f}): corr={neut_metrics['mean_corr']:.5f}, "
            f"sharpe={neut_metrics['sharpe']:.3f}", 2)

    return raw_metrics


# ═══════════════════════════════════════════════════════════════════
# SUBMISSION
# ═══════════════════════════════════════════════════════════════════


def submit_all_slots(ensemble_live, live, features, bench_cols):
    """
    Submit to all configured slots with differentiated neutralization.

    Each slot applies a different feature neutralization proportion,
    creating a portfolio of models ranging from high-CORR to high-MMC.
    """
    import numerapi

    features_df = live[features]
    bench_df = live[bench_cols]
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    for slot_name, slot_cfg in config.SLOTS.items():
        model_id = slot_cfg["model_id"]
        if not model_id:
            log(f"Skipping {slot_name}: no model_id configured", 2)
            continue

        # Two-stage neutralization with per-slot parameters
        slot_preds = two_stage_neutralize(
            ensemble_live, features_df, bench_df,
            feat_prop=slot_cfg["neut_prop"],
            bench_prop=slot_cfg["bench_orth"],
        )

        # Format and save
        pred_df = pd.DataFrame({"prediction": slot_preds}, index=live.index)
        pred_df.index.name = "id"
        submit_path = config.SUBMISSIONS_DIR / f"predictions_{slot_name}.csv"
        pred_df.to_csv(submit_path)

        # Upload
        try:
            napi = numerapi.NumerAPI(
                public_id=config.NUMERAI_PUBLIC_ID,
                secret_key=config.NUMERAI_SECRET_KEY,
            )
            sub_id = napi.upload_predictions(str(submit_path), model_id=model_id)
            log(f"✓ {slot_name} submitted ({slot_cfg['description']}, id={sub_id})", 2)
        except Exception as e:
            log(f"✗ {slot_name} failed: {e}", 2)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Solstice v4 — 5-component Numerai tournament pipeline"
    )
    parser.add_argument("--train", action="store_true", help="Train all 5 components")
    parser.add_argument("--submit", action="store_true", help="Submit to all configured slots")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on validation data")
    args = parser.parse_args()

    if not args.train and not args.submit and not args.evaluate:
        parser.print_help()
        return

    print("═" * 70)
    print("  SOLSTICE v4 — 5-Component Numerai Ensemble Pipeline")
    print("  Components: ERA-LGB × Multi-Target × Residual × CatBoost × FT-Transformer")
    print("═" * 70)
    start = time.time()

    # ── Data ────────────────────────────────────────────────────────
    log("\n[1/8] Downloading data...")
    download_data()

    log("\n[2/8] Loading features...")
    features = load_features()

    model_path = config.MODEL_DIR / "solstice_v4.pkl"
    ft_model_path = config.MODEL_DIR / "ft_transformer_v4.pt"

    if args.train:
        # Load data
        log("\n[3/8] Loading training data...")
        train_df, bench_cols = load_data(features, "train")
        val_df, _ = load_data(features, "validation")
        all_feature_cols = features + bench_cols

        # ── Train all components ────────────────────────────────────
        log("\n[4/8] Component A: Benchmark-Aware Era-Boosted LGB...")
        bench_models = train_benchmark_era_boosted(train_df, all_feature_cols)

        log("\n[5/8] Component B: Multi-Target Blend...")
        mt_models = train_multi_target(train_df, all_feature_cols)

        log("\n[6/8] Component C: Residual Model...")
        resid_models = train_residual(train_df, features)

        log("\n[7/8] Component D: CatBoost...")
        cb_models = train_catboost(train_df, all_feature_cols)

        log("\n[8/8] Component E: FT-Transformer...")
        ft_result = train_ft_transformer(train_df, val_df, all_feature_cols)

        models_dict = {
            "benchmark_era_boost": bench_models,
            "multi_target": mt_models,
            "residual": resid_models,
            "catboost": cb_models,
            "ft_transformer": ft_result,
        }

        # ── Calibrate + Evaluate ────────────────────────────────────
        log("\nEvaluating on validation...")
        val_predictions = predict_all_components(models_dict, val_df, features, bench_cols, all_feature_cols)

        val_eras = sorted(val_df["era"].unique())
        eval_eras = val_eras[-95:]
        eval_mask = np.isin(val_df["era"].values, eval_eras)
        calibrated_weights = calibrate_weights(
            {k: v[eval_mask] for k, v in val_predictions.items()},
            val_df[config.PRIMARY_TARGET].values[eval_mask],
            val_df["era"].values[eval_mask],
            config.COMPONENT_WEIGHTS,
        )

        log("Full evaluation (last 95 val eras):")
        evaluate_pipeline(val_predictions, val_df, features, bench_cols, all_feature_cols, calibrated_weights)

        # ── Save ────────────────────────────────────────────────────
        log("\nSaving models...")
        import torch
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "benchmark_era_boost": bench_models,
            "multi_target": mt_models,
            "residual": resid_models,
            "catboost": cb_models,
            "ft_stats": (ft_result[1], ft_result[2]),
            "calibrated_weights": calibrated_weights,
            "config": {
                "features": features,
                "bench_cols": bench_cols,
                "all_feature_cols": all_feature_cols,
            },
            "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        }
        with open(model_path, "wb") as f:
            pickle.dump(save_dict, f)
        torch.save(ft_result[0].state_dict(), ft_model_path)
        log(f"Models saved to {config.MODEL_DIR}", 1)

        del train_df  # Free memory

    if args.evaluate and not args.train:
        # Load saved model and evaluate
        if not model_path.exists():
            log("ERROR: No saved model. Run with --train first.")
            return
        log("\nLoading saved model for evaluation...")
        with open(model_path, "rb") as f:
            save_data = pickle.load(f)

        import torch
        from rtdl_revisiting_models import FTTransformer

        features = save_data["config"]["features"]
        bench_cols = save_data["config"]["bench_cols"]
        all_feature_cols = save_data["config"]["all_feature_cols"]

        # Reconstruct FT-Transformer
        n_features = len(all_feature_cols)
        cfg = config.FT_TRANSFORMER_CONFIG
        ft_model = FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=cfg["n_layers"],
            d_block=cfg["d_token"],
            attention_n_heads=cfg["n_heads"],
            attention_dropout=cfg["attention_dropout"],
            ffn_d_hidden=int(cfg["d_token"] * cfg["ffn_factor"]),
            ffn_d_hidden_multiplier=None,
            ffn_dropout=cfg["ffn_dropout"],
            residual_dropout=cfg["residual_dropout"],
        )
        ft_model.load_state_dict(torch.load(ft_model_path, map_location="cpu"))
        if torch.backends.mps.is_available():
            ft_model = ft_model.to("mps")
        elif torch.cuda.is_available():
            ft_model = ft_model.to("cuda")

        models_dict = {
            "benchmark_era_boost": save_data["benchmark_era_boost"],
            "multi_target": save_data["multi_target"],
            "residual": save_data["residual"],
            "catboost": save_data["catboost"],
            "ft_transformer": (ft_model, save_data["ft_stats"][0], save_data["ft_stats"][1]),
        }
        weights = save_data["calibrated_weights"]

        val_df, _ = load_data(features, "validation")
        val_predictions = predict_all_components(models_dict, val_df, features, bench_cols, all_feature_cols)
        evaluate_pipeline(val_predictions, val_df, features, bench_cols, all_feature_cols, weights)

    if args.submit:
        # Load saved model and submit
        if not model_path.exists():
            log("ERROR: No saved model. Run with --train first.")
            return

        if not args.train:
            log("\nLoading saved model for submission...")
            with open(model_path, "rb") as f:
                save_data = pickle.load(f)

            import torch
            from rtdl_revisiting_models import FTTransformer

            features = save_data["config"]["features"]
            bench_cols = save_data["config"]["bench_cols"]
            all_feature_cols = save_data["config"]["all_feature_cols"]

            n_features = len(all_feature_cols)
            cfg = config.FT_TRANSFORMER_CONFIG
            ft_model = FTTransformer(
                n_cont_features=n_features,
                cat_cardinalities=[],
                d_out=1,
                n_blocks=cfg["n_layers"],
                d_block=cfg["d_token"],
                attention_n_heads=cfg["n_heads"],
                attention_dropout=cfg["attention_dropout"],
                ffn_d_hidden=int(cfg["d_token"] * cfg["ffn_factor"]),
                ffn_d_hidden_multiplier=None,
                ffn_dropout=cfg["ffn_dropout"],
                residual_dropout=cfg["residual_dropout"],
            )
            ft_model.load_state_dict(torch.load(ft_model_path, map_location="cpu"))
            if torch.backends.mps.is_available():
                ft_model = ft_model.to("mps")
            elif torch.cuda.is_available():
                ft_model = ft_model.to("cuda")

            models_dict = {
                "benchmark_era_boost": save_data["benchmark_era_boost"],
                "multi_target": save_data["multi_target"],
                "residual": save_data["residual"],
                "catboost": save_data["catboost"],
                "ft_transformer": (ft_model, save_data["ft_stats"][0], save_data["ft_stats"][1]),
            }
            calibrated_weights = save_data["calibrated_weights"]
        else:
            # Already trained above — use in-memory models
            pass

        log("\nGenerating live predictions...")
        live_df, live_bench_cols = load_data(features, "live")

        # Fill any missing benchmark columns
        for c in bench_cols:
            if c not in live_df.columns:
                live_df[c] = 0.5

        live_predictions = predict_all_components(models_dict, live_df, features, bench_cols, all_feature_cols)
        ensemble_live = weighted_rank_ensemble(live_predictions, calibrated_weights)

        log("Submitting to all slots...")
        submit_all_slots(ensemble_live, live_df, features, bench_cols)

    elapsed = time.time() - start
    print(f"\n{'═' * 70}")
    print(f"  SOLSTICE v4 COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
