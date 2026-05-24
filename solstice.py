#!/usr/bin/env python3
"""
SOLSTICE — Numerai Tournament Pipeline
========================================
Open-source pipeline targeting top 1% performance through MMC optimization.

Strategy:
- Benchmark-aware modeling (BMC optimization)
- Multi-target ensemble with era boosting
- Architecture diversity (LGB + XGB + residual model)
- Optimal feature neutralization + benchmark orthogonalization

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
from scipy import stats
from scipy.stats import spearmanr

import config


def log(msg, level=0):
    indent = "  " * level
    print(f"{indent}{msg}")


# ==============================================================================
# DATA
# ==============================================================================


def download_data():
    """Download v5.2 data including benchmark models."""
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
            log(f"Already have {name}", 1)


def load_features():
    """Load feature metadata and return feature list."""
    with open(config.DATA_DIR / "features.json") as f:
        metadata = json.load(f)
    features = metadata["feature_sets"][config.FEATURE_SET]
    log(f"Feature set '{config.FEATURE_SET}': {len(features)} features", 1)
    return features


def load_data(features, split="train"):
    """Load data with benchmark models joined."""
    targets = [config.PRIMARY_TARGET] + config.AUXILIARY_TARGETS
    cols = ["era"] + features + [t for t in targets]

    log(f"Loading {split} data...", 1)
    df = pd.read_parquet(config.DATA_DIR / f"{split}.parquet", columns=cols)

    # Load and join benchmark models (KEY DIFFERENTIATOR)
    bench_path = config.DATA_DIR / f"{split}_benchmark_models.parquet"
    if bench_path.exists():
        bench = pd.read_parquet(bench_path)
        bench_cols = [c for c in bench.columns if c != "era"]
        df = df.join(bench[bench_cols], how="inner")
        log(f"  + {len(bench_cols)} benchmark model columns", 1)
    else:
        bench_cols = []

    log(f"  {split}: {len(df):,} rows, {df.shape[1]} columns", 1)
    return df, bench_cols


# ==============================================================================
# MODELS
# ==============================================================================


def train_lgb(X, y, params, seed=42):
    """Train a single LightGBM model."""
    import lightgbm as lgb

    p = {**params, "random_state": seed}
    model = lgb.LGBMRegressor(**p)
    model.fit(X, y)
    return model


def train_xgb(X, y, params, seed=42):
    """Train a single XGBoost model."""
    import xgboost as xgb

    p = {**params, "random_state": seed}
    model = xgb.XGBRegressor(**p)
    model.fit(X, y)
    return model


def train_era_boosted_lgb(X, y, eras, params, seed=42):
    """
    Era-boosted LightGBM: iteratively retrain on worst-performing eras.

    This reduces temporal variance, improving Sharpe ratio and stability
    across market regimes.
    """
    import lightgbm as lgb

    if not config.ERA_BOOST_ENABLED:
        return train_lgb(X, y, params, seed)

    p = {
        **params,
        "random_state": seed,
        "n_estimators": config.ERA_BOOST_TREES_PER_STEP,
    }

    # Initial training on all data
    model = lgb.LGBMRegressor(**p)
    model.fit(X, y)

    for iteration in range(config.ERA_BOOST_ITERATIONS):
        # Evaluate per-era performance
        preds = model.predict(X)
        era_scores = {}
        for era in eras.unique():
            mask = eras == era
            if mask.sum() < 10:
                continue
            corr, _ = spearmanr(preds[mask], y.values[mask])
            era_scores[era] = corr

        scores = pd.Series(era_scores).sort_values()
        cutoff = scores.quantile(config.ERA_BOOST_PROPORTION)
        worst_eras = scores[scores <= cutoff].index

        # Retrain on worst eras
        worst_mask = eras.isin(worst_eras)
        if worst_mask.sum() < 100:
            break

        new_model = lgb.LGBMRegressor(**{**p, "n_estimators": p["n_estimators"]})
        new_model.fit(
            X[worst_mask], y[worst_mask],
            init_model=model.booster_,
        )
        model = new_model

    return model


def train_residual_model(X, y, benchmark_preds, params, seed=42):
    """
    Residual model: predict (target - benchmark) directly.

    This maximizes BMC by learning exactly what the benchmark model misses.
    The meta-model is dominated by benchmark-like predictions, so predicting
    residuals produces orthogonal signal = high MMC.
    """
    residuals = y - benchmark_preds
    return train_lgb(X, y=residuals, params=params, seed=seed)


# ==============================================================================
# ENSEMBLE
# ==============================================================================


def train_full_ensemble(train_df, features, bench_cols):
    """
    Train the complete model ensemble:
    - 3x Deep LGB on primary target (multi-seed, era-boosted)
    - 6x Deep LGB on auxiliary targets
    - 1x Shallow LGB (architecture diversity)
    - 1x XGBoost (different tree structure)
    - 1x Residual model (predicts target - benchmark)
    """
    all_features = features + bench_cols
    X = train_df[all_features]
    eras = train_df["era"]
    models = []

    # --- Primary target: Deep LGB with era boosting (3 seeds) ---
    log(f"Training deep LGB on {config.PRIMARY_TARGET} (era-boosted, {len(config.SEEDS)} seeds)...", 1)
    y_primary = train_df[config.PRIMARY_TARGET]
    for seed in config.SEEDS:
        log(f"  Seed {seed}...", 2)
        model = train_era_boosted_lgb(X, y_primary, eras, config.DEEP_LGB_PARAMS, seed)
        models.append(("deep_lgb_ender_eraboosted", seed, model, all_features))

    # --- Auxiliary targets: Deep LGB (1 each) ---
    log("Training on auxiliary targets...", 1)
    for target in config.AUXILIARY_TARGETS:
        if target not in train_df.columns:
            log(f"  SKIP {target} (not in data)", 2)
            continue
        y_aux = train_df[target].dropna()
        X_aux = X.loc[y_aux.index]
        log(f"  {target}...", 2)
        model = train_lgb(X_aux, y_aux, config.DEEP_LGB_PARAMS, seed=42)
        models.append(("deep_lgb_aux", target, model, all_features))

    # --- Shallow LGB (architecture diversity) ---
    log("Training shallow LGB (diversity)...", 1)
    model = train_lgb(X, y_primary, config.SHALLOW_LGB_PARAMS, seed=99)
    models.append(("shallow_lgb", 99, model, all_features))

    # --- XGBoost (different tree structure) ---
    log("Training XGBoost...", 1)
    model = train_xgb(X, y_primary, config.XGB_PARAMS, seed=42)
    models.append(("xgboost", 42, model, all_features))

    # --- Residual model (predicts what benchmark misses) ---
    if bench_cols:
        log("Training residual model (target - benchmark)...", 1)
        # Use first benchmark column as proxy for meta-model
        bench_pred = train_df[bench_cols[0]]
        model = train_residual_model(
            X, y_primary, bench_pred, config.DEEP_LGB_PARAMS, seed=42
        )
        models.append(("residual_lgb", "benchmark_residual", model, all_features))

    log(f"Total models trained: {len(models)}", 1)
    return models


# ==============================================================================
# PREDICTION & POST-PROCESSING
# ==============================================================================


def predict_ensemble(models, df, features, bench_cols):
    """Generate ensemble predictions from all models."""
    all_features = features + bench_cols
    available = [f for f in all_features if f in df.columns]

    predictions = []
    for name, identifier, model, model_features in models:
        avail = [f for f in model_features if f in df.columns]
        pred = model.predict(df[avail])
        predictions.append(pred)

    # Equal-weight ensemble (can be improved with EWA/HRP)
    ensemble = np.mean(predictions, axis=0)
    return ensemble


def neutralize(predictions, neutralizers, proportion=0.5):
    """
    Feature neutralization via linear regression.

    Removes predictable feature exposure from predictions, increasing
    the unique/orthogonal signal that drives MMC.
    """
    if proportion == 0 or neutralizers.shape[1] == 0:
        return predictions

    scores = predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions
    X = np.column_stack([neutralizers, np.ones(len(neutralizers))])
    coeffs = np.linalg.lstsq(X, scores, rcond=None)[0]
    correction = X @ coeffs
    neutralized = scores - proportion * correction
    return neutralized.flatten()


def post_process(predictions, df, features, bench_cols):
    """
    Full post-processing pipeline:
    1. Feature neutralization (proportion=0.5)
    2. Benchmark orthogonalization (proportion=0.3)
    3. Rank normalization to [0, 1]
    """
    # Step 1: Feature neutralization
    feature_data = df[[f for f in features if f in df.columns]].values
    neutralized = neutralize(predictions, feature_data, config.NEUTRALIZE_PROPORTION)

    # Step 2: Benchmark orthogonalization (if available)
    if bench_cols and config.BENCHMARK_NEUTRALIZE_PROPORTION > 0:
        avail_bench = [c for c in bench_cols if c in df.columns]
        if avail_bench:
            bench_data = df[avail_bench].values
            neutralized = neutralize(
                neutralized, bench_data, config.BENCHMARK_NEUTRALIZE_PROPORTION
            )

    # Step 3: Rank normalize to [0, 1]
    final = pd.Series(neutralized, index=df.index).rank(pct=True)
    return final


# ==============================================================================
# EVALUATION
# ==============================================================================


def evaluate(predictions, targets, eras):
    """Per-era evaluation metrics."""
    era_corrs = []
    for era in eras.unique():
        mask = eras == era
        if mask.sum() < 10:
            continue
        corr, _ = spearmanr(predictions[mask], targets[mask])
        era_corrs.append(corr)

    era_corrs = np.array(era_corrs)
    mean_corr = np.mean(era_corrs)
    std_corr = np.std(era_corrs)
    sharpe = mean_corr / std_corr if std_corr > 0 else 0
    max_dd = np.min(np.minimum.accumulate(era_corrs) - era_corrs)

    return {
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_eras": len(era_corrs),
        "pct_positive": np.mean(era_corrs > 0),
    }


# ==============================================================================
# SUBMISSION
# ==============================================================================


def submit_predictions(predictions):
    """Submit predictions to Numerai for the solstice model."""
    import numerapi

    napi = numerapi.NumerAPI(
        public_id=config.NUMERAI_PUBLIC_ID,
        secret_key=config.NUMERAI_SECRET_KEY,
    )

    current_round = napi.get_current_round()
    log(f"Submitting to R{current_round}...", 1)

    # Save predictions
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submit_path = config.SUBMISSIONS_DIR / f"solstice_r{current_round}.csv"

    # Check if already submitted
    if submit_path.exists():
        log(f"Already submitted for R{current_round}", 1)
        return None

    pred_df = predictions.to_frame("prediction")
    pred_df.to_csv(submit_path)

    # Upload
    submission_id = napi.upload_predictions(
        str(submit_path), model_id=config.NUMERAI_MODEL_ID
    )
    log(f"Submitted! ID: {submission_id}", 1)
    return submission_id


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Solstice Numerai Pipeline")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--submit", action="store_true", help="Submit predictions")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on validation")
    args = parser.parse_args()

    if not args.train and not args.submit and not args.evaluate:
        parser.print_help()
        return

    print("=" * 60)
    print("SOLSTICE — Numerai Tournament Pipeline")
    print("=" * 60)
    start = time.time()

    # Download data
    log("\n[1] Downloading data...")
    download_data()

    # Load features
    log("\n[2] Loading features...")
    features = load_features()

    model_path = config.MODEL_DIR / "solstice_ensemble.pkl"

    if args.train:
        # Load training data
        log("\n[3] Loading training data...")
        train_df, bench_cols = load_data(features, "train")

        # Train ensemble
        log("\n[4] Training ensemble...")
        models = train_full_ensemble(train_df, features, bench_cols)

        # Save models
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_data = {
            "models": models,
            "features": features,
            "bench_cols": bench_cols,
        }
        pickle.dump(save_data, open(model_path, "wb"))
        log(f"\n[*] Models saved to {model_path}", 0)

        # Free memory
        del train_df

    if args.evaluate:
        # Load saved model
        if not model_path.exists():
            log("ERROR: No saved model. Run with --train first.")
            return
        save_data = pickle.load(open(model_path, "rb"))
        models = save_data["models"]
        features = save_data["features"]
        bench_cols = save_data["bench_cols"]

        # Load validation data
        log("\n[*] Evaluating on validation...")
        val_df, _ = load_data(features, "validation")

        # Predict
        raw_pred = predict_ensemble(models, val_df, features, bench_cols)
        final_pred = post_process(raw_pred, val_df, features, bench_cols)

        # Evaluate
        metrics_raw = evaluate(raw_pred, val_df[config.PRIMARY_TARGET], val_df["era"])
        metrics_final = evaluate(final_pred, val_df[config.PRIMARY_TARGET], val_df["era"])

        log("\nRaw ensemble:", 1)
        for k, v in metrics_raw.items():
            log(f"  {k}: {v:.5f}", 2)
        log("\nAfter neutralization:", 1)
        for k, v in metrics_final.items():
            log(f"  {k}: {v:.5f}", 2)

    if args.submit:
        # Load saved model
        if not model_path.exists():
            log("ERROR: No saved model. Run with --train first.")
            return
        save_data = pickle.load(open(model_path, "rb"))
        models = save_data["models"]
        features = save_data["features"]
        bench_cols = save_data["bench_cols"]

        # Load live data
        log("\n[*] Generating live predictions...")
        live_df, live_bench_cols = load_data(features, "live")

        # Fill missing benchmark columns
        for c in bench_cols:
            if c not in live_df.columns:
                live_df[c] = 0.5

        # Predict and post-process
        raw_pred = predict_ensemble(models, live_df, features, bench_cols)
        final_pred = post_process(raw_pred, live_df, features, bench_cols)

        # Submit
        submit_predictions(final_pred)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"SOLSTICE COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
