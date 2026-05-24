"""
Solstice Configuration
======================
All tunable parameters in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === API Credentials ===
NUMERAI_PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID")
NUMERAI_SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY")
NUMERAI_MODEL_ID = os.getenv("NUMERAI_MODEL_ID")

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# === Data ===
DATA_VERSION = "v5.2"
FEATURE_SET = "medium"  # 780 features; benchmark models add ~8 more

# === Targets ===
# Ender20 is the 2026 payout/scoring target
PRIMARY_TARGET = "target_ender_20"
AUXILIARY_TARGETS = [
    "target_cyrusd_20",
    "target_victor_20",
    "target_alpha_20",
    "target_ralph_20",
    "target_teager2b_20",
    "target_xerxes_20",
]

# === Model Hyperparameters ===

# Deep LightGBM (primary workhorse)
DEEP_LGB_PARAMS = {
    "n_estimators": 10000,
    "learning_rate": 0.003,
    "max_depth": 8,
    "num_leaves": 512,
    "colsample_bytree": 0.1,   # Numerai community-validated
    "min_data_in_leaf": 5000,  # Era stability
    "n_jobs": -1,
    "verbosity": -1,
}

# Shallow LightGBM (diversity via architecture)
SHALLOW_LGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 64,
    "colsample_bytree": 0.05,  # Very aggressive feature sampling
    "min_data_in_leaf": 2000,
    "subsample": 0.7,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
}

# XGBoost (different tree structure for diversity)
XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.005,
    "max_depth": 6,
    "colsample_bytree": 0.1,
    "subsample": 0.7,
    "min_child_weight": 5000,
    "tree_method": "hist",
    "verbosity": 0,
}

# === Ensemble ===
SEEDS = [42, 137, 2024]  # Multi-seed for primary target

# === Post-Processing ===
NEUTRALIZE_PROPORTION = 0.5  # Optimal from bias-variance sweep
BENCHMARK_NEUTRALIZE_PROPORTION = 0.3  # Against benchmark predictions

# === Era Boosting ===
ERA_BOOST_ENABLED = True
ERA_BOOST_ITERATIONS = 15
ERA_BOOST_PROPORTION = 0.5  # Fraction of worst eras to retrain
ERA_BOOST_TREES_PER_STEP = 200

# === Validation ===
CV_WINDOW_SIZE = 156  # eras per fold
CV_PURGE_ERAS = 8     # gap between train and val (20D target)
CV_MIN_TRAIN_SIZE = 148
