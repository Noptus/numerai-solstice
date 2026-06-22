"""
Solstice v4 Configuration
==========================
All tunable parameters in one place.

Architecture: 5-component ensemble with two-stage neutralization.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# API Credentials (from .env)
# ═══════════════════════════════════════════════════════════════════
NUMERAI_PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID")
NUMERAI_SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY")

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════
DATA_VERSION = "v5.2"
FEATURE_SET = "medium"  # 780 features; benchmark models add ~8 more

# ═══════════════════════════════════════════════════════════════════
# Targets
# ═══════════════════════════════════════════════════════════════════
PRIMARY_TARGET = "target_ender_20"
BENCHMARK_PRIMARY = "v52_lgbm_ender20"

AUXILIARY_TARGETS = [
    "target_cyrusd_20",
    "target_teager2b_20",
    "target_victor_20",
    "target_alpha_20",
    "target_ralph_20",
    "target_xerxes_20",
]

# ═══════════════════════════════════════════════════════════════════
# Component Ensemble Weights (priors; calibrated on validation)
# ═══════════════════════════════════════════════════════════════════
COMPONENT_WEIGHTS = {
    "benchmark_era_boost": 0.35,
    "multi_target": 0.20,
    "residual": 0.20,
    "catboost": 0.15,
    "ft_transformer": 0.10,
}

# ═══════════════════════════════════════════════════════════════════
# Submission Slots
#
# Configure up to 5 slots with differentiated neutralization.
# Each slot gets the same base ensemble but different post-processing.
#
# Set model IDs via environment variables (SLOT_1_MODEL_ID, etc.)
# or override in .env. Slots with missing model_id are skipped.
# ═══════════════════════════════════════════════════════════════════
SLOTS = {
    "slot_1": {
        "model_id": os.getenv("SLOT_1_MODEL_ID"),
        "neut_prop": 0.50,
        "bench_orth": 0.30,
        "description": "Balanced MMC/CORR",
    },
    "slot_2": {
        "model_id": os.getenv("SLOT_2_MODEL_ID"),
        "neut_prop": 1.00,
        "bench_orth": 0.50,
        "description": "Max MMC (sacrifice CORR)",
    },
    "slot_3": {
        "model_id": os.getenv("SLOT_3_MODEL_ID"),
        "neut_prop": 0.75,
        "bench_orth": 0.30,
        "description": "High MMC, moderate CORR",
    },
    "slot_4": {
        "model_id": os.getenv("SLOT_4_MODEL_ID"),
        "neut_prop": 0.25,
        "bench_orth": 0.00,
        "description": "High CORR, low neutralization",
    },
    "slot_5": {
        "model_id": os.getenv("SLOT_5_MODEL_ID"),
        "neut_prop": 0.50,
        "bench_orth": 0.30,
        "description": "Balanced (duplicate for redundancy)",
    },
}

# ═══════════════════════════════════════════════════════════════════
# Model Hyperparameters
# ═══════════════════════════════════════════════════════════════════

# Component A: Deep LightGBM (era-boosted, benchmark-aware)
LGB_DEEP = {
    "n_estimators": 10000,
    "learning_rate": 0.003,
    "max_depth": 8,
    "num_leaves": 512,
    "colsample_bytree": 0.1,
    "min_child_samples": 5000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}

# Component B: Multi-target LGB
LGB_MULTITARGET = {
    "n_estimators": 5000,
    "learning_rate": 0.005,
    "max_depth": 7,
    "num_leaves": 128,
    "colsample_bytree": 0.1,
    "min_child_samples": 5000,
    "n_jobs": -1,
    "verbosity": -1,
}

# Component C: Residual model LGB
LGB_RESIDUAL = {
    "n_estimators": 5000,
    "learning_rate": 0.005,
    "max_depth": 6,
    "num_leaves": 64,
    "colsample_bytree": 0.15,
    "min_child_samples": 5000,
    "n_jobs": -1,
    "verbosity": -1,
}

# Component D: CatBoost
CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "verbose": 0,
    "thread_count": -1,
}

# Component E: FT-Transformer
FT_TRANSFORMER_CONFIG = {
    "d_token": 192,
    "n_layers": 3,
    "n_heads": 8,
    "ffn_factor": 4.0 / 3.0,
    "attention_dropout": 0.2,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 2048,
    "epochs": 50,
    "patience": 10,
}

# ═══════════════════════════════════════════════════════════════════
# Era Boosting
# ═══════════════════════════════════════════════════════════════════
ERA_BOOST_ITERATIONS = 15
ERA_BOOST_PROPORTION = 0.5   # Fraction of worst eras to retrain on
ERA_BOOST_TREES_PER_STEP = 300

# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════
SEEDS_ERA_BOOST = [42, 123, 456]   # 3 seeds for component A
SEEDS_RESIDUAL = [42, 123]          # 2 seeds for component C
SEEDS_CATBOOST = [42, 123]          # 2 seeds for component D
FT_TRAIN_ERAS = 300                 # Use last N eras for FT-Transformer
FT_VAL_ERAS = 50                    # Validation subset for early stopping
