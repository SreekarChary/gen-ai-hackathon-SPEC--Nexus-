"""
ClaimWatch AI — Global Configuration
"""
import os

# ─── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Model Hyperparameters ──────────────────────────────
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "class_weight": "balanced",
}

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

# ─── Train / Test Split ────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── SHAP / Explainability ─────────────────────────────
TOP_N_FEATURES = 5  # number of top features to highlight in reasoning
