"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"


# --- PARAMETERS ---
N_SPLITS = 5  # Deprecated: kept for backwards compatibility, not used in temporal split
RANDOM_STATE = 42
TARGET = constants.COL_TARGET  # Alias for consistency

# --- TEMPORAL SPLIT CONFIG ---
# Ratio of data to use for training (0 < TEMPORAL_SPLIT_RATIO < 1)
# 0.8 means 80% of data points (by timestamp) go to train, 20% to validation
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 50
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"  # Deprecated: kept for backwards compatibility
MODEL_FILENAME = "lgb_model.txt"  # Single model filename for temporal split

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
# Limit GPU memory usage to 50% to prevent overheating and OOM errors
BERT_GPU_MEMORY_FRACTION = 0.75


# --- FEATURES ---
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]

# --- MODEL PARAMETERS ---
LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}

# LightGBM's fit method allows for a list of callbacks, including early stopping.
# To use it, we need to specify parameters for the early stopping callback.
LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [],  # Placeholder for early stopping callback
}
