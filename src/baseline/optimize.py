"""
Optuna hyperparameter optimization for LightGBM model.
Saves best parameters to output/models/best_params.json
"""

import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def objective(trial):
    # --- Suggest hyperparameters ---
    params = {
        "objective": "rmse",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": config.RANDOM_STATE,
        "n_jobs": -1,

        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
    }

    n_estimators = trial.suggest_int("n_estimators", 500, 5000)

    # Load data (same as in train.py)
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    df = pd.read_parquet(processed_path, engine="pyarrow")
    train_set = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Temporal split
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    # Add aggregates (leakage-safe)
    train_split = add_aggregate_features(train_split.copy(), train_split)
    val_split = add_aggregate_features(val_split.copy(), train_split)

    # Handle missing
    train_split = handle_missing_values(train_split, train_split)
    val_split = handle_missing_values(val_split, train_split)

    # Features
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [c for c in train_split.columns if c not in exclude_cols]
    features = [f for f in features if not train_split[f].dtype == "object"]

    X_train, y_train = train_split[features], train_split[config.TARGET]
    X_val, y_val = val_split[features], val_split[config.TARGET]

    # Train with early stopping
    trial_params = params.copy()
    if config.USE_GPU:
        try:
            trial_params.update({
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
                "max_bin": 255,
                "gpu_use_dp": False,
            })
            print(f"Trial {trial.number}: Using GPU")
        except:
            trial_params["device"] = "cpu"
            print(f"Trial {trial.number}: GPU failed, using CPU")
    else:
        trial_params["device"] = "cpu"

    model = lgb.LGBMRegressor(**trial_params, n_estimators=n_estimators)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(False)]
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


def run_optimization(n_trials=100, study_name="lgb_optimization", storage=None):
    sampler = TPESampler(seed=config.RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    print(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params
    best_params = {
        "n_estimators": trial.params.pop("n_estimators", 2000),
        **trial.params
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    params_path = config.MODEL_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest parameters saved to {params_path}")
    return study


if __name__ == "__main__":
    run_optimization(n_trials=5000)