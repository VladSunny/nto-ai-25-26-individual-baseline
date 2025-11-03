"""
Main training script for the LightGBM model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    """Runs the model training pipeline with temporal split.

    Loads prepared data from data/processed/, performs temporal split based on
    absolute date threshold, computes aggregate features on train split only,
    and trains a single LightGBM model. This ensures methodologically correct
    validation without data leakage from future timestamps.

    Note: Data must be prepared first using prepare_data.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Check for timestamp column
    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Perform temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )
    print("✅ Temporal split validation passed: all validation timestamps are after train timestamps")

    # Compute aggregate features on train split only (to prevent data leakage)
    print("\nComputing aggregate features on train split only...")
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split)  # Use train_split for aggregates!

    # Handle missing values (use train_split for fill values)
    print("Handling missing values...")
    train_split_final = handle_missing_values(train_split_with_agg, train_split)
    val_split_final = handle_missing_values(val_split_with_agg, train_split)

    # Define features (X) and target (y)
    # Exclude timestamp, source, target, prediction columns
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in train_split_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET]

    print(f"Training features: {len(features)}")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train single model
    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(**config.LGB_PARAMS)

    # Update fit params with early stopping callback
    fit_params = config.LGB_FIT_PARAMS.copy()
    fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=fit_params["eval_metric"],
        callbacks=fit_params["callbacks"],
    )

    # Evaluate the model
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    print(f"\nValidation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save the trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.booster_.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
