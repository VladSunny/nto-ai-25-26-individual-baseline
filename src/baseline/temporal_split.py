"""
Temporal splitting utilities for time-series data.

This module provides functions for splitting data based on absolute date thresholds,
ensuring that validation data contains only records with timestamps after the
threshold date, which prevents data leakage from future data.
"""

import pandas as pd

from . import constants


def temporal_split_by_date(
    df: pd.DataFrame, split_date: pd.Timestamp, timestamp_col: str = constants.COL_TIMESTAMP
) -> tuple[pd.Series, pd.Series]:
    """Splits DataFrame into train and validation sets based on absolute date threshold.

    All records with timestamp <= split_date go to train, records with timestamp > split_date
    go to validation. This ensures temporal correctness: validation only contains future data.

    Args:
        df: DataFrame with timestamp column to split.
        split_date: Timestamp threshold. Records with timestamp <= split_date go to train.
        timestamp_col: Name of the timestamp column. Defaults to COL_TIMESTAMP.

    Returns:
        Tuple of (train_mask, val_mask) boolean Series where True indicates
        inclusion in the respective set.

    Raises:
        ValueError: If timestamp_col is not present in DataFrame or if split_date results
            in empty train or validation sets.
    """
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' not found in DataFrame. Available columns: {df.columns.tolist()}"
        )

    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Split based on date threshold
    train_mask = df[timestamp_col] <= split_date
    val_mask = df[timestamp_col] > split_date

    # Validation checks
    if train_mask.sum() == 0:
        raise ValueError(f"No records found with timestamp <= {split_date}. Check split_date.")

    if val_mask.sum() == 0:
        raise ValueError(f"No records found with timestamp > {split_date}. Check split_date.")

    # Additional safety check: ensure all validation timestamps are after train timestamps
    if train_mask.sum() > 0 and val_mask.sum() > 0:
        max_train_timestamp = df.loc[train_mask, timestamp_col].max()
        min_val_timestamp = df.loc[val_mask, timestamp_col].min()

        if min_val_timestamp <= max_train_timestamp:
            raise ValueError(
                f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
                f"is not greater than max train timestamp ({max_train_timestamp}). "
                "This indicates a data integrity issue."
            )

    return train_mask, val_mask


def get_split_date_from_ratio(
    df: pd.DataFrame, ratio: float, timestamp_col: str = constants.COL_TIMESTAMP
) -> pd.Timestamp:
    """Calculates split date based on ratio of data points.

    Args:
        df: DataFrame with timestamp column.
        ratio: Ratio of data to use for training (0 < ratio < 1).
            For example, 0.8 means 80% of data points go to train.
        timestamp_col: Name of the timestamp column.

    Returns:
        Timestamp threshold that splits data according to the ratio.

    Raises:
        ValueError: If ratio is not between 0 and 1, or if timestamp_col is missing.
    """
    if not 0 < ratio < 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Calculate threshold based on ratio
    sorted_timestamps = df[timestamp_col].sort_values()
    threshold_index = int(len(sorted_timestamps) * ratio)

    return sorted_timestamps.iloc[threshold_index]
