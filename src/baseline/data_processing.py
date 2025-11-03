"""
Data loading and merging script.
"""

from typing import Any

import pandas as pd

from . import config, constants


def load_and_merge_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw data files and merges them into a single DataFrame.

    Combines train and test sets, then joins user and book metadata. The genre
    and description data are returned separately as they're needed for feature engineering.

    Only training records with has_read=1 (books that received a rating) are used
    for training. Records with has_read=0 are excluded.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The merged DataFrame (train + test + metadata).
            - The book_genres DataFrame.
            - The genres DataFrame.
            - The book_descriptions DataFrame.
    """
    print("Loading data...")

    # Define dtypes for memory optimization
    dtype_spec: dict[str, Any] = {
        constants.COL_USER_ID: "int32",
        constants.COL_BOOK_ID: "int32",
        constants.COL_TARGET: "float32",
        constants.COL_GENDER: "category",
        constants.COL_AGE: "float32",
        constants.COL_AUTHOR_ID: "int32",
        constants.COL_PUBLICATION_YEAR: "float32",
        constants.COL_LANGUAGE: "category",
        constants.COL_PUBLISHER: "category",
        constants.COL_AVG_RATING: "float32",
        constants.COL_GENRE_ID: "int16",
    }

    # Load datasets
    # CSV files use comma as separator (default pandas behavior)
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k in [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_TARGET]
        },
        parse_dates=[constants.COL_TIMESTAMP],
    )

    # Filter training data: only use books that received a rating (has_read=1)
    initial_count = len(train_df)
    train_df = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    filtered_count = len(train_df)
    print(f"Filtered training data: {initial_count} -> {filtered_count} rows (only has_read=1)")
    test_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TEST_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_BOOK_ID]},
    )
    user_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.USER_DATA_FILENAME,
        dtype={
            k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_GENDER, constants.COL_AGE]
        },
    )
    book_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k
            in [
                constants.COL_BOOK_ID,
                constants.COL_AUTHOR_ID,
                constants.COL_PUBLICATION_YEAR,
                constants.COL_LANGUAGE,
                constants.COL_AVG_RATING,
                constants.COL_PUBLISHER,
            ]
        },
    )
    book_genres_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_BOOK_ID, constants.COL_GENRE_ID]},
    )
    genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.GENRES_FILENAME)
    book_descriptions_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME,
        dtype={constants.COL_BOOK_ID: "int32"},
    )

    print("Data loaded. Merging datasets...")

    # Combine train and test
    train_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TRAIN
    test_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TEST
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    # Join metadata
    combined_df = combined_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")

    # Drop duplicates from book_data_df before merging
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    combined_df = combined_df.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")

    print(f"Merged data shape: {combined_df.shape}")
    return combined_df, book_genres_df, genres_df, book_descriptions_df
