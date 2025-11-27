"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean ratings and interaction counts
    to prevent data leakage from the test set.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data for calculations.

    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")


def add_user_genre_features(df: pd.DataFrame, train_df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Adds user mean rating per genre and preferred genre count.
    
    Computes on train_df to avoid leakage, then merges to df.
    """
    print("Adding user-genre features...")
    
    # Merge genres to train_df
    train_with_genres = train_df.merge(book_genres_df, on=constants.COL_BOOK_ID, how="left")
    
    # User mean rating per genre
    user_genre_mean = train_with_genres.groupby([constants.COL_USER_ID, constants.COL_GENRE_ID])[config.TARGET].mean().reset_index()
    user_genre_mean.columns = [constants.COL_USER_ID, constants.COL_GENRE_ID, 'user_genre_mean_rating']
    
    # For each user-book in df, get the genres and average the user's mean ratings across those genres
    df_with_genres = df.merge(book_genres_df, on=constants.COL_BOOK_ID, how="left")
    df_with_genres = df_with_genres.merge(user_genre_mean, on=[constants.COL_USER_ID, constants.COL_GENRE_ID], how="left")
    user_book_genre_mean = df_with_genres.groupby([constants.COL_USER_ID, constants.COL_BOOK_ID])['user_genre_mean_rating'].mean().reset_index()
    user_book_genre_mean.columns = [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.F_USER_GENRE_MEAN_RATING]
    
    # User's preferred genres count (genres with > mean rating)
    user_mean = train_df.groupby(constants.COL_USER_ID)[config.TARGET].mean().reset_index()
    user_mean.columns = [constants.COL_USER_ID, 'user_overall_mean']
    user_genre_pref = user_genre_mean.merge(user_mean, on=constants.COL_USER_ID)
    user_genre_pref['is_preferred'] = (user_genre_pref['user_genre_mean_rating'] > user_genre_pref['user_overall_mean']).astype(int)
    user_pref_count = user_genre_pref.groupby(constants.COL_USER_ID)['is_preferred'].sum().reset_index()
    user_pref_count.columns = [constants.COL_USER_ID, constants.F_USER_PREFERRED_GENRES_COUNT]
    
    # Merge to df
    df = df.merge(user_book_genre_mean, on=[constants.COL_USER_ID, constants.COL_BOOK_ID], how="left")
    df = df.merge(user_pref_count, on=constants.COL_USER_ID, how="left")
    
    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


def add_temporal_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds book age and user recency features."""
    print("Adding temporal features...")
    
    # Book age at interaction
    df['timestamp_year'] = df[constants.COL_TIMESTAMP].dt.year
    df[constants.F_BOOK_AGE] = df['timestamp_year'] - df[constants.COL_PUBLICATION_YEAR]
    df = df.drop('timestamp_year', axis=1)  # Cleanup
    
    # User's days since last read (compute on train, apply to all)
    user_last_ts = train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].max().reset_index()
    user_last_ts.columns = [constants.COL_USER_ID, 'user_last_ts']
    df = df.merge(user_last_ts, on=constants.COL_USER_ID, how="left")
    df[constants.F_DAYS_SINCE_LAST_READ] = (df[constants.COL_TIMESTAMP] - df['user_last_ts']).dt.days.abs()
    df = df.drop('user_last_ts', axis=1)
    
    return df


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions.

    Trains a TF-IDF vectorizer only on training data descriptions to avoid
    data leakage. Applies the vectorizer to all books and merges the features.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting the vectorizer.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with TF-IDF features added.
    """
    print("Adding text features (TF-IDF)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set
    train_books = train_df[constants.COL_BOOK_ID].unique()

    # Extract descriptions for training books only
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Check if vectorizer already exists (for prediction)
    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fit vectorizer on training descriptions only
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        # Save vectorizer for use in prediction
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    # Transform all book descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Get descriptions in the same order as df[book_id]
    # Create a mapping book_id -> description
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    # Get descriptions for books in df (in the same order)
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert sparse matrix to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    # Concatenate TF-IDF features with main DataFrame
    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf


def add_tfidf_svd_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    n_components: int = 100,
) -> pd.DataFrame:
    """Replace raw TF-IDF with compressed SVD components."""
    print(f"Adding TF-IDF → SVD ({n_components} components)...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    svd_path = config.MODEL_DIR / f"tfidf_svd_{n_components}.pkl"
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Load or fit the same vectorizer (exactly the same as before)
    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
    else:
        raise FileNotFoundError("Run the original pipeline once so the TF-IDF vectorizer is saved.")

    # Fit SVD only on training books (anti-leakage)
    train_books = train_df[constants.COL_BOOK_ID].unique()
    train_desc = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)][constants.COL_DESCRIPTION].fillna("")

    X_train_tfidf = vectorizer.transform(train_desc)

    if svd_path.exists():
        print(f"Loading precomputed SVD ({n_components} components)...")
        svd = joblib.load(svd_path)
    else:
        print(f"Fitting TruncatedSVD(n_components={n_components}) on train TF-IDF...")
        svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_STATE)
        svd.fit(X_train_tfidf)
        joblib.dump(svd, svd_path)
        print(f"SVD saved to {svd_path}")

    # Transform ALL descriptions with the fitted SVD
    all_desc = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_desc[constants.COL_DESCRIPTION] = all_desc[constants.COL_DESCRIPTION].fillna("")
    X_all_tfidf = vectorizer.transform(all_desc[constants.COL_DESCRIPTION])
    X_all_svd = svd.transform(X_all_tfidf)

    # Create DataFrame with SVD features
    svd_cols = [f"tfidf_svd_{i}" for i in range(n_components)]
    svd_df = pd.DataFrame(
        X_all_svd,
        columns=svd_cols,
        index=all_desc[constants.COL_BOOK_ID],
    ).reset_index()  # book_id as column

    # Merge to main df
    df = df.merge(svd_df, left_on=constants.COL_BOOK_ID, right_on=constants.COL_BOOK_ID, how="left")

    # Drop the old 500 raw tfidf_* columns
    old_tfidf_cols = [col for col in df.columns if col.startswith("tfidf_") and "_" in col.split("_")[1]]
    df = df.drop(columns=old_tfidf_cols, errors="ignore")

    print(f"Replaced raw TF-IDF with {n_components} SVD components.")
    return df


def add_bert_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BERT embeddings from book descriptions.

    Extracts 768-dimensional embeddings using a pre-trained Russian BERT model.
    Embeddings are cached on disk to avoid recomputation on subsequent runs.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency, not used for BERT).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with BERT embeddings added.
    """
    print("Adding text features (BERT embeddings)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings (this may take a while)...")
        print(f"Using device: {config.BERT_DEVICE}")

        # Limit GPU memory usage to prevent OOM errors
        if config.BERT_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.BERT_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.BERT_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(config.BERT_DEVICE)
        model.eval()

        # Prepare descriptions: get unique book_id -> description mapping
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

        # Get unique books and their descriptions
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        # Initialize embeddings dictionary
        embeddings_dict = {}

        # Process descriptions in batches
        num_batches = (len(descriptions) + config.BERT_BATCH_SIZE - 1) // config.BERT_BATCH_SIZE

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing BERT batches", unit="batch"):
                start_idx = batch_idx * config.BERT_BATCH_SIZE
                end_idx = min(start_idx + config.BERT_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(config.BERT_DEVICE) for k, v in encoded.items()}

                # Get model outputs
                outputs = model(**encoded)

                # Mean pooling: average over sequence length dimension
                # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
                attention_mask = encoded["attention_mask"]
                # Expand attention mask to match hidden_size dimension for broadcasting
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

                # Sum embeddings, weighted by attention mask
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                # Sum attention mask values for normalization
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

                # Mean pooling
                mean_pooled = sum_embeddings / sum_mask

                # Convert to numpy and store
                batch_embeddings = mean_pooled.cpu().numpy()

                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                # Small pause between batches to let GPU cool down and prevent overheating
                if config.BERT_DEVICE == "cuda":
                    time.sleep(0.2)  # 200ms pause between batches

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved BERT embeddings to {embeddings_path}")

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Create DataFrame with BERT features
    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)

    # Concatenate BERT features with main DataFrame
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(bert_feature_names)} BERT features.")
    return df_with_bert


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Fills missing values using a defined strategy.

    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data, used for calculating fill metrics.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features for "cold start" users/items (only if they exist)
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    df[constants.F_USER_GENRE_MEAN_RATING] = df[constants.F_USER_GENRE_MEAN_RATING].fillna(global_mean)
    df[constants.F_USER_PREFERRED_GENRES_COUNT] = df[constants.F_USER_PREFERRED_GENRES_COUNT].fillna(0)
    df[constants.F_BOOK_AGE] = df[constants.F_BOOK_AGE].fillna(0)
    df[constants.F_DAYS_SINCE_LAST_READ] = df[constants.F_DAYS_SINCE_LAST_READ].fillna(0)

    # Fill TF-IDF features with 0 (for books without descriptions)
    # tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    # for col in tfidf_cols:
    #     df[col] = df[col].fillna(0.0)

    svd_cols = [col for col in df.columns if col.startswith("tfidf_svd_")]
    for col in svd_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0 (for books without descriptions)
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df


def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.

    This function orchestrates the calls to add aggregate features (optional), genre
    features, text features (TF-IDF and BERT), and handle missing values.

    Args:
        df (pd.DataFrame): The merged DataFrame from `data_processing`.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
        include_aggregates (bool): If True, compute aggregate features. Defaults to False.
            Aggregates are typically computed separately during training to avoid data leakage.

    Returns:
        pd.DataFrame: The final DataFrame with all features engineered.
    """
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Aggregate features are computed separately during training to ensure
    # no data leakage from validation set timestamps
    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_user_genre_features(df, train_df, book_genres_df)
    df = add_genre_features(df, book_genres_df)
    df = add_temporal_features(df, train_df)
    df = add_text_features(df, train_df, descriptions_df)
    # df = add_tfidf_svd_features(df, train_df, descriptions_df, n_components=100)
    df = add_bert_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
