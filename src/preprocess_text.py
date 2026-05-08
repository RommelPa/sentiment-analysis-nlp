from pathlib import Path
import html
import json
import re

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "imdb_reviews.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "imdb_reviews_clean.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_reviews.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_reviews.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_reviews.csv"
PREPROCESSING_METADATA_PATH = PROCESSED_DATA_DIR / "preprocessing_metadata.json"

TEXT_COLUMN = "review"
CLEAN_TEXT_COLUMN = "clean_review"
TARGET_COLUMN = "sentiment"
TARGET_LABEL_COLUMN = "sentiment_label"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE_FROM_REMAINING = 0.1764705882
# 0.1764705882 of 85% ≈ 15% of the original dataset.
# Final split: 70% train, 15% validation, 15% test.


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw IMDB reviews dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. "
            "Place imdb_reviews.csv inside data/raw/."
        )

    return pd.read_csv(path)


def validate_no_conflicting_duplicate_reviews(data: pd.DataFrame) -> None:
    """
    Validate that duplicate review texts do not have conflicting sentiment labels.
    """
    duplicate_label_counts = (
        data.groupby(TEXT_COLUMN)[TARGET_COLUMN]
        .nunique()
        .reset_index(name="unique_sentiment_count")
    )

    conflicting_duplicates = duplicate_label_counts[
        duplicate_label_counts["unique_sentiment_count"] > 1
    ]

    if not conflicting_duplicates.empty:
        raise ValueError(
            "Found duplicate review texts with conflicting sentiment labels. "
            "Manual review is required before preprocessing."
        )


def clean_review_text(text: str) -> str:
    """
    Clean raw review text for TF-IDF modeling.

    The cleaning keeps words and apostrophes because negations and contractions
    can be important in sentiment analysis.
    """
    text = str(text)

    # Decode HTML entities first.
    text = html.unescape(text)

    # Remove HTML tags.
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # Lowercase.
    text = text.lower()

    # Remove URLs.
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Keep alphabetic characters and apostrophes. Replace the rest with spaces.
    text = re.sub(r"[^a-zA-Z']", " ", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates, clean text, and create binary sentiment labels.
    """
    data = data.copy()

    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str)
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip().str.lower()

    validate_no_conflicting_duplicate_reviews(data)

    initial_rows = len(data)

    data = (
        data.drop_duplicates(subset=[TEXT_COLUMN])
        .reset_index(drop=True)
    )

    duplicates_removed = initial_rows - len(data)

    data[CLEAN_TEXT_COLUMN] = data[TEXT_COLUMN].apply(clean_review_text)

    data[TARGET_LABEL_COLUMN] = data[TARGET_COLUMN].map(
        {
            "negative": 0,
            "positive": 1,
        }
    )

    if data[TARGET_LABEL_COLUMN].isna().any():
        unexpected_values = data.loc[
            data[TARGET_LABEL_COLUMN].isna(),
            TARGET_COLUMN,
        ].unique()

        raise ValueError(
            "Unexpected sentiment values found after mapping:\n"
            f"{unexpected_values}"
        )

    data[TARGET_LABEL_COLUMN] = data[TARGET_LABEL_COLUMN].astype(int)

    data["raw_character_length"] = data[TEXT_COLUMN].str.len()
    data["clean_character_length"] = data[CLEAN_TEXT_COLUMN].str.len()
    data["clean_word_count"] = data[CLEAN_TEXT_COLUMN].str.split().str.len()

    empty_clean_reviews = data[CLEAN_TEXT_COLUMN].eq("").sum()

    if empty_clean_reviews > 0:
        raise ValueError(
            f"Found {empty_clean_reviews:,} empty reviews after cleaning. "
            "Review cleaning rules need adjustment."
        )

    data.attrs["duplicates_removed"] = duplicates_removed

    return data


def create_stratified_splits(
    clean_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train, validation, and test splits.

    Random stratified splitting is appropriate here because this dataset has no
    temporal ordering requirement.
    """
    train_validation_data, test_data = train_test_split(
        clean_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean_data[TARGET_LABEL_COLUMN],
    )

    train_data, validation_data = train_test_split(
        train_validation_data,
        test_size=VALIDATION_SIZE_FROM_REMAINING,
        random_state=RANDOM_STATE,
        stratify=train_validation_data[TARGET_LABEL_COLUMN],
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


def build_metadata(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict:
    """
    Build preprocessing metadata.
    """
    metadata = {
        "raw_rows": int(len(raw_data)),
        "clean_rows": int(len(clean_data)),
        "duplicates_removed": int(clean_data.attrs.get("duplicates_removed", 0)),
        "target_column": TARGET_COLUMN,
        "target_label_column": TARGET_LABEL_COLUMN,
        "text_column": TEXT_COLUMN,
        "clean_text_column": CLEAN_TEXT_COLUMN,
        "random_state": RANDOM_STATE,
        "splits": {
            "train": {
                "rows": int(len(train_data)),
                "positive_share": float(train_data[TARGET_LABEL_COLUMN].mean()),
            },
            "validation": {
                "rows": int(len(validation_data)),
                "positive_share": float(validation_data[TARGET_LABEL_COLUMN].mean()),
            },
            "test": {
                "rows": int(len(test_data)),
                "positive_share": float(test_data[TARGET_LABEL_COLUMN].mean()),
            },
        },
    }

    return metadata


def save_processed_outputs(
    clean_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metadata: dict,
) -> None:
    """
    Save clean data, modeling splits, and metadata.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    clean_data.to_csv(CLEAN_DATA_PATH, index=False)
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    validation_data.to_csv(VALIDATION_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    with open(PREPROCESSING_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def print_preprocessing_summary(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metadata: dict,
) -> None:
    """
    Print preprocessing summary.
    """
    print("=" * 80)
    print("IMDB TEXT PREPROCESSING SUMMARY")
    print("=" * 80)

    print("\n1. Dataset rows")
    print(f"Raw rows: {len(raw_data):,}")
    print(f"Clean rows: {len(clean_data):,}")
    print(f"Duplicates removed: {metadata['duplicates_removed']:,}")

    print("\n2. Split sizes")
    print(f"Train rows: {len(train_data):,}")
    print(f"Validation rows: {len(validation_data):,}")
    print(f"Test rows: {len(test_data):,}")

    print("\n3. Target distribution - clean data")
    clean_distribution = pd.DataFrame(
        {
            "count": clean_data[TARGET_COLUMN].value_counts(),
            "percent": (
                clean_data[TARGET_COLUMN].value_counts(normalize=True) * 100
            ).round(2),
        }
    )
    print(clean_distribution)

    print("\n4. Target distribution - train")
    print(
        pd.DataFrame(
            {
                "count": train_data[TARGET_COLUMN].value_counts(),
                "percent": (
                    train_data[TARGET_COLUMN].value_counts(normalize=True) * 100
                ).round(2),
            }
        )
    )

    print("\n5. Target distribution - validation")
    print(
        pd.DataFrame(
            {
                "count": validation_data[TARGET_COLUMN].value_counts(),
                "percent": (
                    validation_data[TARGET_COLUMN].value_counts(normalize=True) * 100
                ).round(2),
            }
        )
    )

    print("\n6. Target distribution - test")
    print(
        pd.DataFrame(
            {
                "count": test_data[TARGET_COLUMN].value_counts(),
                "percent": (
                    test_data[TARGET_COLUMN].value_counts(normalize=True) * 100
                ).round(2),
            }
        )
    )

    print("\n7. Clean text length summary")
    print(clean_data[["clean_character_length", "clean_word_count"]].describe())

    print("\n8. Example before/after cleaning")
    example = clean_data.iloc[0]
    print("\nRaw review:")
    print(example[TEXT_COLUMN][:600])
    print("\nClean review:")
    print(example[CLEAN_TEXT_COLUMN][:600])

    print("\n9. Files saved")
    print(f"Clean data: {CLEAN_DATA_PATH}")
    print(f"Train data: {TRAIN_DATA_PATH}")
    print(f"Validation data: {VALIDATION_DATA_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Metadata: {PREPROCESSING_METADATA_PATH}")


if __name__ == "__main__":
    raw_reviews = load_raw_data()
    clean_reviews = preprocess_data(raw_reviews)

    train_reviews, validation_reviews, test_reviews = create_stratified_splits(
        clean_reviews
    )

    preprocessing_metadata = build_metadata(
        raw_data=raw_reviews,
        clean_data=clean_reviews,
        train_data=train_reviews,
        validation_data=validation_reviews,
        test_data=test_reviews,
    )

    save_processed_outputs(
        clean_data=clean_reviews,
        train_data=train_reviews,
        validation_data=validation_reviews,
        test_data=test_reviews,
        metadata=preprocessing_metadata,
    )

    print_preprocessing_summary(
        raw_data=raw_reviews,
        clean_data=clean_reviews,
        train_data=train_reviews,
        validation_data=validation_reviews,
        test_data=test_reviews,
        metadata=preprocessing_metadata,
    )