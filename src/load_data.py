from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_PATH = RAW_DATA_DIR / "imdb_reviews.csv"

EXPECTED_COLUMNS = [
    "review",
    "sentiment",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw IMDB reviews dataset.
    """
    if not path.exists():
        available_files = list(RAW_DATA_DIR.glob("*.csv"))
        available = "\n".join(file.name for file in available_files)

        raise FileNotFoundError(
            f"Dataset not found at {path}.\n\n"
            "Download the IMDB Dataset of 50K Movie Reviews from Kaggle, "
            "rename the file to 'imdb_reviews.csv', "
            "and place it inside data/raw/.\n\n"
            f"CSV files currently found in data/raw/:\n{available or 'None'}"
        )

    data = pd.read_csv(path)

    return data


def validate_columns(data: pd.DataFrame) -> None:
    """
    Validate expected dataset columns.
    """
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]

    if missing_columns:
        raise ValueError(
            "The dataset is missing expected columns:\n"
            f"{missing_columns}"
        )


def validate_basic_content(data: pd.DataFrame) -> None:
    """
    Validate basic dataset content.
    """
    if data.empty:
        raise ValueError("The dataset is empty.")

    invalid_reviews = data["review"].isna().sum()

    if invalid_reviews > 0:
        raise ValueError(f"Found {invalid_reviews:,} missing review values.")

    valid_sentiments = {"positive", "negative"}
    observed_sentiments = set(data["sentiment"].dropna().unique())

    unexpected_sentiments = observed_sentiments - valid_sentiments

    if unexpected_sentiments:
        raise ValueError(
            "Unexpected sentiment values found:\n"
            f"{unexpected_sentiments}\n"
            "Expected only: positive, negative."
        )


if __name__ == "__main__":
    reviews_data = load_raw_data()
    validate_columns(reviews_data)
    validate_basic_content(reviews_data)

    print("Raw IMDB reviews dataset loaded successfully.")
    print(f"Path: {RAW_DATA_PATH}")
    print(f"Rows: {reviews_data.shape[0]:,}")
    print(f"Columns: {reviews_data.shape[1]:,}")

    print("\nColumn names:")
    print(list(reviews_data.columns))

    print("\nSentiment distribution:")
    print(reviews_data["sentiment"].value_counts(dropna=False))

    print("\nSentiment distribution percentage:")
    print((reviews_data["sentiment"].value_counts(normalize=True) * 100).round(2))

    print("\nReview length summary in characters:")
    review_lengths = reviews_data["review"].astype(str).str.len()
    print(review_lengths.describe())

    print("\nSample rows:")
    print(reviews_data.head())