from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "imdb_reviews.csv"


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


def count_html_patterns(text_series: pd.Series) -> dict:
    """
    Count common HTML patterns in raw reviews.
    """
    text_as_string = text_series.astype(str)

    return {
        "reviews_with_br_tags": text_as_string.str.contains(r"<br\s*/?>", regex=True).sum(),
        "reviews_with_any_html_tag": text_as_string.str.contains(r"<[^>]+>", regex=True).sum(),
        "reviews_with_html_entities": text_as_string.str.contains(r"&[a-zA-Z]+;", regex=True).sum(),
    }


def build_length_report(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build review length summary.
    """
    review_text = data["review"].astype(str)

    length_report = pd.DataFrame(
        {
            "character_length": review_text.str.len(),
            "word_count": review_text.str.split().str.len(),
        }
    )

    return length_report


def audit_data(data: pd.DataFrame) -> None:
    """
    Print initial NLP dataset audit.
    """
    print("=" * 80)
    print("IMDB SENTIMENT DATA AUDIT")
    print("=" * 80)

    print("\n1. Dataset shape")
    print(f"Rows: {data.shape[0]:,}")
    print(f"Columns: {data.shape[1]:,}")

    print("\n2. Column names")
    print(list(data.columns))

    print("\n3. Data types")
    print(data.dtypes)

    print("\n4. Missing values")
    missing_report = pd.DataFrame(
        {
            "missing_count": data.isna().sum(),
            "missing_percent": (data.isna().mean() * 100).round(2),
            "dtype": data.dtypes.astype(str),
        }
    )
    print(missing_report)

    print("\n5. Duplicate rows")
    print(f"Exact duplicate rows: {data.duplicated().sum():,}")

    print("\n6. Duplicate review texts")
    print(f"Duplicate review texts: {data['review'].duplicated().sum():,}")

    print("\n7. Sentiment distribution")
    sentiment_counts = data["sentiment"].value_counts(dropna=False)
    sentiment_percent = data["sentiment"].value_counts(normalize=True, dropna=False) * 100

    sentiment_report = pd.DataFrame(
        {
            "count": sentiment_counts,
            "percent": sentiment_percent.round(2),
        }
    )
    print(sentiment_report)

    print("\n8. HTML pattern audit")
    html_report = count_html_patterns(data["review"])
    for key, value in html_report.items():
        print(f"{key}: {value:,}")

    print("\n9. Review length summary")
    length_report = build_length_report(data)

    print("\nCharacter length")
    print(length_report["character_length"].describe())

    print("\nWord count")
    print(length_report["word_count"].describe())

    print("\n10. Review length by sentiment")
    length_by_sentiment = (
        pd.concat([data["sentiment"], length_report], axis=1)
        .groupby("sentiment")
        .agg(
            reviews=("sentiment", "count"),
            mean_character_length=("character_length", "mean"),
            median_character_length=("character_length", "median"),
            mean_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
        )
        .round(2)
    )
    print(length_by_sentiment)

    print("\n11. Shortest reviews")
    shortest_reviews = (
        pd.concat([data[["review", "sentiment"]], length_report], axis=1)
        .sort_values("character_length")
        .head(5)
    )

    for index, row in shortest_reviews.iterrows():
        print("-" * 80)
        print(f"Sentiment: {row['sentiment']}")
        print(f"Characters: {row['character_length']}")
        print(row["review"][:500])

    print("\n12. Longest reviews")
    longest_reviews = (
        pd.concat([data[["review", "sentiment"]], length_report], axis=1)
        .sort_values("character_length", ascending=False)
        .head(5)
    )

    for index, row in longest_reviews.iterrows():
        print("-" * 80)
        print(f"Sentiment: {row['sentiment']}")
        print(f"Characters: {row['character_length']}")
        print(row["review"][:500])

    print("\n13. Sample rows")
    print(data.head())


if __name__ == "__main__":
    reviews_data = load_raw_data()
    audit_data(reviews_data)