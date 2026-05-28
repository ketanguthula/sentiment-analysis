import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


COLUMN_NAMES = ["target", "id", "date", "flag", "user", "text"]


def load_sentiment140_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        names=COLUMN_NAMES,
    )

    df = df[["target", "text"]].copy()
    df["target"] = df["target"].map({0: 0, 4: 1})

    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_series(text_series: pd.Series) -> pd.Series:
    if not isinstance(text_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    return text_series.apply(clean_text)


def build_vectorizer(max_features=50000, ngram_range=(1, 2)):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
        stop_words="english",
    )