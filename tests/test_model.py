import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_vectorizer, clean_text_series, load_sentiment140_data


DATA_PATH = "data/sample/sentiment140_sample.csv"


def train_tiny_model():
    df = load_sentiment140_data(DATA_PATH).sample(1000, random_state=42)
    df["clean_text"] = clean_text_series(df["text"])

    X = df["clean_text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("vectorizer", build_vectorizer(max_features=5000, ngram_range=(1, 1))),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


def test_model_predictions_have_correct_shape():
    model, X_test, y_test = train_tiny_model()

    predictions = model.predict(X_test)

    assert predictions.shape[0] == y_test.shape[0]


def test_model_predictions_are_binary():
    model, X_test, _ = train_tiny_model()

    predictions = model.predict(X_test)

    assert set(predictions).issubset({0, 1})


def test_model_meets_minimum_f1_threshold():
    model, X_test, y_test = train_tiny_model()

    predictions = model.predict(X_test)
    score = f1_score(y_test, predictions)

    assert score >= 0.55