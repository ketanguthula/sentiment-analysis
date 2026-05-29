import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from preprocess import clean_text_series, load_sentiment140_data


def evaluate_model(
    model_path="models/best_model.joblib",
    data_path="data/sample/sentiment140_sample.csv",
    test_size=0.2,
    random_state=42,
):
    model = joblib.load(model_path)

    df = load_sentiment140_data(data_path)
    df["clean_text"] = clean_text_series(df["text"])

    X = df["clean_text"]
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }

    return metrics


def print_metrics(metrics):
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    metrics = evaluate_model()
    print_metrics(metrics)