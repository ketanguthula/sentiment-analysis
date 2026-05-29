import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from preprocess import clean_text, load_sentiment140_data


def create_monitoring_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()

    features["text_length"] = df["text"].astype(str).str.len()
    features["word_count"] = df["text"].astype(str).str.split().str.len()
    features["mention_count"] = df["text"].astype(str).str.count(r"@\w+")
    features["url_count"] = df["text"].astype(str).str.count(r"http\S+|www\S+")
    features["exclamation_count"] = df["text"].astype(str).str.count("!")
    features["clean_text_length"] = df["text"].astype(str).apply(clean_text).str.len()

    return features


def simulate_production_data(features: pd.DataFrame) -> pd.DataFrame:
    production = features.copy()

    production["text_length"] = production["text_length"] * 1.25
    production["word_count"] = production["word_count"] * 1.15
    production["exclamation_count"] = production["exclamation_count"] + 1

    return production


def monitor_drift(
    data_path="data/sample/sentiment140_sample.csv",
    drift_threshold=0.30,
):
    df = load_sentiment140_data(data_path)

    reference_df = df.sample(frac=0.7, random_state=42)
    current_df = df.drop(reference_df.index)

    reference_features = create_monitoring_features(reference_df)
    current_features = create_monitoring_features(current_df)
    current_features = simulate_production_data(current_features)

    report = Report([DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_features,
        current_data=current_features,
    )

    Path("reports").mkdir(exist_ok=True)
    snapshot.save_html("reports/sentiment_drift_report.html")

    snapshot_dict = snapshot.dict()

    drift_share = 0.0

    for metric in snapshot_dict.get("metrics", []):
        value = metric.get("value", {})

        if isinstance(value, dict):
            if "share" in value:
                drift_share = value["share"]

            if "number_of_drifted_columns" in value and "number_of_columns" in value:
                drift_share = (
                    value["number_of_drifted_columns"] / value["number_of_columns"]
                )

    print(f"Drift share: {drift_share:.2f}")
    print("Drift report saved to reports/sentiment_drift_report.html")

    if drift_share > drift_threshold:
        print("Drift threshold exceeded.")
        sys.exit(1)

    print("Drift is within acceptable limits.")


if __name__ == "__main__":
    monitor_drift()