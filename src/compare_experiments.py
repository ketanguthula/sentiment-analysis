import mlflow
import pandas as pd


mlflow.set_tracking_uri("file:./mlruns")

experiment = mlflow.get_experiment_by_name("sentiment140_experiments")

if experiment is None:
    raise ValueError("Experiment not found.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id]
)

if runs.empty:
    raise ValueError("No MLflow runs found.")

runs = runs.sort_values(
    by="metrics.f1",
    ascending=False
)

best_run = runs.iloc[0]

print("Best Run:")
print(f"Run ID: {best_run['run_id']}")
print(f"F1 Score: {best_run['metrics.f1']:.4f}")
print(f"Accuracy: {best_run['metrics.accuracy']:.4f}")
print(f"Precision: {best_run['metrics.precision']:.4f}")
print(f"Recall: {best_run['metrics.recall']:.4f}")

print("\nModel Parameters:")
for column in runs.columns:
    if column.startswith("params."):
        print(f"{column.replace('params.', '')}: {best_run[column]}")