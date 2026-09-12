"""Train and compare three grade predictors on the same held-out students.

Run from the repository root: python python/regression.py
"""

from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80"
FEATURES = {
    "mean_baseline": ["G2"],  # DummyRegressor ignores this input.
    "simple_G2": ["G2"],
    "multiple_G1_G2": ["G1", "G2"],
}
LABELS = {
    "mean_baseline": "Training mean",
    "simple_G2": "Simple: G2",
    "multiple_G1_G2": "Multiple: G1 + G2",
}


def load_data(root=ROOT):
    """Validate the original CSV and align a shared, one-based row-ID split."""
    root = Path(root)
    raw_path = root / "data/student-mat.csv"
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset changed: restore the bundled student-mat.csv before using its split.")
    data = pd.read_csv(raw_path, sep=";")
    split = pd.read_csv(root / "data/split.csv")
    expected_ids = np.arange(1, len(data) + 1)
    if list(split.columns) != ["row_id", "split"]:
        raise ValueError("split.csv must have row_id and split columns.")
    split = split.sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(split.row_id.to_numpy(), expected_ids):
        raise ValueError("Split IDs must cover each original CSV row exactly once.")
    if split.split.value_counts().to_dict() != {"train": 316, "test": 79}:
        raise ValueError("Expected 316 training records and 79 test records.")
    grades = data[["G1", "G2", "G3"]]
    if grades.isna().any().any() or not grades.apply(lambda s: s.between(0, 20).all()).all():
        raise ValueError("Grades must be present and between 0 and 20.")
    data.insert(0, "row_id", expected_ids)
    data["split"] = split["split"].to_numpy()
    return data


def fit_models(train):
    """Fit only on training rows. G3 is the target, never an input feature."""
    models = {}
    for name, features in FEATURES.items():
        estimator = DummyRegressor(strategy="mean") if name == "mean_baseline" else LinearRegression()
        models[name] = estimator.fit(train[features], train["G3"])
    return models


def score_models(models, test):
    """Keep raw OLS predictions: do not round or clip before evaluation."""
    metrics, predictions = [], []
    actual = test["G3"].to_numpy()
    for name, model in models.items():
        predicted = model.predict(test[FEATURES[name]])
        metrics.append({
            "model": name, "test_rows": len(test),
            "mae": mean_absolute_error(actual, predicted),
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "r2": r2_score(actual, predicted),
        })
        predictions.append(pd.DataFrame({
            "row_id": test["row_id"].to_numpy(), "model": name,
            "actual": actual, "predicted": predicted, "residual": actual - predicted,
        }))
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def coefficient_table(models):
    rows = []
    for name, model in models.items():
        if name == "mean_baseline":
            values = [("intercept", float(model.constant_.item()))]
        else:
            values = [("intercept", float(model.intercept_))] + list(zip(FEATURES[name], model.coef_))
        rows.extend({"model": name, "term": term, "estimate": float(value)} for term, value in values)
    return pd.DataFrame(rows)


def plot_overview(train, models, metrics, predictions, output):
    """One figure showing the fitted line, test errors, and a baseline comparison."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#253343", "text.color": "#253343",
        "axes.titleweight": "bold", "svg.fonttype": "none", "svg.hashsalt": "student-regression",
    })
    blue, neutral = "#2663A6", "#4E5A69"
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    figure.suptitle("Student performance · linear regression", fontsize=21, fontweight="bold")
    a, b, c, d = axes.flat
    a.scatter(train.G2, train.G3, s=29, alpha=0.4, color=blue, edgecolors="none")
    grid = pd.DataFrame({"G2": np.linspace(0, 20, 100)})
    a.plot(grid.G2, models["simple_G2"].predict(grid), color=neutral, linewidth=2, label="OLS fitted on training rows")
    a.set(title="Earlier grade and final grade · train n=316", xlabel="Second-period grade (G2, /20)", ylabel="Final grade (G3, /20)", xlim=(-0.5, 20.5), ylim=(-2, 22))
    a.legend(loc="upper left", frameon=False, fontsize=9)
    multi = predictions[predictions.model == "multiple_G1_G2"]
    lower = min(-0.5, float(multi.predicted.min()) - 0.5)
    upper = max(20.5, float(multi.predicted.max()) + 0.5)
    b.scatter(multi.actual, multi.predicted, s=40, alpha=0.6, color=blue, edgecolors="none")
    b.plot([lower, upper], [lower, upper], color=neutral, linestyle="--", linewidth=1.5, label="Perfect prediction")
    b.set(title="Multiple regression · test n=79", xlabel="Actual final grade (/20)", ylabel="Predicted final grade (/20)", xlim=(lower, upper), ylim=(lower, upper))
    b.set_aspect("equal", adjustable="box")
    b.legend(frameon=False, fontsize=9)
    c.scatter(multi.predicted, multi.residual, s=40, alpha=0.6, color=blue, edgecolors="none")
    c.axhline(0, color=neutral, linestyle="--", linewidth=1.5)
    c.set(title="Multiple regression residuals · test n=79", xlabel="Predicted final grade (/20)", ylabel="Actual − predicted (grade points)")
    names = [LABELS[name] for name in metrics.model]
    bars = d.barh(names, metrics.rmse, color=["#B4BDC7", blue, blue], height=0.55)
    d.invert_yaxis()
    d.bar_label(bars, fmt="%.2f", padding=6, fontsize=11)
    d.set(title="Error on the same 79 test rows", xlabel="RMSE in grade points · lower is better", xlim=(0, float(metrics.rmse.max()) * 1.22))
    for axis in axes.flat:
        axis.set_axisbelow(True)
        axis.grid(axis="x" if axis is d else "both", color="#E6E9EE", linewidth=0.7)
    figure.supxlabel("Source: UCI Student Performance, mathematics · one fixed split · raw predictions, zero grades retained", fontsize=10)
    metadata = {"Date": None} if Path(output).suffix.lower() == ".svg" else {}
    figure.savefig(output, metadata=metadata, facecolor="white")
    plt.close(figure)


def run_analysis(root=ROOT):
    root = Path(root)
    data = load_data(root)
    train = data[data.split == "train"].copy()
    test = data[data.split == "test"].copy()
    models = fit_models(train)
    metrics, predictions = score_models(models, test)
    output = root / "reports/python"
    figures = root / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    coefficient_table(models).to_csv(output / "coefficients.csv", index=False)
    multiple = models["multiple_G1_G2"]
    model_card = {
        "schema_version": 1, "target": "G3", "features": FEATURES["multiple_G1_G2"],
        "intercept": float(multiple.intercept_),
        "coefficients": {f: float(c) for f, c in zip(FEATURES["multiple_G1_G2"], multiple.coef_)},
        "grade_range": [0, 20], "training_rows": len(train), "dataset_sha256": DATA_SHA256,
        "note": "Trained on the training split only; educational late-term prediction.",
    }
    (output / "model.json").write_text(json.dumps(model_card, indent=2) + "\n")
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
    (output / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    quality = {"rows": len(data), "source_columns": 33, "missing_cells": int(data.isna().sum().sum()),
               "duplicate_source_rows": int(data.drop(columns=["row_id", "split"]).duplicated().sum()),
               "zero_final_grades_retained": int((data.G3 == 0).sum()),
               "train_rows": len(train), "test_rows": len(test)}
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    plot_overview(train, models, metrics, predictions, figures / "regression_overview.svg")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOutputs saved in reports/python and reports/figures.")
    return metrics


if __name__ == "__main__":
    run_analysis()
