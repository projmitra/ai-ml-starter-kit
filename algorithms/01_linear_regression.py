"""
01. Linear Regression Demonstration
Implements ordinary least squares regression with standard scaling in a pipeline.
Prevents data leakage by fitting transformers strictly on training data.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datasets.synthetic_generators import generate_regression_data


def run_linear_regression(random_state=42):
    df = generate_regression_data(n_samples=500, n_features=5, noise=15.0, random_state=random_state)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "algorithm": "Linear Regression",
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }
    return metrics, pipeline


if __name__ == "__main__":
    results, _ = run_linear_regression()
    print("Linear Regression Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
