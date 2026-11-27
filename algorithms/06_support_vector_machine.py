"""
06. Support Vector Machine Demonstration
Implements maximum-margin classification with Radial Basis Function (RBF) kernel.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from datasets.synthetic_generators import generate_binary_classification_data


def run_svm(c=1.0, kernel="rbf", random_state=42):
    df = generate_binary_classification_data(n_samples=500, n_features=6, random_state=random_state)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(C=c, kernel=kernel, random_state=random_state))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "algorithm": "Support Vector Machine",
        "kernel": kernel,
        "C": c,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    return metrics, pipeline


if __name__ == "__main__":
    results, _ = run_svm()
    print("Support Vector Machine Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
