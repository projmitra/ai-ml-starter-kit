"""
03. K-Nearest Neighbors Demonstration
Implements distance-based instance learning with feature normalization.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from datasets.synthetic_generators import generate_multiclass_classification_data


def run_knn(n_neighbors=5, random_state=42):
    df = generate_multiclass_classification_data(n_samples=600, n_features=6, n_classes=3, random_state=random_state)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean"))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "algorithm": "K-Nearest Neighbors",
        "n_neighbors": n_neighbors,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    }
    return metrics, pipeline


if __name__ == "__main__":
    results, _ = run_knn()
    print("K-Nearest Neighbors Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
