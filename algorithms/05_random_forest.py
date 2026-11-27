"""
05. Random Forest Demonstration
Implements bootstrap aggregating (bagging) of de-correlated decision trees.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datasets.synthetic_generators import generate_binary_classification_data


def run_random_forest(n_estimators=100, max_depth=5, random_state=42):
    df = generate_binary_classification_data(n_samples=500, n_features=6, random_state=random_state)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "algorithm": "Random Forest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    return metrics, model


if __name__ == "__main__":
    results, _ = run_random_forest()
    print("Random Forest Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
