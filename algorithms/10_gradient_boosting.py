"""
10. Gradient Boosting Demonstration
Implements sequential residual-minimizing ensemble boosting for classification.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datasets.synthetic_generators import generate_binary_classification_data


def run_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    df = generate_binary_classification_data(n_samples=500, n_features=6, random_state=random_state)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "algorithm": "Gradient Boosting",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    return metrics, model


if __name__ == "__main__":
    results, _ = run_gradient_boosting()
    print("Gradient Boosting Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
