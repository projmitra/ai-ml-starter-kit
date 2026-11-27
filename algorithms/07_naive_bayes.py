"""
07. Naive Bayes Demonstration
Implements probabilistic classification under Gaussian conditional independence assumption.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from datasets.synthetic_generators import generate_multiclass_classification_data


def run_naive_bayes(random_state=42):
    df = generate_multiclass_classification_data(n_samples=600, n_features=6, n_classes=3, random_state=random_state)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "algorithm": "Gaussian Naive Bayes",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    }
    return metrics, model


if __name__ == "__main__":
    results, _ = run_naive_bayes()
    print("Gaussian Naive Bayes Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
