"""
04. Decision Tree Demonstration
Implements tree-based recursive binary partitioning with constrained maximum depth to control overfitting.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from datasets.synthetic_generators import generate_binary_classification_data


def run_decision_tree(max_depth=4, random_state=42):
    df = generate_binary_classification_data(n_samples=500, n_features=6, random_state=random_state)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=max_depth, criterion="gini", random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "algorithm": "Decision Tree",
        "max_depth": max_depth,
        "tree_depth_actual": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    return metrics, model


if __name__ == "__main__":
    results, _ = run_decision_tree()
    print("Decision Tree Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
