"""
Synthetic Dataset Generators for Educational Machine Learning Demonstrations.
All datasets are synthesized deterministically using controlled random seeds.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs


def generate_regression_data(n_samples=500, n_features=5, noise=10.0, random_state=42):
    """
    Generates synthetic continuous tabular data for linear and non-linear regression.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def generate_binary_classification_data(n_samples=500, n_features=6, class_sep=1.3, random_state=42):
    """
    Generates synthetic binary classification data with controlled class separation.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        class_sep=class_sep,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def generate_multiclass_classification_data(n_samples=600, n_features=6, n_classes=3, class_sep=1.3, random_state=42):
    """
    Generates synthetic multi-class classification data with controlled class separation.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def generate_clustering_data(n_samples=450, n_features=4, centers=3, cluster_std=1.0, random_state=42):
    """
    Generates synthetic isotropic Gaussian blobs for clustering demonstrations.
    """
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    return df, y_true
