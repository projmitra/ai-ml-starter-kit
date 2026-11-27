"""
09. Principal Component Analysis Demonstration
Implements orthogonal linear feature projection and dimensionality reduction.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datasets.synthetic_generators import generate_regression_data


def run_pca(n_components=2, random_state=42):
    df = generate_regression_data(n_samples=500, n_features=6, random_state=random_state)
    X = df.drop(columns=["target"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_projected = pca.fit_transform(X_scaled)

    evr = [float(v) for v in pca.explained_variance_ratio_]
    cumulative_evr = float(np.sum(evr))

    metrics = {
        "algorithm": "Principal Component Analysis",
        "original_features": X.shape[1],
        "projected_components": n_components,
        "explained_variance_ratio": evr,
        "cumulative_variance_explained": cumulative_evr
    }
    return metrics, pca


if __name__ == "__main__":
    results, _ = run_pca()
    print("Principal Component Analysis Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
