"""
08. K-Means Clustering Demonstration
Implements centroid-based partition clustering evaluated with Silhouette Score and Davies-Bouldin index.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from datasets.synthetic_generators import generate_clustering_data


def run_kmeans(n_clusters=3, random_state=42):
    df, y_true = generate_clustering_data(n_samples=450, n_features=4, centers=n_clusters, random_state=random_state)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = float(silhouette_score(X_scaled, labels))
    db_score = float(davies_bouldin_score(X_scaled, labels))
    inertia = float(kmeans.inertia_)

    metrics = {
        "algorithm": "K-Means Clustering",
        "n_clusters": n_clusters,
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "inertia": inertia
    }
    return metrics, kmeans


if __name__ == "__main__":
    results, _ = run_kmeans()
    print("K-Means Clustering Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
