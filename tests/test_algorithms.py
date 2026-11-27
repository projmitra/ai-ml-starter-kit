"""
Automated Test Suite for Machine Learning Algorithms and Synthetic Generators.
Verifies algorithm execution, output contracts, metric validity, determinism, and data leakage prevention.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.synthetic_generators import (
    generate_regression_data,
    generate_binary_classification_data,
    generate_multiclass_classification_data,
    generate_clustering_data
)

import importlib

# Dynamic import of numbered algorithm modules
alg_01 = importlib.import_module("algorithms.01_linear_regression")
alg_02 = importlib.import_module("algorithms.02_logistic_regression")
alg_03 = importlib.import_module("algorithms.03_k_nearest_neighbors")
alg_04 = importlib.import_module("algorithms.04_decision_tree")
alg_05 = importlib.import_module("algorithms.05_random_forest")
alg_06 = importlib.import_module("algorithms.06_support_vector_machine")
alg_07 = importlib.import_module("algorithms.07_naive_bayes")
alg_08 = importlib.import_module("algorithms.08_k_means_clustering")
alg_09 = importlib.import_module("algorithms.09_principal_component_analysis")
alg_10 = importlib.import_module("algorithms.10_gradient_boosting")


class TestSyntheticGenerators(unittest.TestCase):
    def test_regression_generator(self):
        df = generate_regression_data(n_samples=100, n_features=4, random_state=42)
        self.assertEqual(len(df), 100)
        self.assertIn("target", df.columns)
        self.assertEqual(len(df.columns), 5)

    def test_binary_classification_generator(self):
        df = generate_binary_classification_data(n_samples=120, n_features=5, random_state=42)
        self.assertEqual(len(df), 120)
        self.assertIn("target", df.columns)
        self.assertEqual(set(df["target"].unique()), {0, 1})

    def test_multiclass_generator(self):
        df = generate_multiclass_classification_data(n_samples=150, n_features=6, n_classes=3, random_state=42)
        self.assertEqual(len(df), 150)
        self.assertEqual(len(df["target"].unique()), 3)

    def test_clustering_generator(self):
        df, labels = generate_clustering_data(n_samples=90, n_features=3, centers=3, random_state=42)
        self.assertEqual(len(df), 90)
        self.assertEqual(len(labels), 90)
        self.assertEqual(len(np.unique(labels)), 3)


class TestAlgorithms(unittest.TestCase):
    def test_01_linear_regression(self):
        metrics, pipeline = alg_01.run_linear_regression(random_state=42)
        self.assertEqual(metrics["algorithm"], "Linear Regression")
        self.assertGreater(metrics["r2_score"], 0.70)
        self.assertGreater(metrics["samples_train"], metrics["samples_test"])

    def test_02_logistic_regression(self):
        metrics, pipeline = alg_02.run_logistic_regression(random_state=42)
        self.assertEqual(metrics["algorithm"], "Logistic Regression")
        self.assertGreaterEqual(metrics["accuracy"], 0.75)
        self.assertGreaterEqual(metrics["roc_auc"], 0.80)

    def test_03_k_nearest_neighbors(self):
        metrics, pipeline = alg_03.run_knn(n_neighbors=5, random_state=42)
        self.assertEqual(metrics["algorithm"], "K-Nearest Neighbors")
        self.assertGreaterEqual(metrics["accuracy"], 0.70)
        self.assertGreaterEqual(metrics["f1_macro"], 0.70)

    def test_04_decision_tree(self):
        metrics, model = alg_04.run_decision_tree(max_depth=4, random_state=42)
        self.assertEqual(metrics["algorithm"], "Decision Tree")
        self.assertLessEqual(metrics["tree_depth_actual"], 4)
        self.assertGreaterEqual(metrics["accuracy"], 0.70)

    def test_05_random_forest(self):
        metrics, model = alg_05.run_random_forest(n_estimators=50, max_depth=5, random_state=42)
        self.assertEqual(metrics["algorithm"], "Random Forest")
        self.assertGreaterEqual(metrics["accuracy"], 0.75)
        self.assertGreaterEqual(metrics["roc_auc"], 0.80)

    def test_06_support_vector_machine(self):
        metrics, pipeline = alg_06.run_svm(c=1.0, kernel="rbf", random_state=42)
        self.assertEqual(metrics["algorithm"], "Support Vector Machine")
        self.assertGreaterEqual(metrics["accuracy"], 0.75)

    def test_07_naive_bayes(self):
        metrics, model = alg_07.run_naive_bayes(random_state=42)
        self.assertEqual(metrics["algorithm"], "Gaussian Naive Bayes")
        self.assertGreaterEqual(metrics["accuracy"], 0.70)

    def test_08_k_means_clustering(self):
        metrics, model = alg_08.run_kmeans(n_clusters=3, random_state=42)
        self.assertEqual(metrics["algorithm"], "K-Means Clustering")
        self.assertGreater(metrics["silhouette_score"], 0.40)
        self.assertLess(metrics["davies_bouldin_score"], 1.50)

    def test_09_principal_component_analysis(self):
        metrics, pca = alg_09.run_pca(n_components=2, random_state=42)
        self.assertEqual(metrics["algorithm"], "Principal Component Analysis")
        self.assertEqual(metrics["projected_components"], 2)
        self.assertGreater(metrics["cumulative_variance_explained"], 0.30)

    def test_10_gradient_boosting(self):
        metrics, model = alg_10.run_gradient_boosting(n_estimators=50, learning_rate=0.1, random_state=42)
        self.assertEqual(metrics["algorithm"], "Gradient Boosting")
        self.assertGreaterEqual(metrics["accuracy"], 0.75)
        self.assertGreaterEqual(metrics["roc_auc"], 0.80)


class TestDataLeakageAndReproducibility(unittest.TestCase):
    def test_reproducibility(self):
        metrics_1, _ = alg_01.run_linear_regression(random_state=42)
        metrics_2, _ = alg_01.run_linear_regression(random_state=42)
        self.assertEqual(metrics_1["r2_score"], metrics_2["r2_score"])

        metrics_rf1, _ = alg_05.run_random_forest(n_estimators=20, random_state=42)
        metrics_rf2, _ = alg_05.run_random_forest(n_estimators=20, random_state=42)
        self.assertEqual(metrics_rf1["accuracy"], metrics_rf2["accuracy"])

    def test_pipeline_leakage_isolation(self):
        # Verify that StandardScaler inside Pipeline fits exclusively on train set
        _, pipeline = alg_01.run_linear_regression(random_state=42)
        scaler = pipeline.named_steps["scaler"]
        self.assertEqual(scaler.n_samples_seen_, 400)  # 80% of 500 samples


if __name__ == "__main__":
    unittest.main()
