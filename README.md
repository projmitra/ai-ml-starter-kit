# AI/ML Starter Kit

An educational, leak-free starter kit providing clean scikit-learn implementations of 10 classic machine learning algorithms with automated synthetic data generation and reproducibility tests.

## Algorithms Included

1. **Linear Regression** (`algorithms/01_linear_regression.py`)
2. **Logistic Regression** (`algorithms/02_logistic_regression.py`)
3. **K-Nearest Neighbors** (`algorithms/03_k_nearest_neighbors.py`)
4. **Decision Tree** (`algorithms/04_decision_tree.py`)
5. **Random Forest** (`algorithms/05_random_forest.py`)
6. **Support Vector Machine** (`algorithms/06_support_vector_machine.py`)
7. **Gaussian Naive Bayes** (`algorithms/07_naive_bayes.py`)
8. **K-Means Clustering** (`algorithms/08_k_means_clustering.py`)
9. **Principal Component Analysis** (`algorithms/09_principal_component_analysis.py`)
10. **Gradient Boosting** (`algorithms/10_gradient_boosting.py`)

## Architecture & Design Principles

- **Data Leakage Isolation**: Preprocessing transformers (such as `StandardScaler`) are strictly encapsulated inside scikit-learn `Pipeline` objects and fitted exclusively on training splits.
- **Deterministic Synthetics**: Controlled synthetic dataset generators reside in `datasets/synthetic_generators.py` to allow fully reproducible testing without external network or dataset dependencies.
- **Contract Verification**: Every algorithm module exposes a callable execution function returning standard metric dictionaries and fitted estimators.

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-ml-starter-kit

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

Execute the automated test suite across all 16 unit, contract, and leakage tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Running an Algorithm

Run any algorithm directly:

```bash
python algorithms/01_linear_regression.py
```

## Repository Structure

```
ai-ml-starter-kit/
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
├── SECURITY.md
├── algorithms/
│   ├── 01_linear_regression.py
│   ├── 02_logistic_regression.py
│   ├── 03_k_nearest_neighbors.py
│   ├── 04_decision_tree.py
│   ├── 05_random_forest.py
│   ├── 06_support_vector_machine.py
│   ├── 07_naive_bayes.py
│   ├── 08_k_means_clustering.py
│   ├── 09_principal_component_analysis.py
│   └── 10_gradient_boosting.py
├── datasets/
│   └── synthetic_generators.py
└── tests/
    └── test_algorithms.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
