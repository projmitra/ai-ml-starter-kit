# Contributing

Thank you for your interest in contributing to this educational ML starter kit.

## How to Contribute

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Implement your changes without modifying existing algorithm contracts.
4. Ensure all 16 automated tests continue to pass: `python -m unittest discover -s tests -p "test_*.py"`.
5. Submit a pull request with a clear description.

## Standards

- Each algorithm must expose a named callable function (e.g., `run_linear_regression`).
- Functions must return a `(metrics_dict, estimator)` tuple.
- No real-world datasets — use `datasets/synthetic_generators.py` exclusively.
- No secrets, credentials, or personal data may be introduced.
- Code must be compatible with Python 3.10+ and the dependency versions in `requirements.txt`.

## Scope of This Repository

This repository is strictly educational. It is not intended for production use. Commercial IP from other projects must not be introduced.
