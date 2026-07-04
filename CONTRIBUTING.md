# Contributing

Thank you for your interest in A-GRASPQ-FS.

## Development setup

```bash
git clone https://github.com/vagnerereno/A-GRASPQ-FS.git
cd A-GRASPQ-FS
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Contribution guidelines

- Open an issue before large changes to discuss scope.
- Keep the public API compatible with scikit-learn conventions whenever possible.
- Add or update tests for new behavior.
- Document changes in `CHANGELOG.md`.
- Prefer deterministic examples by setting `random_state`.

## Code style

The project intentionally keeps dependencies lightweight. Use clear, typed Python code and avoid adding heavy dependencies unless they materially improve usability or performance.

## Reporting performance results

A-GRASPQ-FS is a stochastic wrapper method. When reporting results, include:

- dataset name and preprocessing;
- estimator and hyperparameters;
- scoring metric;
- selected feature count;
- selected features;
- random seed;
- number of evaluations;
- elapsed time;
- preset or full selector configuration.
