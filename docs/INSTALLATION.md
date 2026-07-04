# Installation

## Stable installation from PyPI

After the package is published:

```bash
pip install agraspqfs
```

## Installation from GitHub

```bash
pip install git+https://github.com/vagnerereno/A-GRASPQ-FS.git
```

## Development installation

```bash
git clone https://github.com/vagnerereno/A-GRASPQ-FS.git
cd A-GRASPQ-FS
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Optional dependencies

For XGBoost support:

```bash
pip install "agraspqfs[xgboost]"
```

or, in development mode:

```bash
pip install -e ".[xgboost,dev]"
```
