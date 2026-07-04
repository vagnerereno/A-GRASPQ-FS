# A-GRASPQ-FS

**Adaptive GRASPQ Feature Selection with self-tuning subset size**

A-GRASPQ-FS is a scikit-learn compatible feature selector based on a reactive GRASP metaheuristic. It jointly solves two decisions that are usually treated separately:

1. **how many features should be selected** for a given dataset and classifier; and
2. **which features should compose that subset**.

Instead of requiring a fixed value of `k`, A-GRASPQ-FS searches over a cardinality interval and adapts the probability of choosing each subset size according to the validation performance observed during the search.

The original command-line research prototype used in the ISCC 2026 paper is still available in `main.py`. Version 1.0 adds a production-oriented Python package in `src/agraspqfs/`, with a plug-and-play API for scikit-learn workflows.

## Associated paper

This repository accompanies the ISCC 2026 paper:

> Vagner E. Quincozes, Silvio E. Quincozes, Célio Albuquerque, Diego Passos, and Daniel Mossé. **Adaptive Feature Selection with Self-Tuning Subset Size for Intrusion Detection**. 31st IEEE Symposium on Computers and Communications (ISCC), Algarve, Portugal, 2026.

If you use this software in academic work, please cite the paper and/or the software release. Citation files are provided in [CITATION.cff](https://github.com/vagnerereno/A-GRASPQ-FS/blob/main/CITATION.cff) and [docs/CITATION.md](https://github.com/vagnerereno/A-GRASPQ-FS/blob/main/docs/CITATION.md).

## Highlights in v1.0

- scikit-learn compatible transformer: `fit`, `transform`, `fit_transform`, `get_support`, `get_feature_names_out`.
- Works with any scikit-learn compatible classifier as the subset evaluator.
- Joint optimization of subset size and feature composition.
- Presets for different budgets: `fast`, `balanced`, `robust`, and `paper`.
- In-memory cache to avoid repeated evaluations of the same subset.
- Explicit computational controls: `max_evaluations`, `time_budget`, and `early_stopping_rounds`.
- Exploratory evaluation on a row sample via `evaluation_sample_size`.
- Final full-data re-evaluation of elite candidate subsets.
- CLI command: `agraspqfs`.
- Unit tests, examples, GitHub Actions workflow, citation metadata, and release checklist.

## Installation

From the repository root:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

Optional XGBoost support:

```bash
pip install -e ".[xgboost]"
```

## Minimal usage

```python
from agraspqfs import AGraspQFeatureSelector
from sklearn.naive_bayes import GaussianNB

selector = AGraspQFeatureSelector(
    estimator=GaussianNB(),
    preset="fast",
    min_features=2,
    max_features=15,
    random_state=42,
)

X_selected = selector.fit_transform(X, y)

print(selector.selected_features_)
print(selector.n_features_selected_)
print(selector.best_score_)
```

## Pipeline usage

```python
from agraspqfs import AGraspQFeatureSelector
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("fs", AGraspQFeatureSelector(
        estimator=GaussianNB(),
        preset="balanced",
        min_features=2,
        max_features=20,
        random_state=42,
    )),
    ("clf", LogisticRegression(max_iter=2000)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

The estimator inside `AGraspQFeatureSelector` is used to evaluate feature subsets. The classifier after the selector in the pipeline can be the same model or a different one.

## CLI usage

```bash
agraspqfs   --csv data/hibrid_dataset_GOOSE_train.csv   --target class   --estimator nb   --preset fast   --min-features 2   --max-features 15   --output results/agraspqfs_selection.json   --history-csv results/agraspqfs_history.csv
```

The CLI prints and saves a JSON report containing selected features, internal validation score, external holdout score, number of evaluations, cache hits, and elapsed time.

## Presets

| Preset | Intended use | Behavior |
|---|---|---|
| `fast` | notebooks, smoke tests, quick exploration | fewer evaluations, holdout validation, row sampling |
| `balanced` | default practical use | moderate budget, 3-fold CV, partial row sampling |
| `robust` | slower experiments | larger budget, 5-fold CV, full exploratory data |
| `paper` | compatibility with the research-style workflow | conservative budget inspired by the original prototype |

Explicit parameters override preset values.

Example:

```python
selector = AGraspQFeatureSelector(
    preset="fast",
    max_evaluations=100,
    time_budget=60,
    early_stopping_rounds=30,
    random_state=42,
)
```

## Main parameters

| Parameter | Description |
|---|---|
| `estimator` | Classifier used to evaluate candidate subsets. Defaults to `GaussianNB`. |
| `scoring` | Any scikit-learn scoring name, e.g., `f1_weighted`, `accuracy`, `roc_auc_ovr`. |
| `min_features`, `max_features` | Minimum and maximum subset size. |
| `rcl_size` | Number of ranked candidate features considered by the metaheuristic. |
| `ranking_method` | `mutual_info`, `variance`, `random`, `none`, or callable. |
| `evaluation_sample_size` | Fraction of rows used in exploratory evaluations. |
| `cv` | Internal CV. Use `None` for holdout validation. |
| `max_evaluations` | Maximum number of non-cached subset evaluations. |
| `time_budget` | Time budget in seconds. |
| `cache` | Reuse repeated subset scores. |
| `refit` | If `True`, fit `estimator_` on the selected features after selection. |

## Main outputs

After `fit`, the selector exposes:

```python
selector.selected_features_      # names of selected features
selector.selected_indices_       # integer indices
selector.n_features_selected_    # selected subset size
selector.support_                # boolean mask
selector.best_score_             # best internal score among final candidates
selector.baseline_score_         # internal score with all features
selector.history_                # pandas DataFrame with evaluation trace
selector.size_weights_           # final adaptive cardinality weights
selector.n_evaluations_          # expensive evaluations actually computed
selector.n_cache_hits_           # repeated evaluations avoided by cache
selector.stop_reason_            # completed, budget_exhausted or early_stopping
```

## Examples

Run:

```bash
python examples/quickstart.py
python examples/pipeline_example.py
python examples/benchmark_small.py
```

A minimal notebook is available at [`examples/agraspqfs_demo.ipynb`](https://github.com/vagnerereno/A-GRASPQ-FS/blob/main/examples/agraspqfs_demo.ipynb).

## Legacy research prototype

The original scripts are still available:

```bash
python main.py -d ereninho -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 50
```

Baseline scripts are also preserved:

```bash
python baselines.py -d ereninho -m rfecv -a nb
```

For new users, prefer the package API or the `agraspqfs` CLI.

## Repository structure

```text
.
├── src/agraspqfs/              # Production package
│   ├── selector.py             # AGraspQFeatureSelector
│   ├── cli.py                  # agraspqfs command
│   └── _version.py
├── tests/                      # Unit tests
├── examples/                   # Quickstart, pipeline, benchmark and notebook examples
├── docs/                       # Documentation and release checklist
├── .github/workflows/          # CI and PyPI publishing workflows
├── main.py                     # Legacy research prototype
├── baselines.py                # Baseline methods
├── utils.py                    # Legacy utilities
├── priority_queue.py           # Legacy queue implementation
├── data/                       # Example/research datasets
├── results/                    # Runtime outputs, ignored except .gitkeep
├── CITATION.cff                # Citation metadata
├── LICENSE                     # MIT license
├── pyproject.toml              # Packaging metadata
└── README.md
```

## Notes on data preprocessing

A-GRASPQ-FS expects the feature matrix passed to `fit` to be numeric, as is common for scikit-learn estimators and mutual information ranking. For categorical features, use a preprocessing pipeline before the selector, for example `OneHotEncoder` or `ColumnTransformer`.

The CLI includes automatic preprocessing for CSV files: numeric columns are scaled and categorical columns are one-hot encoded.

## Performance note

A-GRASPQ-FS is a wrapper method and can be computationally expensive because it repeatedly evaluates machine learning models. Use `preset="fast"`, `max_evaluations`, `time_budget`, `early_stopping_rounds`, and `evaluation_sample_size` to control cost. See [`docs/PERFORMANCE.md`](https://github.com/vagnerereno/A-GRASPQ-FS/blob/main/docs/PERFORMANCE.md).

## License

This project is released under the MIT License. See [`LICENSE`](https://github.com/vagnerereno/A-GRASPQ-FS/blob/main/LICENSE).

The license permits reuse, modification and distribution. For academic use, please cite the associated ISCC 2026 paper and/or the software release as described in [CITATION.cff](https://github.com/vagnerereno/A-GRASPQ-FS/blob/main/CITATION.cff).

## Citation

```bibtex
@inproceedings{quincozes2026agraspqfs,
  title     = {Adaptive Feature Selection with Self-Tuning Subset Size for Intrusion Detection},
  author    = {Quincozes, Vagner E. and Quincozes, Silvio E. and Albuquerque, Célio and Passos, Diego and Mossé, Daniel},
  booktitle = {Proceedings of 31st IEEE Symposium on Computers and Communications (ISCC)},
  address   = {Algarve, Portugal},
  year      = {2026}
}
```