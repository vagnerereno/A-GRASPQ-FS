# A-GRASPQ-FS API

## `AGraspQFeatureSelector`

`AGraspQFeatureSelector` is a scikit-learn compatible transformer for adaptive feature selection. It optimizes feature subset cardinality and feature composition jointly through a reactive GRASP construction phase and a local search phase with `ADD`, `REMOVE`, and `SWAP` moves.

### Important parameters

| Parameter | Meaning |
|---|---|
| `estimator` | scikit-learn compatible model used to evaluate candidate subsets. |
| `scoring` | scikit-learn scoring name or custom scorer. |
| `preset` | `fast`, `balanced`, `robust`, or `paper`. |
| `min_features`, `max_features` | Cardinality search interval. |
| `rcl_size` | Number of ranked features available to the construction/local-search phases. |
| `evaluation_sample_size` | Fraction of rows used during exploratory evaluations. |
| `max_evaluations` | Hard cap on expensive subset evaluations. |
| `time_budget` | Optional time limit in seconds. |
| `early_stopping_rounds` | Stops after several non-improving evaluations. |

### Main attributes after `fit`

| Attribute | Meaning |
|---|---|
| `selected_features_` | Selected feature names. |
| `selected_indices_` | Selected feature integer indices. |
| `support_` | Boolean support mask. |
| `n_features_selected_` | Number of selected features. |
| `best_score_` | Best internal validation score among final candidates. |
| `baseline_score_` | Internal validation score using all features. |
| `history_` | DataFrame with evaluated subsets, scores and phases. |
| `size_weights_` | Final adaptive cardinality weights. |
