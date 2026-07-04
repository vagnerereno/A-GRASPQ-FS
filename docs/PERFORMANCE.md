# Performance Guide

A-GRASPQ-FS is a wrapper feature selector. It evaluates candidate subsets by fitting a machine learning estimator, so it is expected to be more expensive than filter methods.

Use it as an offline optimizer when you want to avoid manually tuning the number of selected features.

## Recommended presets

| Preset | Use case | Typical behavior |
|---|---|---|
| `fast` | notebooks, CI, first trial | low budget and row sampling |
| `balanced` | practical experiments | moderate budget and internal CV |
| `robust` | slower experimental runs | larger budget and stronger validation |
| `paper` | research-style configuration | conservative settings inspired by the original prototype |

## Main controls

| Parameter | Effect |
|---|---|
| `max_evaluations` | hard limit on expensive subset evaluations |
| `time_budget` | maximum wall-clock time in seconds |
| `early_stopping_rounds` | stops when the search stops improving |
| `evaluation_sample_size` | evaluates exploratory candidates on a fraction of rows |
| `cv` | internal validation strategy; higher values improve robustness but increase cost |
| `n_jobs` | parallel evaluations; avoid nested parallelism with estimators that also use `n_jobs` |

## Practical advice

1. Start with `preset="fast"` and a small `max_features`.
2. Increase `max_features` only when selected subsets consistently hit the upper bound.
3. Use `evaluation_sample_size < 1.0` for large datasets.
4. Set the evaluator estimator's own `n_jobs=1` when using `AGraspQFeatureSelector(n_jobs=-1)`.
5. Always report `n_evaluations_`, `n_cache_hits_`, `elapsed_time_`, selected feature count, and the random seed.

## Why the overhead can be acceptable

The method is intended for the offline feature-selection stage. Once the subset is found, deployment uses only the selected features, reducing inference-time cost and simplifying downstream models.
