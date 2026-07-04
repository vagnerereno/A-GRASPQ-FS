"""scikit-learn compatible implementation of A-GRASPQ-FS.

A-GRASPQ-FS is a wrapper feature selector that jointly searches for the subset
cardinality and the feature composition. It is intentionally model-agnostic: any
scikit-learn compatible estimator can be used as the evaluator.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from heapq import heappop, heappush, heapreplace
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import check_scoring
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted

LOGGER = logging.getLogger(__name__)

PresetName = Literal["fast", "balanced", "robust", "paper"]
RankingMethod = Literal["mutual_info", "variance", "random", "none"]


@dataclass(frozen=True)
class _ResolvedParams:
    constructive_iterations: int
    local_iterations: int
    priority_queue_size: int
    evaluation_sample_size: float
    cv: int | Any | None
    validation_fraction: float
    rcl_size: int | None
    min_features: int
    max_features: int | None
    alpha: float
    update_interval: int
    max_evaluations: int | None
    early_stopping_rounds: int | None
    time_budget: float | None
    n_jobs: int | None


_PRESETS: dict[str, dict[str, Any]] = {
    # Good for notebooks, CI tests, and quick exploration.
    "fast": {
        "constructive_iterations": 20,
        "local_iterations": 35,
        "priority_queue_size": 5,
        "evaluation_sample_size": 0.35,
        "cv": None,
        "validation_fraction": 0.30,
        "max_evaluations": 250,
        "early_stopping_rounds": 60,
        "n_jobs": None,
    },
    # Balanced default for typical tabular experiments.
    "balanced": {
        "constructive_iterations": 60,
        "local_iterations": 100,
        "priority_queue_size": 10,
        "evaluation_sample_size": 0.70,
        "cv": 3,
        "validation_fraction": 0.30,
        "max_evaluations": 1500,
        "early_stopping_rounds": 200,
        "n_jobs": None,
    },
    # More robust, more expensive.
    "robust": {
        "constructive_iterations": 120,
        "local_iterations": 250,
        "priority_queue_size": 20,
        "evaluation_sample_size": 1.00,
        "cv": 5,
        "validation_fraction": 0.30,
        "max_evaluations": 5000,
        "early_stopping_rounds": 500,
        "n_jobs": None,
    },
    # Conservative preset close to the original research workflow. The exact
    # legacy CLI is still kept in main.py; this preset only mirrors its intent.
    "paper": {
        "constructive_iterations": 50,
        "local_iterations": 50,
        "priority_queue_size": 10,
        "evaluation_sample_size": 1.00,
        "cv": None,
        "validation_fraction": 0.30,
        "max_evaluations": None,
        "early_stopping_rounds": None,
        "n_jobs": None,
    },
}


class AGraspQFeatureSelector(BaseEstimator, TransformerMixin):
    """Adaptive GRASP-based wrapper feature selector.

    The selector searches simultaneously for:

    1. the best subset size, i.e., the selected cardinality; and
    2. the best feature composition for that cardinality.

    Parameters
    ----------
    estimator : scikit-learn estimator, default=None
        Model used to evaluate feature subsets. If ``None``, ``GaussianNB`` is
        used. The estimator is cloned before every evaluation.
    scoring : str or callable, default="f1_weighted"
        Any scoring accepted by scikit-learn, for example ``"accuracy"``,
        ``"f1_weighted"`` or a custom scorer.
    preset : {"fast", "balanced", "robust", "paper"}, default="balanced"
        Configuration preset. Explicitly provided parameters override preset
        values.
    min_features, max_features : int, optional
        Lower and upper cardinality bounds explored by the algorithm. If
        ``max_features`` is ``None``, all features in the restricted candidate
        list may be selected.
    rcl_size : int, optional
        Size of the restricted candidate list. By default, all available
        features are ranked and considered.
    alpha : float, default=0.7
        Semi-greedy construction parameter in (0, 1]. Smaller values make the
        construction greedier; larger values make it more random.
    constructive_iterations : int, optional
        Number of reactive construction iterations.
    local_iterations : int, optional
        Maximum number of local-search moves per elite solution.
    priority_queue_size : int, optional
        Number of elite constructed solutions sent to local search.
    update_interval : int, default=10
        Interval used to recalibrate size probabilities from observed scores.
    evaluation_sample_size : float, optional
        Fraction of rows used during exploratory evaluations. Final elite
        candidates are re-evaluated with the full data available to ``fit``.
    cv : int, cross-validator or None, optional
        Internal validation strategy. ``None`` uses a holdout split. Integers
        use ``StratifiedKFold``.
    validation_fraction : float, optional
        Holdout fraction used when ``cv=None``.
    ranking_method : {"mutual_info", "variance", "random", "none"} or callable,
        default="mutual_info"
        Method used to rank features before forming the RCL. A callable must
        receive ``(X, y)`` and return one score per feature.
    max_evaluations : int, optional
        Hard budget on the number of non-cached subset evaluations.
    time_budget : float, optional
        Time budget in seconds.
    early_stopping_rounds : int, optional
        Stop after this many non-improving evaluations.
    cache : bool, default=True
        Reuse scores for already evaluated subsets.
    n_jobs : int, optional
        Number of jobs used by scikit-learn cross-validation.
    random_state : int, optional
        Random seed.
    refit : bool, default=False
        If ``True``, fit ``estimator_`` on the full input using selected
        features after feature selection.
    verbose : int, default=0
        Verbosity level. ``0`` is silent; larger values emit progress logs.

    Attributes
    ----------
    support_ : ndarray of shape (n_features,)
        Boolean mask of selected features.
    selected_indices_ : ndarray
        Integer indices of selected features.
    selected_features_ : list[str]
        Names of selected features.
    n_features_selected_ : int
        Number of selected features.
    best_score_ : float
        Best internal validation score found after full-data re-evaluation of
        elite candidates.
    baseline_score_ : float
        Score obtained using all features under the same internal validation
        protocol.
    history_ : pandas.DataFrame
        Evaluation history, including phase, subset size, score and selected
        features.
    n_evaluations_ : int
        Number of non-cached subset evaluations.
    n_cache_hits_ : int
        Number of cached score lookups.
    size_weights_ : dict[int, float]
        Final adaptive weights associated with each cardinality.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        scoring: str | Callable[..., float] = "f1_weighted",
        preset: PresetName | None = "balanced",
        min_features: int | None = None,
        max_features: int | None = None,
        rcl_size: int | None = None,
        alpha: float = 0.7,
        constructive_iterations: int | None = None,
        local_iterations: int | None = None,
        priority_queue_size: int | None = None,
        update_interval: int = 10,
        evaluation_sample_size: float | None = None,
        cv: int | Any | None = None,
        validation_fraction: float | None = None,
        ranking_method: RankingMethod | Callable[[Any, Any], Sequence[float]] = "mutual_info",
        max_evaluations: int | None = None,
        time_budget: float | None = None,
        early_stopping_rounds: int | None = None,
        cache: bool = True,
        n_jobs: int | None = None,
        random_state: int | None = None,
        refit: bool = False,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.scoring = scoring
        self.preset = preset
        self.min_features = min_features
        self.max_features = max_features
        self.rcl_size = rcl_size
        self.alpha = alpha
        self.constructive_iterations = constructive_iterations
        self.local_iterations = local_iterations
        self.priority_queue_size = priority_queue_size
        self.update_interval = update_interval
        self.evaluation_sample_size = evaluation_sample_size
        self.cv = cv
        self.validation_fraction = validation_fraction
        self.ranking_method = ranking_method
        self.max_evaluations = max_evaluations
        self.time_budget = time_budget
        self.early_stopping_rounds = early_stopping_rounds
        self.cache = cache
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.refit = refit
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> "AGraspQFeatureSelector":
        """Run adaptive feature selection."""
        X_prepared, y_array = self._prepare_input(X, y)
        self._is_pandas_input_ = isinstance(X, pd.DataFrame)
        self.feature_names_in_ = np.asarray(self._extract_feature_names(X), dtype=object)
        self.n_features_in_ = X_prepared.shape[1]

        if self.n_features_in_ == 0:
            raise ValueError("A-GRASPQ-FS requires at least one feature.")

        self._params_ = self._resolve_params(self.n_features_in_)
        self._rng_ = np.random.default_rng(self.random_state)
        self._cache_: dict[tuple[tuple[int, ...], float], float] = {}
        self.n_evaluations_ = 0
        self.n_cache_hits_ = 0
        self._no_improvement_evals_ = 0
        self._global_best_score_ = -np.inf
        self._start_time_ = time.perf_counter()
        self._history_records_: list[dict[str, Any]] = []
        self._stop_reason_ = "completed"

        self._X_full_ = X_prepared
        self._y_full_ = y_array
        self._build_validation_data(X_prepared, y_array)

        self._scorer_ = check_scoring(self._base_estimator(), scoring=self.scoring)
        self.ranking_scores_ = self._rank_features(X_prepared, y_array)
        ranked_indices = list(np.argsort(self.ranking_scores_)[::-1])
        if self.ranking_method == "none":
            ranked_indices = list(range(self.n_features_in_))
        self.ranking_ = np.asarray(ranked_indices, dtype=int)

        rcl_size = self._params_.rcl_size or self.n_features_in_
        rcl_size = max(1, min(rcl_size, self.n_features_in_))
        self.rcl_indices_ = np.asarray(self.ranking_[:rcl_size], dtype=int)

        self.baseline_score_ = self._evaluate_subset(
            tuple(range(self.n_features_in_)),
            phase="baseline",
            sample_ratio=1.0,
            record=True,
        )

        elite_heap, size_weights, size_history = self._constructive_phase()
        self.size_weights_ = dict(size_weights)
        self.size_history_ = {int(k): list(v) for k, v in size_history.items()}

        candidates = self._local_search_phase(elite_heap)
        if not candidates:
            # Fallback for extremely small budgets.
            candidates = [(self.baseline_score_, tuple(range(self.n_features_in_)))]
            self._stop_reason_ = "budget_exhausted_before_candidate_selection"

        # Full-data robust re-evaluation of final elite candidates.
        final_candidates: list[tuple[float, tuple[int, ...]]] = []
        seen_final: set[tuple[int, ...]] = set()
        for _, subset in sorted(candidates, reverse=True):
            subset = tuple(sorted(set(subset)))
            if subset in seen_final:
                continue
            seen_final.add(subset)
            if self._should_stop():
                break
            score = self._evaluate_subset(
                subset,
                phase="final_full_evaluation",
                sample_ratio=1.0,
                record=True,
            )
            final_candidates.append((score, subset))

        if final_candidates:
            best_score, best_subset = max(final_candidates, key=lambda item: (item[0], -len(item[1])))
        else:
            # If the full-evaluation budget was exhausted, keep the best sampled
            # solution rather than failing after doing useful work.
            best_score, best_subset = max(candidates, key=lambda item: (item[0], -len(item[1])))

        self.best_score_ = float(best_score)
        self.selected_indices_ = np.asarray(best_subset, dtype=int)
        self.selected_indices_.sort()
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_indices_] = True
        self.selected_features_ = [str(self.feature_names_in_[i]) for i in self.selected_indices_]
        self.n_features_selected_ = int(len(self.selected_indices_))
        self.stop_reason_ = self._stop_reason_
        self.elapsed_time_ = float(time.perf_counter() - self._start_time_)
        self.history_ = pd.DataFrame(self._history_records_)

        if self.refit:
            self.estimator_ = clone(self._base_estimator())
            self.estimator_.fit(self._subset_columns(self._X_full_, self.selected_indices_), self._y_full_)

        # Drop bulky transient arrays so the fitted selector is serializable and
        # light enough for joblib/pickle artifacts.
        for attr in ["_X_full_", "_y_full_", "_X_train_eval_", "_X_valid_eval_", "_y_train_eval_", "_y_valid_eval_"]:
            if hasattr(self, attr):
                delattr(self, attr)

        return self

    def transform(self, X: Any) -> Any:
        """Reduce ``X`` to the selected features."""
        check_is_fitted(self, "support_")
        if getattr(self, "_is_pandas_input_", False) and isinstance(X, pd.DataFrame):
            return X.loc[:, self.selected_features_]
        X_array = np.asarray(X)
        return X_array[:, self.selected_indices_]

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Restore selected data to the original number of columns with zeros."""
        check_is_fitted(self, "support_")
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")
        output = np.zeros((X_array.shape[0], self.n_features_in_), dtype=X_array.dtype)
        output[:, self.selected_indices_] = X_array
        return output

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return the selected feature mask or integer indices."""
        check_is_fitted(self, "support_")
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return selected feature names following scikit-learn conventions."""
        check_is_fitted(self, "support_")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if len(names) != self.n_features_in_:
                raise ValueError("input_features must have length n_features_in_.")
            return names[self.selected_indices_]
        return np.asarray(self.selected_features_, dtype=object)

    def plot_history(self, path: str | None = None):
        """Plot score evolution over evaluated subsets.

        Parameters
        ----------
        path : str, optional
            If provided, save the plot to this path.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot.
        """
        check_is_fitted(self, "history_")
        if self.history_.empty:
            raise ValueError("No history was recorded.")
        import matplotlib.pyplot as plt

        ax = self.history_.reset_index().plot(
            x="index",
            y="score",
            kind="line",
            legend=False,
            xlabel="Evaluation",
            ylabel="Score",
            title="A-GRASPQ-FS evaluation history",
        )
        if path is not None:
            ax.figure.tight_layout()
            ax.figure.savefig(path, dpi=300)
        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_params(self, n_features: int) -> _ResolvedParams:
        if self.preset is None:
            preset_values = _PRESETS["balanced"].copy()
        else:
            if self.preset not in _PRESETS:
                raise ValueError(f"Unknown preset {self.preset!r}. Choose from {tuple(_PRESETS)}.")
            preset_values = _PRESETS[self.preset].copy()

        def choose(name: str, current: Any) -> Any:
            return preset_values[name] if current is None else current

        min_features = 1 if self.min_features is None else int(self.min_features)
        max_features = n_features if self.max_features is None else int(self.max_features)
        rcl_size = n_features if self.rcl_size is None else int(self.rcl_size)

        if min_features < 1:
            raise ValueError("min_features must be >= 1.")
        if max_features < min_features:
            raise ValueError("max_features must be >= min_features.")
        if rcl_size < min_features:
            raise ValueError("rcl_size must be >= min_features.")
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("alpha must be in the interval (0, 1].")
        if self.update_interval < 1:
            raise ValueError("update_interval must be >= 1.")

        evaluation_sample_size = float(choose("evaluation_sample_size", self.evaluation_sample_size))
        if evaluation_sample_size <= 0 or evaluation_sample_size > 1:
            raise ValueError("evaluation_sample_size must be in the interval (0, 1].")

        validation_fraction = float(choose("validation_fraction", self.validation_fraction))
        if validation_fraction <= 0 or validation_fraction >= 1:
            raise ValueError("validation_fraction must be in the interval (0, 1).")

        return _ResolvedParams(
            constructive_iterations=int(choose("constructive_iterations", self.constructive_iterations)),
            local_iterations=int(choose("local_iterations", self.local_iterations)),
            priority_queue_size=int(choose("priority_queue_size", self.priority_queue_size)),
            evaluation_sample_size=evaluation_sample_size,
            cv=choose("cv", self.cv),
            validation_fraction=validation_fraction,
            rcl_size=min(rcl_size, n_features),
            min_features=min(min_features, n_features),
            max_features=min(max_features, n_features),
            alpha=float(self.alpha),
            update_interval=int(self.update_interval),
            max_evaluations=choose("max_evaluations", self.max_evaluations),
            early_stopping_rounds=choose("early_stopping_rounds", self.early_stopping_rounds),
            time_budget=self.time_budget,
            n_jobs=choose("n_jobs", self.n_jobs),
        )

    def _base_estimator(self):
        return GaussianNB() if self.estimator is None else self.estimator

    def _prepare_input(self, X: Any, y: Any) -> tuple[Any, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X_prepared = X.copy()
        else:
            X_prepared = np.asarray(X)
        if getattr(X_prepared, "ndim", 0) != 2:
            raise ValueError("X must be a 2D array or pandas DataFrame.")
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            y_array = y_array.ravel()
        if X_prepared.shape[0] != y_array.shape[0]:
            raise ValueError("X and y have inconsistent numbers of samples.")
        return X_prepared, y_array

    def _extract_feature_names(self, X: Any) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return [str(c) for c in X.columns]
        n_features = np.asarray(X).shape[1]
        return [f"x{i}" for i in range(n_features)]

    def _rank_features(self, X: Any, y: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X)
        method = self.ranking_method
        if callable(method):
            scores = np.asarray(method(X, y), dtype=float)
        elif method == "mutual_info":
            try:
                scores = mutual_info_classif(X_array, y, random_state=self.random_state)
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ValueError(
                    "ranking_method='mutual_info' requires numeric tabular features. "
                    "Use a preprocessing pipeline or set ranking_method='variance'/'random'."
                ) from exc
        elif method == "variance":
            scores = np.nanvar(X_array.astype(float), axis=0)
        elif method == "random":
            scores = self._rng_.random(X_array.shape[1])
        elif method == "none":
            # Scores arranged so original order is preserved after argsort below.
            scores = np.arange(X_array.shape[1], 0, -1, dtype=float)
        else:
            raise ValueError("ranking_method must be 'mutual_info', 'variance', 'random', 'none' or callable.")
        if scores.shape[0] != X_array.shape[1]:
            raise ValueError("ranking_method must return one score per feature.")
        return np.nan_to_num(scores, nan=-np.inf)

    def _build_validation_data(self, X: Any, y: np.ndarray) -> None:
        params = self._params_
        if params.cv is not None:
            self._X_train_eval_ = X
            self._y_train_eval_ = y
            return

        stratify = y if self._can_stratify(y) else None
        try:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=params.validation_fraction,
                stratify=stratify,
                random_state=self.random_state,
            )
        except ValueError:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=params.validation_fraction,
                random_state=self.random_state,
            )
        self._X_train_eval_ = X_train
        self._X_valid_eval_ = X_valid
        self._y_train_eval_ = y_train
        self._y_valid_eval_ = y_valid

    @staticmethod
    def _can_stratify(y: np.ndarray) -> bool:
        _, counts = np.unique(y, return_counts=True)
        return len(counts) > 1 and np.min(counts) >= 2

    def _sample_rows(self, X: Any, y: np.ndarray, ratio: float) -> tuple[Any, np.ndarray]:
        if ratio >= 1.0 or X.shape[0] < 10:
            return X, y
        n_samples = max(2, int(math.ceil(X.shape[0] * ratio)))
        n_samples = min(n_samples, X.shape[0])
        stratify = y if self._can_stratify(y) and n_samples >= len(np.unique(y)) * 2 else None
        try:
            X_sample, _, y_sample, _ = train_test_split(
                X,
                y,
                train_size=n_samples,
                stratify=stratify,
                random_state=self.random_state,
            )
            return X_sample, y_sample
        except ValueError:
            indices = self._rng_.choice(X.shape[0], size=n_samples, replace=False)
            if isinstance(X, pd.DataFrame):
                return X.iloc[indices], y[indices]
            return X[indices], y[indices]

    def _subset_columns(self, X: Any, subset: Sequence[int]) -> Any:
        subset_array = np.asarray(subset, dtype=int)
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, subset_array]
        return np.asarray(X)[:, subset_array]

    def _evaluate_subset(
        self,
        subset: Sequence[int],
        *,
        phase: str,
        sample_ratio: float,
        record: bool,
    ) -> float:
        subset_tuple = tuple(sorted(set(int(i) for i in subset)))
        if not subset_tuple:
            return -np.inf
        cache_key = (subset_tuple, round(float(sample_ratio), 6))
        if self.cache and cache_key in self._cache_:
            self.n_cache_hits_ += 1
            score = self._cache_[cache_key]
            if record:
                self._record_history(phase, subset_tuple, score, sample_ratio, cached=True)
            return score

        if self._budget_exhausted():
            self._stop_reason_ = "budget_exhausted"
            return -np.inf

        estimator = clone(self._base_estimator())
        params = self._params_

        if params.cv is not None:
            X_eval, y_eval = self._sample_rows(self._X_train_eval_, self._y_train_eval_, sample_ratio)
            X_subset = self._subset_columns(X_eval, subset_tuple)
            cv = params.cv
            if isinstance(cv, int):
                n_splits = int(cv)
                if self._can_stratify(y_eval):
                    _, counts = np.unique(y_eval, return_counts=True)
                    n_splits = min(n_splits, int(np.min(counts)))
                n_splits = max(2, n_splits)
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                estimator,
                X_subset,
                y_eval,
                scoring=self.scoring,
                cv=cv,
                n_jobs=params.n_jobs,
                error_score="raise",
            )
            score = float(np.mean(scores))
        else:
            X_train, y_train = self._sample_rows(self._X_train_eval_, self._y_train_eval_, sample_ratio)
            X_train_subset = self._subset_columns(X_train, subset_tuple)
            X_valid_subset = self._subset_columns(self._X_valid_eval_, subset_tuple)
            estimator.fit(X_train_subset, y_train)
            score = float(self._scorer_(estimator, X_valid_subset, self._y_valid_eval_))

        self.n_evaluations_ += 1
        if self.cache:
            self._cache_[cache_key] = score

        improved = score > self._global_best_score_
        if improved:
            self._global_best_score_ = score
            self._no_improvement_evals_ = 0
        else:
            self._no_improvement_evals_ += 1

        if record:
            self._record_history(phase, subset_tuple, score, sample_ratio, cached=False)

        if self.verbose:
            LOGGER.info("%s | score=%.5f | n_features=%d", phase, score, len(subset_tuple))
        return score

    def _record_history(self, phase: str, subset: tuple[int, ...], score: float, sample_ratio: float, cached: bool) -> None:
        self._history_records_.append(
            {
                "evaluation": len(self._history_records_) + 1,
                "phase": phase,
                "score": float(score),
                "n_features": int(len(subset)),
                "indices": list(subset),
                "features": [str(self.feature_names_in_[i]) for i in subset],
                "sample_ratio": float(sample_ratio),
                "cached": bool(cached),
                "elapsed_time": float(time.perf_counter() - self._start_time_),
            }
        )

    def _budget_exhausted(self) -> bool:
        params = self._params_
        if params.max_evaluations is not None and self.n_evaluations_ >= params.max_evaluations:
            return True
        if params.time_budget is not None and (time.perf_counter() - self._start_time_) >= params.time_budget:
            return True
        return False

    def _should_stop(self) -> bool:
        params = self._params_
        if self._budget_exhausted():
            self._stop_reason_ = "budget_exhausted"
            return True
        if params.early_stopping_rounds is not None and self._no_improvement_evals_ >= params.early_stopping_rounds:
            self._stop_reason_ = "early_stopping"
            return True
        return False

    def _constructive_phase(self) -> tuple[list[tuple[float, tuple[int, ...]]], dict[int, float], dict[int, list[float]]]:
        params = self._params_
        min_k = params.min_features
        max_k = min(params.max_features or self.n_features_in_, len(self.rcl_indices_))
        possible_sizes = list(range(min_k, max_k + 1))
        size_weights = {k: 1.0 for k in possible_sizes}
        size_history = {k: [] for k in possible_sizes}
        elite_heap: list[tuple[float, tuple[int, ...]]] = []
        seen_initial: set[tuple[int, ...]] = set()

        for iteration in range(params.constructive_iterations):
            if self._should_stop():
                break
            k = self._weighted_choice(possible_sizes, [size_weights[s] for s in possible_sizes])
            subset = self._construct_solution(k)
            attempts = 0
            while subset in seen_initial and attempts < 50:
                subset = self._construct_solution(k)
                attempts += 1
            seen_initial.add(subset)

            score = self._evaluate_subset(
                subset,
                phase="construction",
                sample_ratio=params.evaluation_sample_size,
                record=True,
            )
            if np.isfinite(score):
                size_history[len(subset)].append(score)
                self._push_elite(elite_heap, score, subset, params.priority_queue_size)

            if (iteration + 1) % params.update_interval == 0:
                self._update_size_weights(size_weights, size_history)

        return elite_heap, size_weights, size_history

    def _weighted_choice(self, values: Sequence[int], weights: Sequence[float]) -> int:
        weights_array = np.asarray(weights, dtype=float)
        weights_array = np.maximum(weights_array, 1e-12)
        probabilities = weights_array / weights_array.sum()
        return int(self._rng_.choice(np.asarray(values), p=probabilities))

    def _construct_solution(self, k: int) -> tuple[int, ...]:
        pool = list(self.rcl_indices_)
        selected: list[int] = []
        while len(selected) < k and pool:
            limit = max(1, int(math.ceil(len(pool) * self._params_.alpha)))
            candidate_position = int(self._rng_.integers(0, limit))
            selected_feature = pool.pop(candidate_position)
            selected.append(int(selected_feature))
        return tuple(sorted(selected))

    @staticmethod
    def _push_elite(heap: list[tuple[float, tuple[int, ...]]], score: float, subset: tuple[int, ...], max_size: int) -> None:
        item = (float(score), tuple(subset))
        if max_size <= 0:
            return
        if len(heap) < max_size:
            heappush(heap, item)
        elif score > heap[0][0]:
            heapreplace(heap, item)

    @staticmethod
    def _update_size_weights(size_weights: dict[int, float], size_history: dict[int, list[float]]) -> None:
        averages = {k: (float(np.mean(v)) if v else 0.0) for k, v in size_history.items()}
        max_average = max(averages.values()) if averages else 0.0
        smooth_factor = 0.5
        for k in size_weights:
            normalized = averages[k] / max_average if max_average > 0 else 0.0
            size_weights[k] = normalized + smooth_factor

    def _local_search_phase(self, elite_heap: list[tuple[float, tuple[int, ...]]]) -> list[tuple[float, tuple[int, ...]]]:
        params = self._params_
        if not elite_heap:
            return []
        candidates: list[tuple[float, tuple[int, ...]]] = []
        elite_sorted = sorted(elite_heap, reverse=True)

        for initial_score, initial_subset in elite_sorted:
            if self._should_stop():
                break
            current_subset = tuple(sorted(initial_subset))
            current_score = float(initial_score)
            best_subset = current_subset
            best_score = current_score
            seen = {current_subset}

            for _ in range(params.local_iterations):
                if self._should_stop():
                    break
                moves = list(self._neighborhood_moves(best_subset))
                if not moves:
                    break
                self._rng_.shuffle(moves)
                improved = False
                for move in moves:
                    if self._should_stop():
                        break
                    neighbor = self._apply_move(best_subset, move)
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    score = self._evaluate_subset(
                        neighbor,
                        phase="local_search",
                        sample_ratio=params.evaluation_sample_size,
                        record=True,
                    )
                    if score > best_score:
                        best_score = score
                        best_subset = neighbor
                        improved = True
                        break  # first-improvement VNS-style strategy
                if not improved:
                    break
            candidates.append((best_score, best_subset))

        return candidates

    def _neighborhood_moves(self, subset: tuple[int, ...]) -> Iterable[tuple[str, int, int | None]]:
        current = set(subset)
        rcl = set(int(i) for i in self.rcl_indices_)
        possible_adds = sorted(rcl - current)
        possible_removes = sorted(current)
        params = self._params_

        if len(current) < params.max_features:
            for feature in possible_adds:
                yield ("add", feature, None)
        if len(current) > params.min_features:
            for feature in possible_removes:
                yield ("remove", feature, None)
        if possible_adds and possible_removes:
            for out_feature in possible_removes:
                for in_feature in possible_adds:
                    yield ("swap", out_feature, in_feature)

    @staticmethod
    def _apply_move(subset: tuple[int, ...], move: tuple[str, int, int | None]) -> tuple[int, ...]:
        operation, first, second = move
        values = set(subset)
        if operation == "add":
            values.add(first)
        elif operation == "remove":
            values.discard(first)
        elif operation == "swap":
            values.discard(first)
            if second is not None:
                values.add(second)
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown move: {operation}")
        return tuple(sorted(values))
