"""Command line interface for A-GRASPQ-FS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from . import __version__
from .selector import AGraspQFeatureSelector


def _make_estimator(name: str, random_state: int | None) -> Any:
    name = name.lower()
    if name == "nb":
        return GaussianNB()
    if name == "dt":
        return DecisionTreeClassifier(random_state=random_state)
    if name == "rf":
        return RandomForestClassifier(random_state=random_state, n_jobs=1)
    if name == "knn":
        return KNeighborsClassifier()
    if name == "svm":
        return SVC(random_state=random_state)
    if name == "linear_svc":
        return LinearSVC(random_state=random_state, dual=False, max_iter=5000)
    if name == "sgd":
        return SGDClassifier(random_state=random_state, max_iter=2000, tol=1e-3)
    if name == "logreg":
        return LogisticRegression(max_iter=5000, random_state=random_state)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("xgboost is not installed. Install agraspqfs[xgboost] or xgboost.") from exc
        return XGBClassifier(eval_metric="mlogloss", random_state=random_state, n_jobs=1)
    raise ValueError(f"Unknown estimator: {name}")


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _auto_preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale numeric columns and one-hot encode categorical columns for CLI use."""
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", _make_one_hot_encoder(), categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:  # pragma: no cover
        feature_names = [f"x{i}" for i in range(X_train_processed.shape[1])]

    return (
        pd.DataFrame(X_train_processed, columns=[str(c) for c in feature_names], index=X_train.index),
        pd.DataFrame(X_test_processed, columns=[str(c) for c in feature_names], index=X_test.index),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-GRASPQ-FS adaptive feature selector")
    parser.add_argument("--version", action="version", version=f"agraspqfs {__version__}")
    parser.add_argument("--csv", required=True, help="Input CSV file.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--sep", default=",", help="CSV separator. Default: ','.")
    parser.add_argument("--estimator", default="nb", choices=["nb", "dt", "rf", "knn", "svm", "linear_svc", "sgd", "logreg", "xgboost"])
    parser.add_argument("--scoring", default="f1_weighted", help="scikit-learn scoring name.")
    parser.add_argument("--preset", default="balanced", choices=["fast", "balanced", "robust", "paper"])
    parser.add_argument("--min-features", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--rcl-size", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--time-budget", type=float, default=None, help="Time budget in seconds.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30, help="External holdout fraction for reporting.")
    parser.add_argument("--output", default="results/agraspqfs_selection.json", help="JSON output path.")
    parser.add_argument("--history-csv", default=None, help="Optional path to save the evaluation history.")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--no-auto-preprocess", action="store_true", help="Disable automatic numeric scaling and categorical one-hot encoding.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path, sep=args.sep, skipinitialspace=True)
    if args.target not in df.columns:
        raise KeyError(f"Target column {args.target!r} not found in {csv_path}.")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y if y.nunique() > 1 and y.value_counts().min() >= 2 else None,
        random_state=args.random_state,
    )

    preprocessing_applied = False
    if not args.no_auto_preprocess:
        X_train, X_test = _auto_preprocess(X_train, X_test)
        preprocessing_applied = True

    estimator = _make_estimator(args.estimator, args.random_state)
    selector = AGraspQFeatureSelector(
        estimator=estimator,
        scoring=args.scoring,
        preset=args.preset,
        min_features=args.min_features,
        max_features=args.max_features,
        rcl_size=args.rcl_size,
        max_evaluations=args.max_evaluations,
        time_budget=args.time_budget,
        random_state=args.random_state,
        verbose=args.verbose,
    )
    selector.fit(X_train, y_train)

    final_model = _make_estimator(args.estimator, args.random_state)
    final_model.fit(selector.transform(X_train), y_train)
    scorer = get_scorer(args.scoring)
    test_score = float(scorer(final_model, selector.transform(X_test), y_test))

    result = {
        "package_version": __version__,
        "input_csv": str(csv_path),
        "target": args.target,
        "estimator": args.estimator,
        "scoring": args.scoring,
        "preset": args.preset,
        "preprocessing_applied": preprocessing_applied,
        "baseline_score": selector.baseline_score_,
        "best_internal_score": selector.best_score_,
        "test_score": test_score,
        "n_input_features": int(selector.n_features_in_),
        "n_selected_features": int(selector.n_features_selected_),
        "selected_features": selector.selected_features_,
        "selected_indices": selector.selected_indices_.tolist(),
        "n_evaluations": int(selector.n_evaluations_),
        "n_cache_hits": int(selector.n_cache_hits_),
        "elapsed_time": selector.elapsed_time_,
        "stop_reason": selector.stop_reason_,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.history_csv:
        history_path = Path(args.history_csv)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        selector.history_.to_csv(history_path, index=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
