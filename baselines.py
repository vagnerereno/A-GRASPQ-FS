import argparse
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def get_classifier(algo_name):
    if algo_name == 'knn':
        return KNeighborsClassifier(n_jobs=-1)
    elif algo_name == 'nb':
        return GaussianNB(var_smoothing=1e-9)
    elif algo_name == 'dt':
        return DecisionTreeClassifier(random_state=42)
    elif algo_name == 'linear_svc':
        return LinearSVC(max_iter=1000, random_state=42, dual=False)
    elif algo_name == 'rf':
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Algorithm '{algo_name}' not supported.")


def get_selector_estimator(target_algo_name):
    """
    Returns the estimator to be used WITHIN the RFE/RFECV.
    Required because RFE requires 'feature_importances_' or 'coef_'.
    """
    # Algorithms that work natively with RFE
    if target_algo_name in ['dt', 'linear_svc', 'rf']:
        return get_classifier(target_algo_name)
    else:
        # KNN and NB do not have feature weights. We use Decision Tree as a substitute ONLY for selection.
        logging.warning(
            f"Algorithm '{target_algo_name}' does not expose feature weights. Using Decision Tree as surrogate for RFE/RFECV selection phase.")
        return DecisionTreeClassifier(random_state=42)


def run_baseline(args):
    start_global = time.time()

    X, y, feature_names = utils.load_data(args.dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test)

    feature_names = X_train.columns.tolist()
    logging.info(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    target_clf = get_classifier(args.algorithm)
    selector = None

    logging.info(f"--- Running Baseline: {args.method.upper()} with {args.algorithm.upper()} ---")

    selection_start = time.time()

    if args.method in ['sfs_forward', 'sfs_backward']:
        direction = 'backward' if args.method == 'sfs_backward' else 'forward'

        # SFS: Uses its own classifier (target_clf) because it is based on CV score, not weights.
        # If args.n_features is None, uses 'auto' (selects half or via tool)
        n_to_sel = args.n_features if args.n_features else 'auto'

        logging.info(f"Starting SFS ({direction.capitalize()}). Target features: {n_to_sel}")
        selector = SequentialFeatureSelector(
            target_clf,
            n_features_to_select=n_to_sel,
            direction=direction,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        selector.fit(X_train, y_train)

    elif args.method == 'rfe':
        # RFE: Requires feature_importances_. Use surrogate if necessary.
        estimator = get_selector_estimator(args.algorithm)

        # If n_features is not specified, it assumes half the features (RFE default) or 10.
        n_to_sel = args.n_features if args.n_features else max(1, len(feature_names) // 2)

        logging.info(f"Starting RFE. Target features: {n_to_sel}")
        selector = RFE(estimator=estimator, n_features_to_select=n_to_sel, step=1)
        selector.fit(X_train, y_train)

    elif args.method == 'rfecv':
        # RFECV: Decides the optimal number on its own via Cross-Validation.
        estimator = get_selector_estimator(args.algorithm)

        logging.info(f"Starting RFECV (Automatic size selection).")
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            min_features_to_select=1
        )
        selector.fit(X_train, y_train)

    selection_time = time.time() - selection_start

    mask = selector.get_support()
    selected_indices = np.where(mask)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    logging.info(f"Selection Phase Completed in {selection_time:.2f}s")
    logging.info(f"Selected {len(selected_features)} features.")

    X_train_reduced = selector.transform(X_train)
    X_test_reduced = selector.transform(X_test)

    X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=selected_features)
    X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=selected_features)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(target_clf, X_train_reduced, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)

    holdout_f1 = utils.evaluate_model(target_clf, X_train_reduced_df, y_train, X_test_reduced_df, y_test)

    logging.info("-" * 40)
    logging.info("FINAL BASELINE RESULTS:")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Method: {args.method.upper()}")
    logging.info(f"Algorithm: {args.algorithm.upper()}")
    logging.info(f"Num Features Selected: {len(selected_features)}")
    logging.info(f"Selection Time: {selection_time:.4f}s")
    logging.info(f"CV F1-Score (Mean): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    logging.info(f"Test Set F1-Score: {holdout_f1:.4f}")
    logging.info("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Feature Selection Runner")

    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name (ereninho, batadal, wadi, etc)')
    parser.add_argument('-m', '--method', type=str, required=True,
                        choices=['sfs_forward', 'sfs_backward', 'rfe', 'rfecv'],
                        help='Baseline method to run')
    parser.add_argument('-a', '--algorithm', type=str, required=True, choices=['knn', 'nb', 'dt', 'linear_svc', 'rf'],
                        help='Classifier algorithm')
    parser.add_argument('-nf', '--n_features', type=int, default=None,
                        help='Number of features to select (Required for RFE, Optional for SFS, Ignored for RFECV)')


    args = parser.parse_args()

    try:
        run_baseline(args)
    except Exception as e:
        logging.error(f"Critical failure: {e}")
        import traceback

        traceback.print_exc()