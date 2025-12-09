import argparse
import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
import logging

logger = logging.getLogger(__name__)

def load_data(dataset_name='ereninho', base_path='data/'):
    """
    Loads and preprocesses datasets based on the provided name.
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == 'ereninho':
        filepath = os.path.join(base_path, 'ERENINHO_10k.csv')
        target_column = 'class'
        try:
            df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded {dataset_name}. Shape: {df.shape}")

            constant_columns = ['frameLen', 'gooseLen', 'numDatSetEntries', 'APDUSize',
                                'gooseLengthDiff', 'apduSizeDiff', 'frameLengthDiff',
                                'ethDst', 'ethSrc', 'ethType', 'gooseAppid', 'TPID',
                                'gocbRef', 'datSet', 'goID', 'test', 'ndsCom', 'protocol']

            constant_columns_exist = [col for col in constant_columns if col in df.columns]
            if constant_columns_exist:
                logger.warning(
                    f"Removing {len(constant_columns_exist)} constant columns for {dataset_name}: {constant_columns_exist}")
                df = df.drop(columns=constant_columns_exist)

            X = df.drop(columns=[target_column])
            y = df[target_column]

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except KeyError:
            logger.error(f"Target column '{target_column}' not found in {filepath}.")
            raise

    elif dataset_name == 'batadal':
        file_train1 = os.path.join(base_path, 'BATADAL_dataset03.csv')
        file_train2 = os.path.join(base_path, 'BATADAL_dataset04.csv')
        target_column = 'ATT_FLAG'
        try:
            df1 = pd.read_csv(file_train1, sep=',', skipinitialspace=True)
            df2 = pd.read_csv(file_train2, sep=',', skipinitialspace=True)
            df1.columns = df1.columns.str.strip()
            df2.columns = df2.columns.str.strip()

            df2[target_column] = df2[target_column].replace(-999, 0)

            df_combined = pd.concat([df1, df2], ignore_index=True)
            logger.info(f"Loaded and combined BATADAL datasets. Combined shape: {df_combined.shape}")

            if 'DATETIME' in df_combined.columns:
                logger.info("Removing DATETIME column.")
                df_combined = df_combined.drop(columns=['DATETIME'])

            constant_columns = [col for col in df_combined.columns if df_combined[col].nunique(dropna=False) == 1]
            if constant_columns:
                logger.warning(
                    f"Removing {len(constant_columns)} constant columns for {dataset_name}: {constant_columns}")
                df_combined = df_combined.drop(columns=constant_columns)
            else:
                logger.info("No constant columns found after combining.")

            X = df_combined.drop(columns=[target_column])
            y = df_combined[target_column]

        except FileNotFoundError as e:
            logger.error(f"BATADAL file not found: {e.filename}")
            raise
        except KeyError:
            logger.error(f"Target column '{target_column}' not found in one of the BATADAL files.")
            raise
    elif dataset_name == 'wadi':
        filepath = os.path.join(base_path, 'WADI.csv')
        original_target_col = 'Attack LABLE (1:No Attack, -1:Attack)'
        new_target_col = 'Attack_Label'
        try:
            df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded {dataset_name}. Initial Shape: {df.shape}")

            if original_target_col in df.columns:
                df.rename(columns={original_target_col: new_target_col}, inplace=True)
                target_column = new_target_col
            else:
                raise KeyError(f"Original target column '{original_target_col}' not found.")

            cols_to_drop_null = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
            cols_to_drop_null_exist = [col for col in cols_to_drop_null if col in df.columns]
            if cols_to_drop_null_exist:
                logger.warning(f"Removing fully null columns: {cols_to_drop_null_exist}")
                df = df.drop(columns=cols_to_drop_null_exist)

            cols_to_drop_meta = ['Row', 'Date', 'Time']
            cols_to_drop_meta_exist = [col for col in cols_to_drop_meta if col in df.columns]
            if cols_to_drop_meta_exist:
                logger.info(f"Removing metadata columns: {cols_to_drop_meta_exist}")
                df = df.drop(columns=cols_to_drop_meta_exist)

            if target_column not in df.columns:
                raise KeyError(f"Target column '{target_column}' missing after initial processing.")
            X = df.drop(columns=[target_column])
            y = df[target_column]

            initial_rows = X.shape[0]
            rows_before_drop = len(y)
            if initial_rows != rows_before_drop:
                logger.error(
                    f"Inconsistent row counts before dropping NaNs: X has {initial_rows}, y has {rows_before_drop}")
                raise ValueError("Row count mismatch before dropping NaNs.")

            valid_indices = X.dropna().index
            if len(valid_indices) < initial_rows:
                num_dropped = initial_rows - len(valid_indices)
                logger.warning(f"Removing {num_dropped} rows with NaN values in features.")
                X = X.loc[valid_indices]
                y = y.loc[valid_indices]

            if X.shape[0] != len(y):
                logger.error(f"Inconsistent row counts after dropping NaNs: X has {X.shape[0]}, y has {len(y)}")
                raise ValueError("Row count mismatch after dropping NaNs.")

            logger.info("Mapping target variable: 1 -> 0 (Normal), -1 -> 1 (Attack)")
            y = y.map({1: 0, -1: 1})
            if y.isnull().any():
                logger.error("NaNs created during target mapping. Check original target values.")
                raise ValueError("Target mapping created NaNs.")

            constant_columns = [col for col in X.columns if X[col].nunique() == 1]
            if constant_columns:
                logger.warning(f"Removing {len(constant_columns)} constant columns found: {constant_columns}")
                X = X.drop(columns=constant_columns)
            else:
                logger.info("No constant columns found after cleaning.")
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except KeyError as e:
            logger.error(f"Column processing error for WADI: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during WADI processing: {e}")
            raise
    elif dataset_name == 'wustl':
        filepath = os.path.join(base_path, 'wustl-ehms-2020.csv')

        target_column = 'Attack Category'
        try:
            df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded {dataset_name}. Shape: {df.shape}")

            cols_to_drop = [
                'Dir', 'SrcAddr', 'DstAddr', 'DstMac',
                'Sport',
                'Label',
                'Dport', 'SrcGap', 'DstGap', 'DIntPktAct', 'dMinPktSz', 'Trans',
                'Flgs',
                'SrcMac'
            ]

            if target_column in cols_to_drop:
                cols_to_drop.remove(target_column)

            cols_to_drop_exist = [col for col in cols_to_drop if col in df.columns]
            if cols_to_drop_exist:
                logger.warning(
                    f"Removing {len(cols_to_drop_exist)} constant/meta/other columns for {dataset_name}: {cols_to_drop_exist}")
                df = df.drop(columns=cols_to_drop_exist)
            else:
                logger.info("No specified columns to drop were found.")

            if target_column not in df.columns:
                raise KeyError(f"Target column '{target_column}' not found after cleaning.")
            X = df.drop(columns=[target_column])
            y = df[target_column]

            if X.isnull().values.any() or y.isnull().values.any():
                logger.warning(
                    "NaN values detected unexpectedly after loading/cleaning WUSTL. Consider adding dropna().")
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}"); raise
        except KeyError as e:
            logger.error(f"Column processing error for WUSTL: {e}"); raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during WUSTL processing: {e}"); raise
    elif dataset_name == 'drone':
        filepath = os.path.join(base_path, 'drone-reduced.csv')
        target_column = 'label'

        try:
            df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded {dataset_name}. Shape: {df.shape}")

            cols_to_drop = ['fwd_seg_size_min']
            cols_to_drop_exist = [col for col in cols_to_drop if col in df.columns]

            if cols_to_drop_exist:
                logger.warning(
                    f"Removing {len(cols_to_drop_exist)} constant column(s) for {dataset_name}: {cols_to_drop_exist}")
                df = df.drop(columns=cols_to_drop_exist)
            else:
                logger.info("No specified columns to drop were found.")

            if target_column not in df.columns:
                raise KeyError(f"Target column '{target_column}' not found after cleaning.")

            X = df.drop(columns=[target_column])
            y = df[target_column]

            if X.isnull().values.any() or y.isnull().values.any():
                logger.warning(
                    "NaN values detected unexpectedly. Consider adding dropna().")

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}");
            raise
        except KeyError as e:
            logger.error(f"Column processing error for Drone dataset: {e}");
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}");
            raise
    elif dataset_name == 'ransomset':
        filepath = os.path.join(base_path, 'ransomset-multiclass-dataset.csv')
        target_column = 'classe'
        col_to_remove = 'score_binary'
        try:
            df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded {dataset_name}. Shape: {df.shape}")

            if col_to_remove in df.columns:
                logger.warning(f"Removing column '{col_to_remove}' for {dataset_name}.")
                df = df.drop(columns=[col_to_remove])
            else:
                logger.info(f"Column '{col_to_remove}' not found, skipping removal.")

            if target_column not in df.columns:
                raise KeyError(f"Target column '{target_column}' not found.")

            X = df.drop(columns=[target_column])
            y = df[target_column]

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except KeyError as e:
            logger.error(f"Column processing error for Ransomset: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during Ransomset processing: {e}")
            raise
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}. Choose 'ereninho', 'batadal', 'wadi', or 'ransomset'.")

    feature_names = X.columns.tolist()
    logger.info(f"Dataset '{dataset_name}' processed. Final features: {len(feature_names)}. Target: '{target_column}'.")
    return X, y, feature_names

def preprocess_data(X_train, y_train, X_test, y_test):
    # Identify numerical columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    # Use StandardScaler to normalize numerical data
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Use OneHotEncoder for categorical columns
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(X_train[cat_cols])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Drop original categorical columns and concat encoded columns
        X_train = pd.concat([X_train[num_cols], cat_encoded_df], axis=1)
        cat_encoded_test = encoder.transform(X_test[cat_cols])
        cat_encoded_test_df = pd.DataFrame(cat_encoded_test, columns=encoder.get_feature_names_out(cat_cols))
        X_test = pd.concat([X_test[num_cols], cat_encoded_test_df], axis=1)

    # Initialize LabelEncoder for labels/classes
    le = LabelEncoder()

    le.fit(y_train)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test) # Direct transformation of test labels

    return y_train, y_test, X_train, X_test, le

def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')

    return f1

def evaluate_model(classifier, X_train, y_train, X_test, y_test):
    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict labels for the test dataset
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    f1 = calculate_metrics(y_test, y_pred)

    return f1

def plot_solutions_with_priority(all_solutions, priority_queue):
    # Convert the priority queue into a set for fast lookup
    priority_set = set([tuple(sol) for _, sol in priority_queue.heap])

    # Extracting iteration indices and F1-Scores
    iterations = [iteration for _, iteration, _ in all_solutions]
    f1_scores = [f1 for f1, _, _ in all_solutions]

    # Checking which solutions are in the top 10
    priority_colors = ['red' if tuple(sol) in priority_set else 'blue' for _, _, sol in all_solutions]

    plt.scatter(f1_scores, iterations, color=priority_colors)
    plt.ylabel('F1-Score')
    plt.xlabel('Índice da Solução')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Top 10', markersize=10, markerfacecolor='red'),
                       plt.Line2D([0], [0], marker='o', color='w', label='Outras Soluções', markersize=10, markerfacecolor='blue')],
               loc='lower right')
    plt.savefig(f"priority_plot.png")


def plot_solutions(all_solutions, priority_queue_snapshot, local_search_improvements):
    plt.figure(figsize=(9, 4))

    # Convert the priority queue into a set for fast lookup
    priority_set = set([tuple(sol) for _, sol in priority_queue_snapshot])

    # Extracting iteration indices and F1-Scores
    iterations = [iteration for iteration, _, _ in all_solutions]
    f1_scores = [f1 for _, f1, _ in all_solutions]
    solutions = [sol for _, _, sol in all_solutions]

    # Draw all blue bars first
    plt.bar(iterations, f1_scores, color='blue', label='Soluções Iniciais (SI)')

    # Draw bars for priority queue solutions in red
    for i, sol in enumerate(solutions):
        if tuple(sol) in priority_set:
            plt.bar(iterations[i], f1_scores[i], color='red', label='SI Incluídas na Fila de Prioridades')

    # Overpaint improvements in green where applicable
    for i, sol in enumerate(solutions):
        improvement = local_search_improvements.get(tuple(sol), 0)
        if improvement > 0:
            plt.bar(iterations[i], improvement, bottom=f1_scores[i], color='green', label='SI Melhoradas na Busca Local')

    plt.xlabel('Índice da Solução', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # by_label = {label: handle for label, handle in by_label.items() if label in ['Soluções Iniciais', 'Top 10', 'Melhoradas na Busca Local']}
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', prop={'weight': 'bold'})
    plt.xlim(min(iterations) - 1, max(iterations) + 1)
    plt.tick_params(axis='x', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo x
    plt.tick_params(axis='y', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo y
    plt.tight_layout()

    plt.savefig(f"all_bestsolution.png")
    plt.savefig(f"all_bestsolution.pdf")



# Função para parsear argumentos da linha de comando
def parse_args():
    parser = argparse.ArgumentParser(
        description="GRASPQ-FS for Feature Selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str, choices=['ereninho', 'drone', 'wadi', 'batadal', 'wustl', 'ransomset'],
        default='ereninho',
        help="Dataset to use."
    )

    parser.add_argument(
        "-a", "--algorithm", "--alg",
        type=str, choices=['knn', 'dt', 'nb', 'svm', 'rf', 'xgboost', 'linear_svc', 'sgd'],
        default='nb',
        help="Algorithm to be used for evaluation (knn, dt, nb, rf, svm, xgboost, linear_svc, sgd)."
    )
    parser.add_argument(
        "-rcl", "--rcl_size", "--rcl",
        type=int, default=10,
        help="Size of the Restricted Candidate List (RCL)."
    )
    parser.add_argument(
        "-is", "--initial_solution", "--init_sol",
        type=int, default=5,
        help="Size of the initial solution generated."
    )
    parser.add_argument(
        "-pq", "--priority_queue", "--pq_size",
        type=int, default=10,
        help="Maximum size of the priority queue."
    )
    parser.add_argument(
        "-lc", "--local_iterations", "--ls",
        type=int, default=50,
        help="Number of iterations in the local search phase."
    )
    parser.add_argument(
        "-cc", "--constructive_iterations", "--const",
        type=int, default=100,
        help="Number of iterations in the constructive phase."
    )

    parser.add_argument("-max_is", "--max_initial_solution", type=int, default=30,
                        help="Maximum size for initial solution in constructive phase.")
    parser.add_argument("-alpha", "--alpha", type=float, default=0.3,
                        help="Randomness factor for semi-greedy construction (0.0 to 1.0).")
    parser.add_argument("-ess", "--evaluation_sample_size", type=float, default=0.1,
                        help="Fraction of training data to use for fast evaluation (0.0 to 1.0).")

    return parser.parse_args()

