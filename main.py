import datetime
import json

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
import utils
from priority_queue import MaxPriorityQueue
import logging
log_filename = f"results/log.txt"
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Nível de log (INFO, DEBUG, ERROR, etc.)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()  # Exibe no terminal
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def evaluate_algorithm(features_idx, algorithm):
    features = [feature_names[i] for i in features_idx]
    if algorithm == 'knn':
        model = KNeighborsClassifier(n_jobs=-1)
    elif algorithm == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm == 'nb':
        model = GaussianNB(var_smoothing=1e-9)
    elif algorithm == 'svm': # Warning: This is EXTREMELY SLOW on large datasets
        model = SVC(random_state=42)
    elif algorithm == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
    elif algorithm == 'linear_svc':
        model = LinearSVC(max_iter=1000, random_state=42, dual=False)
    elif algorithm == 'sgd':
        model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unsupported algorithm")

    return utils.evaluate_model(model, X_train[features], y_train, X_test[features], y_test)

def evaluate_baseline(feature_names, X_train, y_train, X_test, y_test, algorithm):
    logging.info("\nBaseline Evaluation with all features using the selected algorithm:")
    f1 = evaluate_algorithm(list(range(len(feature_names))), algorithm)
    logging.info(f"Baseline F1-Score ({algorithm.upper()}): {f1:.4f}")
    logging.info("-" * 50)
    return f1


def load_and_preprocess():
    X_train, y_train, X_test, y_test = utils.load_data()
    y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test)
    logging.info("Preprocessing completed successfully.")
    feature_names = X_train.columns.tolist()

    logging.info("Ranking Features using Mutual Information for composing RCL.")
    # Mutual Information (MI) measures the mutual dependence between two random variables.
    # In the context of feature selection, it evaluates how much information about the label
    # is provided by a particular feature.
    ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
    logging.info("Feature ranking completed.")

    sorted_features = sorted(zip(feature_names, ig_scores), key=lambda x: x[1], reverse=True)

    return X_train, y_train, X_test, y_test, feature_names, sorted_features, le

def print_feature_scores(sorted_features):
    logging.info("\nMutua Information for Features:")
    for feature, score in sorted_features:
        logging.info(f"Feature {feature}: MI = {score:.4f}")

def local_search(initial_solution, repeated_solutions_count, algorithm, rcl_size):
    max_f1_score = evaluate_algorithm(initial_solution, algorithm)
    best_solution = initial_solution.copy()
    seen_solutions = {frozenset(initial_solution)}

    logging.info(f"Starting Local Search with initial solution: {initial_solution}, F1-Score: {max_f1_score}")

    for iteration in range(args.local_iterations):
        new_solution = best_solution.copy()

        logging.info(
            f"  → Local Iteration {iteration + 1}/{args.local_iterations} | Current best F1: {max_f1_score:.4f}")

        for replace_index in range(len(new_solution)):
            RCL = [feature_names.index(feature) for feature, score in sorted_features[:rcl_size]
                   if feature_names.index(feature) not in new_solution]
            if not RCL:
                logging.info("    ✖ RCL is empty. No replacement possible.")
                break

            new_feature = random.choice(RCL)
            new_solution[replace_index] = new_feature

        new_solution_set = frozenset(new_solution)
        if new_solution_set in seen_solutions:
            repeated_solutions_count += 1
            logging.info(f"    ↺ Duplicate feature combination: {list(new_solution_set)} — Skipping")
            continue

        sorted_indices = sorted(new_solution)
        f1_score = evaluate_algorithm(new_solution, algorithm)
        logging.info(f"    ✓ Evaluated F1-Score: {f1_score:.4f} for solution: {sorted_indices}")

        if f1_score > max_f1_score and new_solution_set != frozenset(best_solution):
            max_f1_score = f1_score
            best_solution = new_solution
            seen_solutions.add(new_solution_set)
            sorted_best = sorted(best_solution)
            logging.info(
                f"        Improvement found! New best solution: {sorted_best} with F1-Score: {max_f1_score:.4f}")
        elif new_solution_set == frozenset(best_solution):
            logging.info("No real improvement (same as best solution)")

    logging.info(f"Local Search completed. Best F1-Score: {max_f1_score}, Best Solution: {best_solution}")
    return max_f1_score, best_solution, repeated_solutions_count

def construction(args):
    # 'sorted_features' is a list of tuples (feature, IG) sorted by IG. Picking the top X to compose the RCL.
    RCL = [feature for feature, _ in sorted_features[:args.rcl_size]]

    RCL_indices = [feature_names.index(feature) for feature in RCL]

    logging.info(f"RCL Features: {RCL}")
    logging.info(f"RCL Feature Indices: {RCL_indices}")

    all_solutions = []
    local_search_improvements = {}  # Dictionary to store results of local search

    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_solution = []

    seen_initial_solutions = set()
    repeated_solutions_count = 0  # Initialize the counter for repeated solutions
    repeated_solutions_count_local_search = 0  # Initialize the counter for repeated solutions during local search

    start_time = time.perf_counter()

    if args.rcl_size > len(feature_names):
        raise ValueError("The RCL size cannot exceed the number of available features.")
    if args.initial_solution > args.rcl_size:
        raise ValueError("The initial solution size cannot exceed the RCL size.")

    for iteration in range(args.constructive_iterations):
        # Ensure the initial solution is unique
        while True:
            # Randomly select k features from RCL to generate initial solutions
            selected_features = random.sample(RCL, k=args.initial_solution)
            # Convert feature names into indices
            solution = [feature_names.index(feature_name) for feature_name in selected_features]
            solution_set = frozenset(selected_features)

            if solution_set not in seen_initial_solutions:
                seen_initial_solutions.add(solution_set)
                break
            else:
                repeated_solutions_count += 1  # Incrementa o contador
                logging.info(f"Repeated initial solution found: {solution}, generating a new solution...")

        f1_score = evaluate_algorithm(solution, args.algorithm)
        logging.info(f"F1-Score: {f1_score} for solution: {solution}")
        all_solutions.append((iteration, f1_score, solution))

        if f1_score > 0.0:
            # If the priority queue is not full, simply insert the new F1-Score.
            if len(priority_queue.heap) < args.priority_queue:
                priority_queue.insert((f1_score, solution))
            else:
                # If the priority queue is full, find the lowest F1-Score in the queue.
                lowest_f1 = min(priority_queue.heap, key=lambda x: x[0])[0]
                if f1_score > lowest_f1:
                    # Remove the item with the lowest F1-Score before inserting the new item.
                    priority_queue.heap.remove((lowest_f1, [item[1] for item in priority_queue.heap if item[0] == lowest_f1][0]))
                    priority_queue.insert((f1_score, solution))
        local_search_improvements[tuple(solution)] = 0

        # visualize_heap(priority_queue.heap)
    total_elapsed_time = time.perf_counter() - start_time
    logging.info(f"Total repeated initial solutions: {repeated_solutions_count}")
    logging.info(f"Total execution time for Constructive Phase: {total_elapsed_time} seconds")
    print_priority_queue(priority_queue)
    utils.plot_solutions_with_priority(all_solutions, priority_queue)

    start_time = time.perf_counter()  # Local Search Phase
    total_iterations = len(priority_queue.heap) * args.local_iterations  # Total predicted iterations
    queue_progress = 0

    priority_queue_snapshot = list(priority_queue.heap) # saves priority solutions

    while not priority_queue.is_empty():
        _, current_solution = priority_queue.extract_max()

        original_f1_score = evaluate_algorithm(current_solution, args.algorithm)  # Evaluate the current solution once
        improved_f1_score, improved_solution, repeated_solutions_count_local_search = local_search(
        current_solution, repeated_solutions_count_local_search, args.algorithm, args.rcl_size)

        # Increment iteration count
        queue_progress += 1

        # Progress log
        elapsed_time = time.perf_counter() - start_time
        estimated_total_time = (elapsed_time / queue_progress) * total_iterations
        logging.info(
            f"[{queue_progress}/{args.priority_queue}] Best solution: F1-Score {improved_f1_score:.4f} |"
            f" Estimated remaining time: {estimated_total_time - elapsed_time:.2f}s")

        # Check if there was an improvement compared to the original F1-Score of the specific solution
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score
            logging.info(f"Improvement in Local Search! F1-Score: {improved_f1_score} for solution: {current_solution}. New solution: {improved_solution}")

        # Check if the improved solution is the global best solution
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            logging.info(f"New Global Best Solution! F1-Score: {max_f1_score} for solution: {best_solution}")

    total_local_search_time = time.perf_counter() - start_time  # Busca Local

    utils.plot_solutions(all_solutions, priority_queue_snapshot, local_search_improvements)

    logging.info(f"Total repeated solutions in local search: {repeated_solutions_count_local_search}")
    logging.info(f"Initial Solution Size: {selected_features}")
    logging.info(f"RCL Size: {len(RCL)}")
    logging.info(f"Best F1-Score: {max_f1_score}")
    logging.info(f"Best Feature Set (indices): {best_solution}")

    # Map indices to feature names
    best_feature_names = [(feature_names[i], i) for i in best_solution]
    formatted_best_features = ", ".join([f"'{name}' ({index})" for name, index in best_feature_names])

    logging.info(f"Best Feature Set (names): {formatted_best_features}")

    logging.info(f"Total execution time for Constructive Phase: {total_elapsed_time} seconds")
    logging.info(f"Total execution time for Local Search Phase: {total_local_search_time} seconds")

def print_priority_queue(priority_queue):
    logging.info("Priority Queue:")
    for score, solution in priority_queue.heap:
        logging.info(f"F1-Score: {-score}, Solution: {solution}")

if __name__ == '__main__':
    args = utils.parse_args()

    logging.info("Execution parameters:")
    logging.info(f"  Algorithm: {args.algorithm}")
    logging.info(f"  RCL Size: {args.rcl_size}")
    logging.info(f"  Initial Solution Size: {args.initial_solution}")
    logging.info(f"  Priority Queue Size: {args.priority_queue}")
    logging.info(f"  Local Search Iterations: {args.local_iterations}")
    logging.info(f"  Constructive Iterations: {args.constructive_iterations}")
    logging.info("-" * 50)

    # Load and preprocess the data
    X_train, y_train, X_test, y_test, feature_names, sorted_features, le = load_and_preprocess()

    # Print IG scores
    print_feature_scores(sorted_features)

    # Initial evaluation (baseline)
    baseline_f1 = evaluate_baseline(feature_names, X_train, y_train, X_test, y_test, args.algorithm)

    # Continue with the selected algorithm for the next steps
    logging.info(f"Selected algorithm for constructive and local search phases: {args.algorithm.upper()}")

    # Execute construction and local search
    construction(args)
    logging.info(f"Baseline F1-Score (All Features with {args.algorithm.upper()}): {baseline_f1:.4f}")

