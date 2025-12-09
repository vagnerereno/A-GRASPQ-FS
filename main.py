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
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
from sklearn.model_selection import train_test_split

def evaluate_algorithm(features_idx, algorithm, use_sampling=False):
    features = [feature_names[i] for i in features_idx]

    X_train_sub = X_train[features]
    X_test_sub = X_test[features]
    y_train_sub = y_train

    # Sampling Logic
    if use_sampling:
        sample_ratio = args.evaluation_sample_size
        if sample_ratio < 1.0 and len(y_train) > 100:
            try:
                X_train_sample, _, y_train_sample, _ = train_test_split(
                    X_train_sub, y_train_sub,
                    train_size=sample_ratio,
                    stratify=y_train_sub,
                    random_state=42
                )
                X_train_sub = X_train_sample
                y_train_sub = y_train_sample
                print(f"DEBUG: Treinando com {len(y_train_sub)} amostras (Sampling: {sample_ratio})")
            except ValueError:
                pass

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

    return utils.evaluate_model(model, X_train_sub, y_train_sub, X_test_sub, y_test)

def evaluate_baseline(feature_names, X_train, y_train, X_test, y_test, algorithm):
    logging.info("\nBaseline Evaluation with all features using the selected algorithm:")
    f1 = evaluate_algorithm(list(range(len(feature_names))), algorithm)
    logging.info(f"Baseline F1-Score ({algorithm.upper()}): {f1:.4f}")
    logging.info("-" * 50)
    return f1


def load_and_preprocess(args):
    X, y, feature_names = utils.load_data(args.dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test)

    logging.info("Preprocessing completed successfully.")

    feature_names = X_train.columns.tolist()

    logging.info("Ranking Features using Mutual Information for composing RCL.")
    ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
    logging.info("Feature ranking completed.")

    sorted_features = sorted(zip(feature_names, ig_scores), key=lambda x: x[1], reverse=True)

    return X_train, y_train, X_test, y_test, feature_names, sorted_features, le

def print_feature_scores(sorted_features):
    logging.info("\nMutua Information for Features:")
    for feature, score in sorted_features:
        logging.info(f"Feature {feature}: MI = {score:.4f}")


def local_search(initial_solution, repeated_solutions_count, algorithm, rcl_size):
    # Evaluate the initial solution.
    max_f1_score = evaluate_algorithm(initial_solution, algorithm, use_sampling=True)
    best_solution = initial_solution.copy()
    seen_solutions = {frozenset(initial_solution)}

    logging.info(f"Starting VNS Local Search. Initial F1: {max_f1_score:.4f}, Size: {len(best_solution)}")

    for iteration in range(args.local_iterations):
        current_solution = best_solution.copy()

        # --- 1. Define Candidates ---
        rcl_indices = [feature_names.index(feat) for feat, _ in sorted_features[:rcl_size]]
        candidates_to_add = [idx for idx in rcl_indices if idx not in current_solution]
        candidates_to_remove = current_solution[:]

        # --- 2. Define Movements ---
        possible_moves = []
        if candidates_to_add:
            possible_moves.append('add')
            if candidates_to_remove:
                possible_moves.append('swap')
        if len(candidates_to_remove) > 2:
            possible_moves.append('remove')

        if not possible_moves:
            logging.info(f"    → Local Iteration {iteration + 1} | No moves possible. Stopping.")
            break

        # --- 3. Choose the Movement and Execute ---
        move_type = random.choice(possible_moves)
        neighbor_solution = []

        try:
            if move_type == 'swap':
                feat_out = random.choice(candidates_to_remove)
                feat_in = random.choice(candidates_to_add)
                neighbor_solution = [f for f in current_solution if f != feat_out]
                neighbor_solution.append(feat_in)
            elif move_type == 'add':
                feat_in = random.choice(candidates_to_add)
                neighbor_solution = current_solution[:]
                neighbor_solution.append(feat_in)
            elif move_type == 'remove':
                feat_out = random.choice(candidates_to_remove)
                neighbor_solution = [f for f in current_solution if f != feat_out]
        except IndexError:
            continue

        neighbor_solution.sort()
        neighbor_solution_set = frozenset(neighbor_solution)

        # --- 4. Verification of Duplicates ("Visited?") ---
        if not neighbor_solution: continue

        if neighbor_solution_set in seen_solutions:
            repeated_solutions_count += 1
            logging.info(f" ↺ Duplicate skipping")
            continue

        seen_solutions.add(neighbor_solution_set)

        # --- 5. Fast Evaluation ---
        # Note: The evaluate_algorithm uses the 'slice' defined in args.evaluation_sample_size
        f1_score = evaluate_algorithm(neighbor_solution, algorithm, use_sampling=True)

        log_msg = f"    → Local Iter {iteration + 1}/{args.local_iterations} | Move: {move_type.upper():<6} | Size: {len(neighbor_solution):<2} | F1: {f1_score:.4f}"

        # --- 6. Acceptance Criteria ("Improved?") ---
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            best_solution = neighbor_solution
            logging.info(f"    >>> New Best! Move: {move_type.upper()} | Size: {len(neighbor_solution)} | Fast F1: {f1_score:.4f}")
            # Return to the loop with the new best solution (Arrow "Yes")
        else:
            logging.info(log_msg)
            # Return to the loop while keeping the previous one (Arrow "No")

    # --- 7. Final Robust Evaluation ---
    # Recalculate the F1 value of the best solution using 100% of the data to obtain the true value.
    logging.info("Running Final Robust Evaluation (100% data)...")

    # Saves the original sample size.
    final_robust_f1 = evaluate_algorithm(best_solution, algorithm, use_sampling=False)

    logging.info(
        f"Local Search completed. Robust F1: {final_robust_f1:.4f} (Est: {max_f1_score:.4f}), Size: {len(best_solution)}")

    # The Robust F1 returns.
    return final_robust_f1, best_solution, repeated_solutions_count


def construction(args):
    RCL = [feature for feature, _ in sorted_features[:args.rcl_size]]
    RCL_indices = [feature_names.index(feature) for feature in RCL]

    logging.info(f"RCL Features: {RCL}")
    logging.info(f"RCL Feature Indices: {RCL_indices}")

    all_solutions = []
    local_search_improvements = {}
    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_solution = []
    seen_initial_solutions = set()
    repeated_solutions_count = 0
    repeated_solutions_count_local_search = 0

    start_time = time.perf_counter()

    # 1. Semi-Greedy (Alpha)
    alpha = getattr(args, 'alpha', 1.0)
    alpha = max(0.01, min(1.0, alpha))

    # 2. Variable Size (Min/Max K)
    min_k = getattr(args, 'min_initial_solution', args.initial_solution)
    max_k = min(getattr(args, 'max_initial_solution', args.rcl_size), len(RCL))
    if min_k > max_k: min_k = max_k

    # 3. Initialize Weights
    possible_sizes = list(range(min_k, max_k + 1))
    size_weights = {k: 1.0 for k in possible_sizes}  # Equal weights at the beginning
    size_history = {k: [] for k in possible_sizes}  # History for Feedback
    smooth_factor = 0.05

    if args.rcl_size > len(feature_names):
        raise ValueError("RCL size cannot exceed available features.")

    # --- PART 1: REACTIVE CONSTRUCTION PHASE ---
    logging.info(f"  [Constructive Phase] Starting Reactive GRASP. Size Range: [{min_k}-{max_k}]")

    for iteration in range(args.constructive_iterations):
        attempts = 0
        while True:
            # A. Probabilistic Selection (SelectK)
            # Randomly select k based on current weights (Roulette)
            weights_list = [size_weights[k] for k in possible_sizes]
            current_k_size = random.choices(possible_sizes, weights=weights_list, k=1)[0]

            # B. Semi-Greedy Construction (BuildSol)
            selected_features = []
            current_rcl_pool = RCL.copy()

            while len(selected_features) < current_k_size and current_rcl_pool:
                # Apply Alpha to restrict candidates.
                limit = max(1, int(len(current_rcl_pool) * alpha))
                candidates = current_rcl_pool[:limit]
                chosen = random.choice(candidates)
                selected_features.append(chosen)
                current_rcl_pool.remove(chosen)

            # Convert to indexes
            solution = [feature_names.index(feature_name) for feature_name in selected_features]
            solution_set = frozenset(selected_features)

            # Uniqueness Verification
            if solution_set not in seen_initial_solutions:
                seen_initial_solutions.add(solution_set)
                break
            else:
                repeated_solutions_count += 1

            attempts += 1
            if attempts > 50: break  # Avoid infinite loop

        # C. Fast Evaluation
        f1_score = evaluate_algorithm(solution, args.algorithm, use_sampling=True)

        logging.debug(f"    [Const. Iter {iteration + 1}] Size: {len(solution)} | F1: {f1_score:.4f}")
        all_solutions.append((iteration, f1_score, solution))

        # D. Register Feedback
        size_history[current_k_size].append(f1_score)

        # E. Check Update?
        # Recalibrates every 10 iterations ("Every X iterations")
        if (iteration + 1) % 10 == 0:
            # F. Recalibrate Weights
            max_avg_f1 = 0
            avgs = {}
            for k in possible_sizes:
                hist = size_history[k]
                avg = sum(hist) / len(hist) if hist else 0
                avgs[k] = avg
                if avg > max_avg_f1: max_avg_f1 = avg

            # Updates weights proportionally to performance + smoothing
            for k in possible_sizes:
                normalized_score = (avgs[k] / max_avg_f1) if max_avg_f1 > 0 else 0
                size_weights[k] = normalized_score + smooth_factor

            # Log in to see the learning happening.
            # best_current_k = max(size_weights, key=size_weights.get)
            # logging.debug(f"    [Reactive Update] Weights updated. Top performing size: {best_current_k}")

        # G. Queueing Criteria (PQ Criteria Met?)
        if f1_score > 0.0:
            if len(priority_queue.heap) < args.priority_queue:
                priority_queue.insert((f1_score, solution))
            else:
                lowest_f1 = min(priority_queue.heap, key=lambda x: x[0])[0]
                if f1_score > lowest_f1:
                    # Remove the worst and insert the new.
                    for item in priority_queue.heap:
                        if item[0] == lowest_f1:
                            priority_queue.heap.remove(item)
                            break
                    priority_queue.insert((f1_score, solution))

        local_search_improvements[tuple(solution)] = 0

    total_elapsed_time = time.perf_counter() - start_time
    logging.info(
        f"  [Constructive Phase] Finished in {total_elapsed_time:.2f}s. Repeated solutions: {repeated_solutions_count}")

    # Generates a graph of the solutions (optional)
    # utils.plot_solutions_with_priority(all_solutions, priority_queue)

    # --- PART 2: LOCAL VNS SEARCH PHASE ---
    start_time = time.perf_counter()

    # Snapshot for plotting
    priority_queue_snapshot = list(priority_queue.heap)

    # Extracts solutions from the queue for processing.
    solutions_to_process = []
    while not priority_queue.is_empty():
        solutions_to_process.append(priority_queue.extract_max())

    total_solutions_ls = len(solutions_to_process)
    logging.info(f"  [Local Search Phase] Starting VNS on {total_solutions_ls} solutions...")

    queue_progress = 0
    total_iterations_estimated = total_solutions_ls * args.local_iterations

    for initial_f1, current_solution in solutions_to_process:

        original_f1_score = initial_f1

        # Call the Local VNS Search (Add/Remove/Swap)
        improved_f1_score, improved_solution, repeated_solutions_count_local_search = local_search(
            current_solution, repeated_solutions_count_local_search, args.algorithm, args.rcl_size
        )

        queue_progress += 1
        elapsed_time = time.perf_counter() - start_time
        # Simple estimate of remaining time.
        if queue_progress > 0:
            avg_time_per_sol = elapsed_time / queue_progress
            remaining_sols = total_solutions_ls - queue_progress
            eta = avg_time_per_sol * remaining_sols
        else:
            eta = 0

        logging.info(
            f"    [{queue_progress}/{total_solutions_ls}] Best Local F1: {improved_f1_score:.4f} (Size {len(improved_solution)}) | ETA: {eta:.0f}s")

        # Improvement recorded
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score

        # Update Best Global
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            logging.info(f"    >>> NEW GLOBAL BEST! F1: {max_f1_score:.4f} <<<")

    total_local_search_time = time.perf_counter() - start_time

    # utils.plot_solutions(all_solutions, priority_queue_snapshot, local_search_improvements)

    logging.info(f"Total repeated solutions in local search: {repeated_solutions_count_local_search}")
    logging.info(f"RCL Size: {len(RCL)}")
    logging.info(f"Best F1-Score: {max_f1_score}")

    logging.info(f"Best Feature Set (indices): {best_solution}")

    best_feature_names = [(feature_names[i], i) for i in best_solution]
    formatted_best_features = ", ".join([f"'{name}' ({index})" for name, index in best_feature_names])
    logging.info(f"Best Feature Set (names): {formatted_best_features}")
    logging.info(f"Best Solution Size: {len(best_solution)}")

    logging.info(f"Total execution time for Constructive Phase: {total_elapsed_time} seconds")
    logging.info(f"Total execution time for Local Search Phase: {total_local_search_time} seconds")

    return max_f1_score, best_solution, total_elapsed_time, total_local_search_time

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
    X_train, y_train, X_test, y_test, feature_names, sorted_features, le = load_and_preprocess(args)

    # Print IG scores
    print_feature_scores(sorted_features)

    # Initial evaluation (baseline)
    baseline_f1 = evaluate_baseline(feature_names, X_train, y_train, X_test, y_test, args.algorithm)

    # Continue with the selected algorithm for the next steps
    logging.info(f"Selected algorithm for constructive and local search phases: {args.algorithm.upper()}")

    # Execute construction and local search
    construction(args)
    logging.info(f"Baseline F1-Score (All Features with {args.algorithm.upper()}): {baseline_f1:.4f}")

