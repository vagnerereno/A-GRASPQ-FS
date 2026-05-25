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
import os
import utils
from priority_queue import MaxPriorityQueue
import logging
os.makedirs("results", exist_ok=True)
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
import itertools

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
                # print(f"DEBUG: Treinando com {len(y_train_sub)} amostras (Sampling: {sample_ratio})")
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


def get_shuffled_neighborhood(solution, all_features, rcl_size, sorted_features):
    current_set = set(solution)

    rcl_indices = [all_features.index(feat) for feat, _ in sorted_features[:rcl_size]]

    possible_adds = [idx for idx in rcl_indices if idx not in current_set]
    possible_removes = list(solution)

    moves = []

    # A. Move: ADD
    for feat in possible_adds:
        moves.append(('add', feat))

    # B. Move: REMOVE
    if len(solution) > 2:
        for feat in possible_removes:
            moves.append(('remove', feat))

    # C. Move: SWAP
    if possible_adds and possible_removes:
        for out_f in possible_removes:
            for in_f in possible_adds:
                moves.append(('swap', out_f, in_f))

    random.shuffle(moves)

    for move in moves:
        yield move

def local_search(initial_solution, repeated_solutions_count, algorithm, rcl_size):
    max_f1_score = evaluate_algorithm(initial_solution, algorithm, use_sampling=True)
    best_solution = initial_solution.copy()
    seen_solutions = {frozenset(initial_solution)}
    unique_neighbors_explored = 0

    logging.info(f"Starting Smart VNS. Initial F1: {max_f1_score:.4f}, Size: {len(best_solution)}")

    neighbor_iterator = get_shuffled_neighborhood(best_solution, feature_names, rcl_size, sorted_features)

    current_solution_base = best_solution[:]

    for iteration in range(args.local_iterations):

        # 1. Try to grab the next neighbor from the shuffled list.
        try:
            move = next(neighbor_iterator)
        except StopIteration:
            # If the list is exhausted, it means we visited ALL the neighbors of this solution and didn't find anything better.
            # We are in a strict Great Location. We stop.
            logging.info(f"    → Neighborhood exhausted (Local Optimum reached). Stopping early.")
            break

        # 2. Apply the movement
        neighbor_solution = current_solution_base[:]
        move_type = move[0]

        if move_type == 'add':
            neighbor_solution.append(move[1])
        elif move_type == 'remove':
            if move[1] in neighbor_solution:
                neighbor_solution.remove(move[1])
            else:
                continue
        elif move_type == 'swap':
            if move[1] in neighbor_solution:
                neighbor_solution.remove(move[1])
                neighbor_solution.append(move[2])
            else:
                continue

        neighbor_solution.sort()
        neighbor_set = frozenset(neighbor_solution)

        if neighbor_set in seen_solutions:
            repeated_solutions_count += 1
            continue

        unique_neighbors_explored += 1
        seen_solutions.add(neighbor_set)

        # Fast evaluation
        f1_score = evaluate_algorithm(neighbor_solution, algorithm, use_sampling=True)

        logging.debug(f"    Iter {iteration+1}: {move_type} | F1: {f1_score:.4f}")

        # Acceptance Criteria (Improved?)
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            best_solution = neighbor_solution

            logging.info(
                f"    >>> New Best! Move: {move_type.upper()} | Size: {len(neighbor_solution)} | Fast F1: {f1_score:.4f}")

            # If we find a better solution, we'll shift the focus of our search!
            # We need to create a NEW neighborhood around this new, improved solution.
            current_solution_base = best_solution[:]
            neighbor_iterator = get_shuffled_neighborhood(best_solution, feature_names, rcl_size, sorted_features)

        # If it hasn't improved, the loop continues and in the next iteration we get the next neighbor in the queue

        # of the SAME base solution. This ensures systematic scanning.

    # Robust Evaluation
    logging.info("Running Final Robust Evaluation (100% data)...")
    final_robust_f1 = evaluate_algorithm(best_solution, algorithm, use_sampling=False)

    logging.info(f"Local Search completed. Robust F1: {final_robust_f1:.4f}, Size: {len(best_solution)}")

    return final_robust_f1, best_solution, repeated_solutions_count, unique_neighbors_explored




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
    plot_data_initial_size = []
    plot_data_final_size = []
    plot_data_initial_f1 = []
    plot_data_final_f1 = []

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
    smooth_factor = 0.5

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

    logging.info("\n" + "=" * 60)
    logging.info("  [ANALYSIS] SOLUTIONS ENTERING LOCAL SEARCH (Constructive Output)")
    logging.info("  Checking for size diversity before VNS:")

    sorted_pq = sorted(priority_queue.heap, key=lambda x: x[0], reverse=True)

    sizes_constructive = []
    for i, (f1, sol) in enumerate(sorted_pq):
        logging.info(f"    Sol #{i + 1}: Size {len(sol):<2} | F1: {f1:.4f} | Features: {sol}")
        sizes_constructive.append(len(sol))

    logging.info(f"  -> Summary of Sizes from Construction: {sorted(sizes_constructive)}")
    logging.info("=" * 60 + "\n")

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
    N_total_features = len(feature_names)

    for initial_f1, current_solution in solutions_to_process:

        original_f1_score = initial_f1
        original_size = len(current_solution)

        # --- MATHEMATICAL CALCULATION OF NEIGHBORHOOD ---
        k = len(current_solution)  # adaptive size — varies per solution
        possible_adds = N_total_features - k
        possible_removes = k
        possible_swaps = k * (N_total_features - k)
        total_neighbors_1step = possible_adds + possible_removes + possible_swaps
        # ------------------------------------------------------------------

        # Call the Local VNS Search (Add/Remove/Swap)
        improved_f1_score, improved_solution, repeated_solutions_count_local_search, unique_count = local_search(
            current_solution, repeated_solutions_count_local_search, args.algorithm, args.rcl_size
        )

        final_size = len(improved_solution)
        coverage_pct = (unique_count / total_neighbors_1step) * 100 if total_neighbors_1step > 0 else 0

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

        size_diff = final_size - original_size
        status_tag = "[=] SAME"
        if size_diff > 0:
            status_tag = f"[+] GREW (+{size_diff})"
        elif size_diff < 0:
            status_tag = f"[-] SHRANK ({size_diff})"

        logging.info(
            f"    [{queue_progress}/{total_solutions_ls}] {status_tag} Size {original_size} -> {final_size} | F1: {original_f1_score:.4f} -> {improved_f1_score:.4f}")
        logging.info(
            f"       Analysis: Explored {unique_count}/{total_neighbors_1step} neighbors ({coverage_pct:.1f}% coverage). ETA: {eta:.0f}s")

        # Improvement recorded
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score

        # Update Best Global
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            logging.info(f"    >>> NEW GLOBAL BEST! F1: {max_f1_score:.4f} <<<")

        plot_data_initial_size.append(original_size)
        plot_data_final_size.append(final_size)
        plot_data_initial_f1.append(initial_f1)
        plot_data_final_f1.append(improved_f1_score)

    total_local_search_time = time.perf_counter() - start_time

    utils.plot_solutions(all_solutions, priority_queue_snapshot, local_search_improvements)

    try:
        utils.plot_size_evolution(plot_data_initial_size, plot_data_final_size, plot_data_initial_f1, plot_data_final_f1,
                                  filename="results/evolution_plot.png")
    except Exception as e:
        logging.error(f"Could not plot size evolution: {e}")

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