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

    logging.info(f"Starting VNS Local Search with initial solution: {initial_solution}, F1-Score: {max_f1_score:.4f}")

    for iteration in range(args.local_iterations):
        current_solution = best_solution.copy()

        rcl_indices = [feature_names.index(feat) for feat, _ in sorted_features[:rcl_size]]

        # Candidatos para ADICIONAR (estão na RCL mas não na solução)
        candidates_to_add = [idx for idx in rcl_indices if idx not in current_solution]

        # Candidatos para REMOVER (estão na solução atual)
        candidates_to_remove = current_solution[:]

        # Definir Movimentos Possíveis
        possible_moves = []

        # Add/Swap só são possíveis se houver candidatos fora da solução
        if candidates_to_add:
            possible_moves.append('add')
            if candidates_to_remove:
                possible_moves.append('swap')
        # Remove só é possível se a solução tiver um tamanho mínimo (ex: > 1 ou 2)
        if len(candidates_to_remove) > 2:
            possible_moves.append('remove')
        if not possible_moves:
            logging.info(
                f"    → Local Iteration {iteration + 1}/{args.local_iterations} | No moves possible. Stopping.")
            break

        # Escolher e Executar Movimento
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

        neighbor_solution.sort()  # Ordena para garantir consistência no frozenset
        neighbor_solution_set = frozenset(neighbor_solution)

        # Verifica se solução é válida ou duplicada
        if not neighbor_solution:
            continue
        if neighbor_solution_set in seen_solutions:
            repeated_solutions_count += 1
            logging.info(f" ↺ Duplicate: {list(neighbor_solution_set)} — Skipping")
            continue

        seen_solutions.add(neighbor_solution_set)

        f1_score = evaluate_algorithm(neighbor_solution, algorithm)

        log_msg = f"    → Local Iter {iteration + 1}/{args.local_iterations} | Move: {move_type.upper():<6} | Size: {len(neighbor_solution):<2} | F1: {f1_score:.4f}"

        if f1_score > max_f1_score:
            max_f1_score = f1_score
            best_solution = neighbor_solution
            logging.info(f"{log_msg} >>> New Best! <<<")
        else:
            logging.info(log_msg)

        logging.info(f"Local Search completed. Best F1: {max_f1_score:.4f}, Size: {len(best_solution)}")

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

    # --- NOVOS PARÂMETROS PARA CONSTRUÇÃO VARIÁVEL E SEMI-GREEDY ---
    # Define alpha (padrão 1.0 se não existir, ou seja, aleatório puro)
    alpha = getattr(args, 'alpha', 1.0)
    alpha = max(0.01, min(1.0, alpha))  # Garante limites seguros

    # Define intervalo de tamanho (min_k e max_k)
    # Se não definidos nos args, usa o initial_solution fixo como fallback para o min
    min_k = getattr(args, 'min_initial_solution', args.initial_solution)
    # O máximo nunca pode ser maior que a própria RCL
    max_k = min(getattr(args, 'max_initial_solution', args.rcl_size), len(RCL))

    if min_k > max_k: min_k = max_k
    # --- ---

    # --- LÓGICA REATIVA- Inicialização ---
    # Lista de tamanhos possíveis
    possible_sizes = list(range(min_k, max_k + 1))
    # Pesos iniciais iguais para todos os tamanhos (chance igual de ser sorteado)
    size_weights = {k: 1.0 for k in possible_sizes}
    # Histórico para guardar os F1-Scores de cada tamanho
    size_history = {k: [] for k in possible_sizes}
    # Fator de suavização para garantir que nenhum tamanho fique com probabilidade zero
    smooth_factor = 0.05
    # -------------------------------------------------------

    if args.rcl_size > len(feature_names):
        raise ValueError("The RCL size cannot exceed the number of available features.")
    if args.initial_solution > args.rcl_size:
        raise ValueError("The initial solution size cannot exceed the RCL size.")

    for iteration in range(args.constructive_iterations):
        # Ensure the initial solution is unique
        attempts = 0
        while True:
            # --- 1. ESCOLHA DO TAMANHO (REATIVA) ---
            # Em vez de randint, sorteia com base nos pesos aprendidos
            weights_list = [size_weights[k] for k in possible_sizes]
            # random.choices retorna uma lista, pegamos o primeiro item
            current_k_size = random.choices(possible_sizes, weights=weights_list, k=1)[0]

            selected_features = []
            current_rcl_pool = RCL.copy()  # Cópia para remover itens sem afetar a lista original

            # 2. Construção iterativa baseada em Alpha
            while len(selected_features) < current_k_size and current_rcl_pool:
                # Define o tamanho da lista restrita de candidatos (RCL da iteração)
                # Alpha % dos melhores restantes
                limit = max(1, int(len(current_rcl_pool) * alpha))
                candidates = current_rcl_pool[:limit]

                # Escolhe um aleatoriamente dessa fatia superior
                chosen = random.choice(candidates)
                selected_features.append(chosen)
                current_rcl_pool.remove(chosen)

            # -------------------------------------------------------

            # Convert feature names into indices
            solution = [feature_names.index(feature_name) for feature_name in selected_features]
            solution_set = frozenset(selected_features)

            if solution_set not in seen_initial_solutions:
                seen_initial_solutions.add(solution_set)
                break
            else:
                repeated_solutions_count += 1  # Incrementa o contador
                logging.info(f"Repeated initial solution found: {solution}, generating a new solution...")

            attempts += 1
            if attempts > 50:  # Evita loop infinito se o espaço de busca for pequeno
                break

        f1_score = evaluate_algorithm(solution, args.algorithm)
        logging.info(f"F1-Score: {f1_score} for solution: {solution}")
        size_history[current_k_size].append(f1_score)

        # A cada 10 iterações, recalibra a "roleta"
        if (iteration + 1) % 10 == 0:
            max_avg_f1 = 0
            avgs = {}
            # Calcula a média de F1 para cada tamanho testado até agora
            for k in possible_sizes:
                if size_history[k]:
                    avg = sum(size_history[k]) / len(size_history[k])
                else:
                    avg = 0  # Se ainda não foi sorteado
                avgs[k] = avg
                if avg > max_avg_f1: max_avg_f1 = avg

            # Atualiza os pesos: quanto maior o F1 médio, maior o peso
            for k in possible_sizes:
                # Normaliza pelo melhor para manter a escala, soma o fator de suavização
                normalized_score = (avgs[k] / max_avg_f1) if max_avg_f1 > 0 else 0
                size_weights[k] = normalized_score + smooth_factor

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

