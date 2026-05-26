import time
import random
from collections import deque
from geometry import Container, Box
from constructive import ExtremePointsHeuristic

VOLUME_TOLERANCE = 1e-6


def evaluate_solution(solution, container_width, container_height, container_depth,
                      original_boxes, box_order):

    test_container = Container(container_width, container_height, container_depth)
    heuristic = ExtremePointsHeuristic(test_container)

    for index, box_idx in enumerate(box_order):
        ep_choice = solution[index]  

        original_box = original_boxes[box_idx]
        box = Box(original_box.id, original_box.width, original_box.height, original_box.depth)

        _try_pack_with_ep_choice(heuristic, box, ep_choice)

    occupied_volume = sum(
        b.width * b.height * b.depth for b in test_container.placed_boxes
    )
    return occupied_volume, test_container


def _try_pack_with_ep_choice(heuristic, box, ep_choice):
    heuristic.remove_duplicates()
    heuristic.sort_extreme_points()

    feasible_eps = []
    for ep in heuristic.extreme_points_list:
        for (w, h, d) in box.get_allowed_orientations():
            box.rotate(w, h, d)
            box.place(ep[0], ep[1], ep[2])
            if heuristic.container.can_place_box(box):
                feasible_eps.append((ep, (w, h, d)))
                break  

    if not feasible_eps:
        return False  

    safe_index = min(ep_choice - 1, len(feasible_eps) - 1)
    chosen_ep, chosen_orientation = feasible_eps[safe_index]

    box.rotate(*chosen_orientation)
    box.place(chosen_ep[0], chosen_ep[1], chosen_ep[2])
    heuristic.container.add_box(box)
    heuristic.extreme_points_list.remove(chosen_ep)
    heuristic.generate_new_extreme_points(box)
    return True


def generate_neighbors(solution, element_range, neighborhood_size):
    neighbors = []
    n = len(solution)

    for i in range(n):

        if solution[i] + 1 <= element_range[i]:
            new_sol = solution.copy()
            new_sol[i] = solution[i] + 1
            neighbors.append((i, new_sol[i], new_sol))

        if solution[i] - 1 >= 1:
            new_sol = solution.copy()
            new_sol[i] = solution[i] - 1
            neighbors.append((i, new_sol[i], new_sol))

    if len(neighbors) > neighborhood_size:
        neighbors = random.sample(neighbors, neighborhood_size)

    return neighbors


def run_tabu_search(
    container_width,
    container_height,
    container_depth,
    box_list,
    max_iterations=100,
    max_iterations_without_improvement=15,
    tabu_list_size=20,
    progress_callback=None,
):
    start_time = time.time()

    number_of_boxes = len(box_list)
    container_volume = container_width * container_height * container_depth

    box_order = sorted(range(number_of_boxes),
                       key=lambda i: -(box_list[i].width * box_list[i].height * box_list[i].depth))

    element_range = [max(3, i + 1) for i in range(number_of_boxes)]

    current_solution = [1] * number_of_boxes
    current_fitness, _ = evaluate_solution(
        current_solution, container_width, container_height, container_depth,
        box_list, box_order
    )

    best_solution = current_solution.copy()
    best_fitness = current_fitness

    tabu_list = deque(maxlen=tabu_list_size)

    history = []  # pentru analiza ulterioara
    history.append({
        "iteration": 0,
        "current_fitness": current_fitness,
        "best_fitness": best_fitness,
    })

    if progress_callback is not None:
        progress_callback({
            "iteration": 0,
            "current_fitness": current_fitness,
            "best_fitness": best_fitness,
        })

    neighborhood_size = min(50, number_of_boxes * 2)
    iterations_without_improvement = 0
    stopped_early = False
    iteration = 0

    if best_fitness >= container_volume - VOLUME_TOLERANCE:
        stopped_early = True

    while (iteration < max_iterations
           and iterations_without_improvement < max_iterations_without_improvement
           and not stopped_early):

        iteration += 1

        neighbors = generate_neighbors(current_solution, element_range, neighborhood_size)

        best_neighbor = None
        best_neighbor_fitness = -1
        best_neighbor_move = None

        for (pos, new_val, neighbor_sol) in neighbors:
            move = (pos, new_val)
            neighbor_fitness, _ = evaluate_solution(
                neighbor_sol, container_width, container_height, container_depth,
                box_list, box_order
            )

            is_tabu = move in tabu_list
            if is_tabu and neighbor_fitness <= best_fitness:
                continue

            if neighbor_fitness > best_neighbor_fitness:
                best_neighbor_fitness = neighbor_fitness
                best_neighbor = neighbor_sol
                best_neighbor_move = move

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        current_fitness = best_neighbor_fitness
        tabu_list.append(best_neighbor_move)

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        history.append({
            "iteration": iteration,
            "current_fitness": current_fitness,
            "best_fitness": best_fitness,
        })

        if progress_callback is not None:
            progress_callback({
                "iteration": iteration,
                "current_fitness": current_fitness,
                "best_fitness": best_fitness,
            })

        if best_fitness >= container_volume - VOLUME_TOLERANCE:
            stopped_early = True

    _, final_container = evaluate_solution(
        best_solution, container_width, container_height, container_depth,
        box_list, box_order
    )

    final_order = []
    final_orientations = []
    placed_ids = {b.id: b for b in final_container.placed_boxes}

    for box_idx in box_order:
        original_box = box_list[box_idx]
        final_order.append(box_idx)
        if original_box.id in placed_ids:
            placed = placed_ids[original_box.id]
            # gasim orientarea
            allowed = original_box.get_allowed_orientations()
            placed_dims = (placed.width, placed.height, placed.depth)
            try:
                ori_idx = allowed.index(placed_dims)
            except ValueError:
                ori_idx = 0
            final_orientations.append(ori_idx)
        else:
            final_orientations.append(0)

    execution_time = time.time() - start_time

    return {
        "best_individual": [final_order, final_orientations],
        "best_fitness": best_fitness,
        "best_solution_encoded": best_solution,
        "history": history,
        "config_used": {
            "max_iterations": max_iterations,
            "max_iterations_without_improvement": max_iterations_without_improvement,
            "tabu_list_size": tabu_list_size,
            "neighborhood_size": neighborhood_size,
        },
        "execution_time": execution_time,
        "iterations_run": iteration,
        "stopped_early": stopped_early,
    }