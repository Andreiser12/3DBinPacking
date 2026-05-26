import random
import time
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools, algorithms
from geometry import Container, Box
from constructive import ExtremePointsHeuristic


if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


VOLUME_TOLERANCE = 1e-6


def create_individual(number_of_boxes):
    order = list(range(number_of_boxes))
    random.shuffle(order)
    orientations = [random.randint(0, 5) for _ in range(number_of_boxes)]
    return creator.Individual([order, orientations])


def evaluate_individual(individual, container_width, container_height, container_depth, original_boxes):
    test_container = Container(container_width, container_height, container_depth)
    heuristic = ExtremePointsHeuristic(test_container)

    order = individual[0]
    orientations = individual[1]

    for index in range(len(order)):
        box_index = order[index]
        orientation_index = orientations[index]

        original_box = original_boxes[box_index]
        box = Box(original_box.id, original_box.width, original_box.height, original_box.depth)

        allowed_orientations = box.get_allowed_orientations()
        safe_orientation_index = orientation_index % len(allowed_orientations)

        width, height, depth = allowed_orientations[safe_orientation_index]
        box.rotate(width, height, depth)

        heuristic.pack_box(box)

    occupied_volume = sum(
        placed_box.width * placed_box.height * placed_box.depth
        for placed_box in test_container.placed_boxes
    )

    return (occupied_volume,)


def custom_crossover(individual1, individual2):
    tools.cxPartialyMatched(individual1[0], individual2[0])
    tools.cxTwoPoint(individual1[1], individual2[1])
    del individual1.fitness.values
    del individual2.fitness.values
    return individual1, individual2


def custom_mutation(individual, probability_order=0.05, probability_orientation=0.05):
    for index in range(len(individual[0])):
        if random.random() < probability_order:
            swap_index = random.randint(0, len(individual[0]) - 1)
            individual[0][index], individual[0][swap_index] = individual[0][swap_index], individual[0][index]

    for index in range(len(individual[1])):
        if random.random() < probability_orientation:
            individual[1][index] = random.randint(0, 5)

    del individual.fitness.values
    return (individual,)


def _compute_solution_metrics(individual, container_width, container_height, container_depth, original_boxes):
    """Calculeaza metrici pentru o solutie: fill%, nr cutii plasate, total cutii."""
    test_container = Container(container_width, container_height, container_depth)
    heuristic = ExtremePointsHeuristic(test_container)

    order = individual[0]
    orientations = individual[1]

    for index in range(len(order)):
        box_index = order[index]
        orientation_index = orientations[index]
        original_box = original_boxes[box_index]
        box = Box(original_box.id, original_box.width, original_box.height, original_box.depth)
        allowed_orientations = box.get_allowed_orientations()
        safe_orientation_index = orientation_index % len(allowed_orientations)
        width, height, depth = allowed_orientations[safe_orientation_index]
        box.rotate(width, height, depth)
        heuristic.pack_box(box)

    occupied_volume = sum(b.width * b.height * b.depth for b in test_container.placed_boxes)
    container_volume = container_width * container_height * container_depth
    fill_percentage = (occupied_volume / container_volume) * 100 if container_volume > 0 else 0

    return {
        "placed_boxes": len(test_container.placed_boxes),
        "total_boxes": len(original_boxes),
        "fill_percentage": fill_percentage,
        "occupied_volume": occupied_volume,
        "container_volume": container_volume,
    }


def run_genetic_algorithm(
    container_width,
    container_height,
    container_depth,
    box_list,
    population_size=50,
    maximum_generations=40,
    crossover_probability=0.7,
    mutation_probability=0.2,
    elitism_ratio=0.0,
    use_parallel=True,
    progress_callback=None,
):
    start_time = time.time()

    number_of_boxes = len(box_list)
    container_volume = container_width * container_height * container_depth

    elitism_count = max(1, int(elitism_ratio * population_size)) if elitism_ratio > 0 else 0

    toolbox = base.Toolbox()

    toolbox.register("individual", create_individual, number_of_boxes=number_of_boxes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_individual,
        container_width=container_width,
        container_height=container_height,
        container_depth=container_depth,
        original_boxes=box_list,
    )
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, probability_order=0.1, probability_orientation=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    executor = None
    if use_parallel:
        executor = ProcessPoolExecutor()
        toolbox.register("map", executor.map)

    stopped_early = False
    stop_reason = "max_iterations"  # default: am rulat tot bugetul
    generations_completed = 0
    total_evaluations = 0

    try:
        population = toolbox.population(n=population_size)

        hof_size = max(1, elitism_count)
        hall_of_fame = tools.HallOfFame(hof_size)

        statistics = tools.Statistics(lambda individual: individual.fitness.values)
        statistics.register("avg", lambda fitness_values: sum(val[0] for val in fitness_values) / len(fitness_values))
        statistics.register("max", lambda fitness_values: max(val[0] for val in fitness_values))
        statistics.register("min", lambda fitness_values: min(val[0] for val in fitness_values))

        logbook = tools.Logbook()
        logbook.header = ["gen", "avg", "max", "min"]

        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        total_evaluations += len(population)

        hall_of_fame.update(population)

        record = statistics.compile(population)
        logbook.record(gen=0, **record)
        if progress_callback is not None:
            progress_callback({
                "gen": 0, "avg": record["avg"], "max": record["max"],
                "min": record["min"], "best_so_far": hall_of_fame[0].fitness.values[0],
            })

        if hall_of_fame[0].fitness.values[0] >= container_volume - VOLUME_TOLERANCE:
            stopped_early = True
            stop_reason = "optimal_fill"
        else:
            for generation in range(1, maximum_generations + 1):
                offspring = algorithms.varAnd(population, toolbox, crossover_probability, mutation_probability)

                invalid = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
                for ind, fit in zip(invalid, fitnesses):
                    ind.fitness.values = fit
                total_evaluations += len(invalid)

                hall_of_fame.update(offspring)

                if elitism_count > 0:
                    selected = toolbox.select(offspring, population_size - elitism_count)
                    elites = [toolbox.clone(ind) for ind in hall_of_fame[:elitism_count]]
                    population[:] = elites + selected
                else:
                    population[:] = toolbox.select(offspring, population_size)

                record = statistics.compile(population)
                logbook.record(gen=generation, **record)
                generations_completed = generation

                if progress_callback is not None:
                    progress_callback({
                        "gen": generation, "avg": record["avg"], "max": record["max"],
                        "min": record["min"], "best_so_far": hall_of_fame[0].fitness.values[0],
                    })

                if hall_of_fame[0].fitness.values[0] >= container_volume - VOLUME_TOLERANCE:
                    stopped_early = True
                    stop_reason = "optimal_fill"
                    break

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    execution_time = time.time() - start_time

    fitness_history = []
    best_so_far = 0
    for rec in logbook:
        best_so_far = max(best_so_far, rec["max"])
        fitness_history.append((rec["gen"], best_so_far))

    # calculam metrici pentru cel mai bun individ
    best_metrics = _compute_solution_metrics(
        hall_of_fame[0], container_width, container_height, container_depth, box_list
    )

    return {
        "best_individual": hall_of_fame[0],
        "best_fitness": hall_of_fame[0].fitness.values[0],
        "logbook": logbook,
        "hall_of_fame": list(hall_of_fame),
        "config_used": {
            "population_size": population_size,
            "maximum_generations": maximum_generations,
            "crossover_probability": crossover_probability,
            "mutation_probability": mutation_probability,
            "elitism_ratio": elitism_ratio,
            "elitism_count": elitism_count,
        },
        "execution_time": execution_time,
        "generations_run": generations_completed,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "total_evaluations": total_evaluations,
        "fitness_history": fitness_history,

        "placed_boxes": best_metrics["placed_boxes"],
        "total_boxes": best_metrics["total_boxes"],
        "fill_percentage": best_metrics["fill_percentage"],
    }