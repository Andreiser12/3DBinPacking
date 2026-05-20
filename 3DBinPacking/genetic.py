import random
import copy
from deap import base, creator, tools, algorithms
from geometry import Container, Box
from constructive import ExtremePointsHeuristic

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    
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
        
    occupied_volume = sum(placed_box.width * placed_box.height * placed_box.depth for placed_box in test_container.placed_boxes)
    
    return (occupied_volume, )


def custom_crossover(individual1, individual2):
    tools.cxPartialyMatched(individual1[0], individual2[0])
    tools.cxTwoPoint(individual1[1], individual2[1])
    return individual1, individual2


def custom_mutation(individual, probability_order=0.05, probability_orientation=0.05):
    for index in range(len(individual[0])):
        if random.random() < probability_order:
            swap_index = random.randint(0, len(individual[0]) - 1)
            individual[0][index], individual[0][swap_index] = individual[0][swap_index], individual[0][index]
            
    for index in range(len(individual[1])):
        if random.random() < probability_orientation:
            individual[1][index] = random.randint(0, 5)
            
    return individual,


def run_genetic_algorithm(container_width, container_height, container_depth, box_list, population_size=50, maximum_generations=40):
    number_of_boxes = len(box_list)
    
    toolbox = base.Toolbox()
    
    toolbox.register("individual", create_individual, number_of_boxes=number_of_boxes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_individual, container_width=container_width, container_height=container_height, container_depth=container_depth, original_boxes=box_list)
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, probability_order=0.1, probability_orientation=0.1)
    
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population = toolbox.population(n=population_size)
    
    hall_of_fame = tools.HallOfFame(1)
    
    statistics = tools.Statistics(lambda individual: individual.fitness.values)
    statistics.register("avg", lambda fitness_values: sum(val[0] for val in fitness_values) / len(fitness_values))
    statistics.register("max", lambda fitness_values: max(val[0] for val in fitness_values))
    
    crossover_probability = 0.7
    mutation_probability = 0.2
    
    population, logbook = algorithms.eaSimple(population, toolbox, 
                                              cxpb=crossover_probability, 
                                              mutpb=mutation_probability, 
                                              ngen=maximum_generations, 
                                              stats=statistics, 
                                              halloffame=hall_of_fame, 
                                              verbose=True)
    
    best_individual = hall_of_fame[0]
    
    return best_individual, logbook