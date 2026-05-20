import random
import math
import copy
from geometry import Container, Box
from constructive import ExtremePointsHeuristic

def evaluate_state(state, container_width, container_height, container_depth, original_boxes):
    test_container = Container(container_width, container_height, container_depth)
    heuristic = ExtremePointsHeuristic(test_container)
    
    order = state[0]
    orientations = state[1]
    
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
    return occupied_volume


def generate_neighbor(current_state):
    neighbor_order = list(current_state[0])
    neighbor_orientations = list(current_state[1])
    
    # Alegem aleator 50-50 dacă schimbăm ordinea SAU orientarea
    if random.random() < 0.5:
        # Inversăm două cutii în ordinea de introducere
        index1, index2 = random.sample(range(len(neighbor_order)), 2)
        neighbor_order[index1], neighbor_order[index2] = neighbor_order[index2], neighbor_order[index1]
    else:
        # Schimbăm orientarea unei cutii aleatoare
        index = random.randint(0, len(neighbor_orientations) - 1)
        neighbor_orientations[index] = random.randint(0, 5)
        
    return [neighbor_order, neighbor_orientations]


def run_simulated_annealing(container_width, container_height, container_depth, box_list, 
                            initial_temperature=1000.0, cooling_rate=0.95, 
                            stopping_temperature=0.1, iterations_per_temperature=20):
    
    number_of_boxes = len(box_list)
    
    current_order = list(range(number_of_boxes))
    random.shuffle(current_order)
    current_orientations = [random.randint(0, 5) for _ in range(number_of_boxes)]
    
    current_state = [current_order, current_orientations]
    current_fitness = evaluate_state(current_state, container_width, container_height, container_depth, box_list)
    
    best_state = copy.deepcopy(current_state)
    best_fitness = current_fitness
    
    current_temperature = initial_temperature
    logbook = [] 
    
    while current_temperature > stopping_temperature:
        for _ in range(iterations_per_temperature):
            neighbor_state = generate_neighbor(current_state)
            neighbor_fitness = evaluate_state(neighbor_state, container_width, container_height, container_depth, box_list)
            
            delta_fitness = neighbor_fitness - current_fitness
            
            if delta_fitness > 0:
                current_state = neighbor_state
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best_state = copy.deepcopy(current_state)
                    best_fitness = current_fitness
            else:
                probability_to_accept = math.exp(delta_fitness / current_temperature)
                if random.random() < probability_to_accept:
                    current_state = neighbor_state
                    current_fitness = neighbor_fitness
                    
        logbook.append({'temperature': current_temperature, 'best_fitness': best_fitness})
        
        current_temperature *= cooling_rate
        
    return best_state, logbook