import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import itertools
import math
from math import ceil, sqrt
import random 

def calculate_total_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i+1]]
    return total_distance

def find_shortest_path_bruteforce(distance_matrix):
    n = len(distance_matrix)
    shortest_path = None
    min_distance = float('inf')

    for path in itertools.permutations(range(n)):
        # Considering cycles, we start and end at the same node
        current_path = list(path).append(path[0])
        current_distance = calculate_total_distance(current_path, distance_matrix)

        if current_distance < min_distance:
            min_distance = current_distance
            shortest_path = current_path

    return shortest_path, min_distance


def insertion(distance_matrix):
    path = [0, 1]

    # Iterate over the remaining nodes
    for new_node in range(2, len(distance_matrix)):
        best_position = 1
        min_increase = float('inf')
        for i in range(1, len(path)):
            increase = (distance_matrix[path[i - 1]][new_node] + 
                        distance_matrix[new_node][path[i]] - 
                        distance_matrix[path[i - 1]][path[i]])
            if (increase < min_increase) | (i == 1):
                min_increase = increase
                best_position = i

        path.insert(best_position, new_node)

    # Close the loop
    path.append(0)
    return path

def nearest_neighbor(distance_matrix, start_node=0):
    n = len(distance_matrix)
    visited = [False] * n
    path = [start_node]
    visited[start_node] = True
    current_node = start_node

    for _ in range(n - 1):
        unvisited_neighbors = [j for j in range(n) if not visited[j]] #just rebuild it every time
        nearest_neighbor = min(unvisited_neighbors, key=lambda x: distance_matrix[current_node][x])
        path.append(nearest_neighbor)
        visited[nearest_neighbor] = True
        current_node = nearest_neighbor
    
    path.append(start_node)  # Returning to the start node
    return path

def read_distance_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read the number of nodes
    num_nodes = int(lines[0].strip())

    # Initialize the distance matrix
    distance_matrix = [[0 for j in range(num_nodes)] for i in range(num_nodes)]

    # Parse each line for distances
    for line in lines[2:]:
        if line.strip() != 'null':
            node1, node2, distance = line.split()
            i, j = int(node1) - 1, int(node2) - 1  # zero indexing
            distance = float(distance)
            distance_matrix[i][j] = distance_matrix[j][i] = distance

    return np.array(distance_matrix) 


def resolve_opt2(path, distance_matrix):
    cycle = 0
    n = len(path) - 1  # Adjust for the closed loop
    improved = True
    #run until no improvemnt is possible
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                # Check the potential improvement with a 2-opt swap
                next_j = j + 1
                if next_j == n:
                    next_j = 0

                old_distance = distance_matrix[path[i]][path[i+1]] + distance_matrix[path[j]][path[next_j]]
                new_distance = distance_matrix[path[i]][path[j]] + distance_matrix[path[i+1]][path[next_j]]
                cycle = cycle + 1
                # Update the path if there's an improvement
                if new_distance < old_distance:
                    #print('before', path)
                    path[i+1:j+1] = reversed(path[i+1:j+1])  # In-place reversal of the segment
                    improved = True
                    #print('after', path)
                    break

            if improved:
                break
    print('opt2 cycles: ', cycle)
    return path



def calculate_total_distance2(tour, distance_matrix):
    total_dist = distance_matrix[tour[-1]][tour[0]]  # Distance from last to first node for a cycle
    for i in range(len(tour) - 1):
        total_dist += distance_matrix[tour[i]][tour[i + 1]]
    return total_dist

def simulated_annealing(distance_matrix, current_path=None):
    n = len(distance_matrix)

    # use a random path if none is provided
    if (current_path == None):
        current_path = list(range(n))
        random.shuffle(current_path)
    
    current_distance = calculate_total_distance2(current_path, distance_matrix)
    best_path = current_path.copy()
    best_distance = current_distance
    
    T = 1.0  # Initial temperature
    T_min = 0.00001  # Minimum temperature
    alpha = 0.995  # Rate of temperature decrease

    while T > T_min:
        for i in range(100):  # Number of iterations at each temperature
            new_path = current_path.copy()
            a, b = random.sample(range(n), 2)
            new_path[a], new_path[b] = new_path[b], new_path[a] #swap two nodes at random
            #random.shuffle(new_path)

            new_distance = calculate_total_distance2(new_path, distance_matrix)
            delta = new_distance - current_distance
            
            if (delta < 0) or (math.exp(-delta / T) > random.random()):
                current_path = new_path
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance
        #print percentage of completion
        
        T *= alpha  # Cool down
    best_path.append(best_path[0]) 
    return best_path, best_distance


file_path = '1000_randomDistance.txt'
distances_m = read_distance_matrix(file_path)
#distances_s = distances_m[0:10, 0:10]

# mds = MDS(n_components=2, dissimilarity='precomputed', random_state=6)
# coords = mds.fit_transform(distances_m)  # Slicing the first 10 nodes


path1 = nearest_neighbor(distances_m)
path2 = resolve_opt2(path1.copy(), distances_m)
path3 = insertion(distances_m)


best_distance = float('inf')
best_path = None
for _ in range(1):
    path_a, distance_a = simulated_annealing(distances_m)
    print(distance_a)
    if distance_a < best_distance:
        print('better path found', distance_a, path_a)
        best_distance = distance_a
        best_path = path_a



print('NN ', calculate_total_distance(path1, distances_m))
print(path1)
print('NN with opt2 ', calculate_total_distance(path2, distances_m))
print(path2)
print('Insertion ', calculate_total_distance(path3, distances_m))
print(path3)
print('Simulated Annealing ', calculate_total_distance(best_path, distances_m))
print(best_path)

