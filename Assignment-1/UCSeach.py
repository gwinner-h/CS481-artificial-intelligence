"""
Heidi Gwinner
CS-481, Artificial Intelligence
Dr. Farmer, Winter 2025
Programming Assignment 1: Traveling Admin Problem
NNSearch.py
"""

from GraphHandler import *
import heapq

DATA_GRAPH = initGraph()
STARTING_INDEX = 0

day_1_route = [STARTING_INDEX]
day_1_distance = 0
day_2_route = [STARTING_INDEX]
day_2_distance = 0

# Precompute the smallest edge weight in the graph for use in heuristic
smallest_edge_weight = float("inf")
for u, v, weight in DATA_GRAPH.edges(data='weight'):
    if weight < smallest_edge_weight:
        smallest_edge_weight = weight


def heuristic(current_path, total_steps):
    """
    Calculate the heuristic value for the remaining steps in a path by 
    multiplying the smallest edge on the graph by the remaining number
    of steps to complete a path. This is an admissible heuristic.

    Args:
        current_path (list): The current path being evaluated
        total_steps (int): The total number of steps required for a complete path

    Returns:
        float: The heuristic value representing the estimated minimum cost of completing the path
    """
    remaining_steps = total_steps - len(current_path)
    return remaining_steps * smallest_edge_weight


def find_optimal_circuit_of_length(graph, target_length):
    """
    Find the optimal (minimum weight) circuit (round-trip path) of a specified length in 
    a weighted graph. This function implements an A* search algorithm to find the shortest circuit  
    of a given length in a weighted graph. The circuit must start and end at the same node.
    Args:
        graph (networkx.Graph): An undirected weighted graph where the circuit will be found.
                               The graph must have edge weights.
        target_length (int): The desired number of nodes in the path (excluding the return to start).
                            Must be at least 2 and no more than the number of nodes in the graph.
    Returns:
        tuple: A tuple containing:
            - list: The optimal circuit as a list of nodes, including the return to start node
            - float: The total weight of the optimal circuit
    Raises:
        ValueError: If target_length is greater than the number of available nodes or less than 2.
    Notes:
        - The function uses A* search with a heuristic function to prune the search space
        - The returned path always starts and ends with the same node (first node in graph)
        - The function assumes the graph is connected and undirected
    """
    node_list = list(graph.nodes)
    if target_length > len(node_list):
        raise ValueError(f"There are not enough nodes to make a path of length {target_length}.")
    if target_length <= 1:
        raise ValueError("You must provide a graph with at least two nodes to create a valid path.")

    start = node_list[0]
    # Priority queue items are (minimum_estimated_total_cost, vertex, path)
    priority_queue = []

    # Initialize best path with a large weight
    best_path = []
    best_weight = float("inf")
    
    # Push the initial (minimum_estimated_total_cost, vertex, path) onto the priority queue
    initial_heuristic = heuristic([start], target_length)
    heapq.heappush(priority_queue, (initial_heuristic, start, [start]))
    
    while priority_queue:
        min_total_cost, vertex, path = heapq.heappop(priority_queue)

        # Prune paths that are already worse than the best found so far
        if min_total_cost >= best_weight:
            continue

        # Add neighbor nodes to path until desired length is reached
        for next_node in set(graph.neighbors(vertex)):
            if next_node not in path:
                new_path = path + [next_node]
                new_path_length = len(new_path)

                # End search once a path longer than the target is checked
                if new_path_length > target_length:
                    continue
                else: new_path_weight = sum_node_edge_weights(graph, new_path)
                
                # If path reaches target length, append the starting node
                # to complete a round-trip and compare to best known path
                if new_path_length == target_length:
                    final_path = new_path + [start]
                    final_path_weight = sum_node_edge_weights(graph, final_path)
                    
                    # Update best_path
                    if final_path_weight < best_weight:
                        best_path = final_path
                        best_weight = final_path_weight
                else:
                    # Add heuristic estimate to the path weight for priority queue
                    min_total_cost = new_path_weight + heuristic(new_path, target_length)
                    
                    # Only add to queue if this path might be better than the best known
                    if min_total_cost < best_weight:
                        heapq.heappush(priority_queue, (min_total_cost, next_node, new_path))
    
    return best_path, best_weight



# Find the optimal path from the starting point back to the starting point
# containing exactly half of the available non-home nodes
num_nodes = DATA_GRAPH.number_of_nodes()
day_1_route, day_1_distance = find_optimal_circuit_of_length(DATA_GRAPH, num_nodes / 2)

# Remove visited nodes from the graph and find the optimal circuit with containing all remaining nodes
data_copy = DATA_GRAPH.copy() # Create a copy of the graph to preserve the original
data_copy.remove_nodes_from(day_1_route[1:-1])  # Remove all nodes except starting point
num_nodes = data_copy.number_of_nodes() # Recalculate node count after removals
day_2_route, day_2_distance = find_optimal_circuit_of_length(data_copy, num_nodes)


print("Uniform Cost Traversal Results")
print("----------------------------------")
print(f"First Day Route: \n{day_1_route}")
print()
print(f"Second Day Route: \n{day_2_route}")

print()
print("     Routes by Location Names     ")
print("----------------------------------")
print("First Day Route:")
for i, node_index in enumerate(day_1_route):
    print(f"{i + 1}. {DATA_GRAPH.nodes[node_index]['location']}")
print(f"First Day Distance Traveled (Including Trip Home): {day_1_distance}")
print()
print("Second Day Route:")
for i, node_index in enumerate(day_2_route):
    print(f"{i + 1}. {DATA_GRAPH.nodes[node_index]['location']}")
print(f"Second Day Distance Traveled (Including Trip Home): {day_2_distance}")