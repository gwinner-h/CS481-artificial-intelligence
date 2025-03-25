"""
Heidi Gwinner
CS-481, Artificial Intelligence
Dr. Farmer, Winter 2025
Programming Assignment 1: Traveling Admin Problem
NNSearch.py
"""

from GraphHandler import *

DATA_GRAPH = initGraph()
STARTING_INDEX = 0

day_1_route = [STARTING_INDEX]
day_1_distance = 0
day_2_route = [STARTING_INDEX]
day_2_distance = 0
num_nodes = DATA_GRAPH.number_of_nodes()


def get_visited_nodes():
    """
    Provides the union of node indexes in each day's route.

    Returns:
        list: A list of unique nodes visited over the two days.
    """
    return list(set(day_1_route + day_2_route))


def get_unvisited_neighbors(node_index):
    """
    Provides the indexes of traversable neighbor nodes not present in either route.

    Args:
        node_index (int): The index of the node for which to find unvisited neighbors.

    Returns:
        list: A list of unvisited neighbor nodes.
    """
    visited_nodes = get_visited_nodes()
    connected_nodes = list(DATA_GRAPH.neighbors(node_index))
    unvisited_nodes = [node for node in connected_nodes if node not in visited_nodes]
    return unvisited_nodes


def get_nearest_unvisited_neighbor(node_index):
    """
    Finds the nearest unvisited neighbor of a given node in a graph.

    Args:
        node_index (int): The index of the current node.

    Returns:
        int: The index of the nearest unvisited neighbor node.
    """
    unvisited_neighbors = get_unvisited_neighbors(node_index)
    nearest_neighbor = min(unvisited_neighbors, key=lambda node: DATA_GRAPH[node_index][node]['weight'])
    return nearest_neighbor


# Build the route for day 1 using NN traversal,
# stopping once half of all nodes have been visited
while len(day_1_route) < num_nodes / 2:
    day_1_route.append(get_nearest_unvisited_neighbor(day_1_route[-1]))
# Get the day's total travelled distance
day_1_route.append(STARTING_INDEX)
day_1_distance += sum_node_edge_weights(DATA_GRAPH, day_1_route)

# Build the route for day 2 using NN traversal,
# stopping once there are no nodes left to traverse
while len(get_visited_nodes()) < num_nodes:
    day_2_route.append(get_nearest_unvisited_neighbor(day_2_route[-1]))
# Get the day's total travelled distance
day_2_route.append(STARTING_INDEX)
day_2_distance += sum_node_edge_weights(DATA_GRAPH, day_2_route)


print("Nearest Neighbor Traversal Results")
print("----------------------------------")
print(f"First Day Route: \n{day_1_route}")
for i, node_index in enumerate(day_1_route):
    print(f"{i + 1}. {DATA_GRAPH.nodes[node_index]['location']}")
print(f"First Day Distance Traveled (Including Trip Home): {day_1_distance}")
print()
print(f"Second Day Route: \n{day_2_route}")
for i, node_index in enumerate(day_2_route):
    print(f"{i + 1}. {DATA_GRAPH.nodes[node_index]['location']}")
print(f"Second Day Distance Traveled (Including Trip Home): {day_2_distance}")
print()
