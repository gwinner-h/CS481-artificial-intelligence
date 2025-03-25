"""
Heidi Gwinner
CS-481, Artificial Intelligence
Dr. Farmer, Winter 2025
Programming Assignment 1: Traveling Admin Problem
GraphHandler.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def initGraph():
    """
    Initialize and create a weighted undirected graph representing locations and travel times.
    The graph is constructed from a predefined list of locations (addresses) and a matrix of 
    travel times between these locations.
    Returns:
        nx.Graph: A NetworkX undirected graph where:
            - Nodes represent locations, with indices 0 to n-1
            - Each node has a 'location' attribute containing the full address
            - Edge weights represent travel times in minutes between locations
    """
    locations = np.array([
        "4510 Five Lakes Rd., North Branch",
        "200 North Barrie Rd., Bad Axe",
        "2020 Union Street, Ubly",
        "2203 Wildner Road, Sebewaing Township",
        "7166 Main St., Owendale",
        "6609 Vine Street, Caseville",
        "4868 Seeger St., Cass City",
        "2633 Black River Street, Deckerville",
        "191 East Pinetree Lane, Sandusky",
        "301 North Hooper St., Caro",
        "3051 Moore Street, Marlette",
        "100 North Goetze Road, Carsonville",
        "2800 North Thomas Road, Fairgrove Township",
        "525 East Genesee Street, Frankenmuth",
        "220 Athletic Street, Vassar",
        "6250 Fulton Street, Mayville",
        "8780 Dean Drive, Millington",
        "5461 Peck Road, Croswell",
        "4400 2nd Street, Brown City",
        "247 School Drive, Yale",
        "402 South 5th Street, Harbor Beach",
        "5790 State St., Kingston"
    ])
    travel_times_matrix = np.array([
        [0, 63, 62, 58, 53, 72, 48, 60, 49, 42, 27, 57, 52, 47, 38, 24, 31, 49, 22, 36, 84, 31],
        [63, 0, 16, 33, 19, 25, 24, 41, 46, 44, 40, 53, 48, 72, 60, 56, 67, 68, 55, 67, 25, 39],
        [62, 16, 0, 33, 21, 38, 22, 28, 32, 42, 37, 40, 47, 70, 58, 54, 67, 54, 44, 54, 25, 39],
        [58, 33, 33, 0, 15, 29, 26, 57, 62, 23, 51, 70, 18, 43, 32, 40, 43, 82, 63, 81, 56, 39],
        [53, 19, 21, 15, 0, 21, 15, 45, 51, 29, 40, 58, 32, 57, 45, 41, 53, 70, 54, 70, 40, 33],
        [72, 25, 38, 29, 21, 0, 33, 62, 68, 45, 58, 75, 45, 70, 61, 60, 68, 87, 71, 87, 44, 51],
        [48, 24, 22, 26, 15, 33, 0, 35, 40, 25, 30, 49, 28, 52, 40, 36, 49, 60, 44, 60, 45, 25],
        [60, 41, 28, 57, 45, 62, 35, 0, 18, 46, 36, 15, 52, 73, 60, 54, 68, 34, 41, 42, 30, 36],
        [49, 46, 32, 62, 51, 68, 40, 18, 0, 46, 22, 14, 55, 61, 49, 43, 55, 27, 28, 27, 46, 26],
        [42, 44, 42, 23, 29, 45, 25, 46, 46, 0, 37, 58, 16, 34, 22, 24, 29, 68, 48, 65, 66, 23],
        [27, 40, 37, 51, 40, 58, 30, 36, 22, 37, 0, 32, 46, 49, 37, 24, 36, 34, 17, 33, 59, 16],
        [57, 53, 40, 70, 58, 75, 49, 15, 14, 58, 32, 0, 63, 71, 58, 53, 65, 22, 38, 38, 38, 36],
        [52, 48, 47, 18, 32, 45, 28, 52, 55, 16, 46, 63, 0, 31, 19, 31, 30, 76, 56, 76, 69, 33],
        [47, 72, 70, 43, 57, 70, 52, 73, 61, 34, 49, 71, 31, 0, 14, 30, 22, 74, 55, 73, 92, 37],
        [38, 60, 58, 32, 45, 61, 40, 60, 49, 22, 37, 58, 19, 14, 0, 18, 14, 67, 46, 65, 80, 25],
        [24, 56, 54, 40, 41, 60, 36, 54, 43, 24, 24, 53, 31, 30, 18, 0, 19, 52, 32, 50, 77, 20],
        [31, 67, 67, 43, 53, 68, 49, 68, 55, 29, 36, 65, 30, 22, 14, 19, 0, 58, 39, 57, 87, 32],
        [49, 68, 54, 82, 70, 87, 60, 34, 27, 68, 34, 22, 76, 74, 67, 52, 58, 0, 28, 27, 51, 47],
        [22, 55, 44, 63, 54, 71, 44, 41, 28, 48, 17, 38, 56, 55, 46, 32, 39, 28, 0, 21, 67, 30],
        [36, 67, 54, 81, 70, 87, 60, 42, 27, 65, 33, 38, 76, 73, 65, 50, 57, 27, 21, 0, 68, 47],
        [84, 25, 25, 56, 40, 44, 45, 30, 46, 66, 59, 38, 69, 92, 80, 77, 87, 51, 67, 68, 0, 60],
        [31, 39, 39, 39, 33, 51, 25, 36, 26, 23, 16, 36, 33, 37, 25, 20, 32, 47, 30, 47, 60, 0],
    ])

    # Create a weighted graph
    graph = nx.Graph()

    # Add nodes with locations as attributes
    for i, location in enumerate(locations):
        graph.add_node(i, location=location)

    # Add edges with travel times as weights
    for i in range(len(travel_times_matrix)):
        for j in range(i + 1, len(travel_times_matrix)):
            graph.add_edge(i, j, weight=travel_times_matrix[i][j])

    return graph



def drawGraph(graph):
    """
    Visualizes a networkx graph using matplotlib.
    This function creates a visual representation of a graph using a spring layout,
    displaying nodes, edges, node labels, and edge weights.
    Args:
        graph (nx.Graph): A networkx graph object to be visualized.
    """
    pos = nx.spring_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_size=700)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    # edge weight labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()



def sum_node_edge_weights(graph, nodes):
    """
    Calculate the total weight of the edges between a sequence of nodes in a graph.
    Args:
        nodes (list): A list of node indices representing a path in the graph.
    Returns:
        int: The total weight of the edges along the path defined by the nodes list.
    """
    total_weight = 0
    # Sum the weight of the edge between each given node
    for i, node_index in enumerate(nodes[:-1]):
        total_weight += graph[node_index][nodes[i + 1]]['weight']

    return total_weight