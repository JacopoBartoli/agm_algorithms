import networkx as nx
from networkx import Graph
import numpy as np
import json
import os
import timeit
import math
from tqdm import tqdm

DATA_FOLDER = "data"
FILE_JSON_DATA = os.path.join(DATA_FOLDER, "dpc-covid19-ita-province.json")
FILE_GRAPH_P = os.path.join(DATA_FOLDER, "graph_P.yml")
FILE_GRAPH_R = os.path.join(DATA_FOLDER, "graph_R.yml")

LOAD_GRAPHS_FROM_FILES = True  # Flag to load graphs from saved file instead of build them from JSON data


def counting_triangles(graph):
    nodes = list(graph.nodes)
    triangle_count = 0
    for v in nodes:
        for u in graph.neighbors(v):
            for w in graph.neighbors(v):
                if u != w and w in graph.neighbors(u):
                    triangle_count = triangle_count + 1
    return triangle_count / 6


def counting_triangles_ldv(graph: Graph) -> int:
    count_triangles = 0
    for v, nbrs in graph.adjacency():
        v_degree = graph.degree[v]

        list_nbrs = list(nbrs.keys())
        n_nbrs = len(list_nbrs)
        for i in range(n_nbrs):
            u = list_nbrs[i]
            u_degree = graph.degree[u]
            if u_degree > v_degree or (u_degree == v_degree and v < u):
                for j in range(i + 1, n_nbrs):
                    w = list_nbrs[j]
                    w_degree = graph.degree[w]
                    if w_degree > v_degree or (w_degree == v_degree and v < w):
                        # If it exists and edge connecting u and w we found a triangle
                        if u in graph.adj[w]:
                            count_triangles += 1

    return count_triangles


def clustering_coefficients(graph) -> dict:
    nodes = list(graph.nodes)
    coefficients = {}
    for v in graph.nodes:
        v_neighbors = set(graph.neighbors(v))
        v_neighbors_len = len(v_neighbors)

        if v_neighbors_len > 1:
            n_triangles = 0
            for u in v_neighbors:
                n_triangles += len(v_neighbors.intersection(graph.neighbors(u)))

            coeff = n_triangles / (v_neighbors_len * (v_neighbors_len - 1))
            coefficients[v] = coeff
        else:
            coefficients[v] = 0

    return coefficients


def build_graphs() -> tuple:
    print("Reading JSON data...")
    # Read the data. Each item of the datastore array represent a province.
    with open(FILE_JSON_DATA) as f:
        datastore = json.load(f)

    # Clean data: Discharge the provinces where the field "sigla_provincia" is null (None) (Those data are not available yet)
    datastore = [data for data in datastore if data["sigla_provincia"] is not None]

    print("JSON data loaded")

    graph_P = build_graph_P(datastore)
    graph_R = build_graph_R()

    print("Saving graphs files...")
    nx.write_yaml(graph_P, FILE_GRAPH_P)
    nx.write_yaml(graph_R, FILE_GRAPH_R)
    print("Files successfully saved")

    return graph_P, graph_R


def build_graph_P(datastore) -> Graph:
    """
    Creates the graph of provinces (graph P)
    :param datastore:
    :return:
    """

    print("Building graph P...")

    max_distance = 0.8
    graph = nx.Graph()

    # Add nodes
    for data in datastore:
        graph.add_node(data["codice_provincia"], **data)

    # Add edges -> Complexity O(n^2)
    n_provinces = len(datastore)
    for i in tqdm(range(n_provinces)):
        data_i = datastore[i]
        for j in range(i + 1, n_provinces):
            data_j = datastore[j]
            # There is an edge between two nodes/provinces if their coordinates differ for less than max_distance
            if abs(data_i["lat"] - data_j["lat"]) <= max_distance and abs(
                    data_i["long"] - data_j["long"]) <= max_distance and data_i["codice_provincia"] != data_j["codice_provincia"]:
                graph.add_edge(data_i["codice_provincia"], data_j["codice_provincia"])

    print(f"Graph P built. Nodes: {graph.number_of_nodes()} - Edges: {graph.number_of_edges()}")

    return graph


def build_graph_R() -> Graph:
    """
    Creates the graph using random pairs (x, y) of doubles (graph R)
    :return:
    """

    print("Building graph R...")

    n_nodes = 2000
    max_distance = 0.08
    graph = nx.Graph()

    # Generate 2000 random pairs (x,y) of doubles with x in[30,50) and y in [10,20)
    list_pairs = [(i, np.random.uniform(30, 50), np.random.uniform(10, 20)) for i in range(n_nodes)]

    # Add nodes
    for key, x, y in list_pairs:
        graph.add_node(key, x=x, y=y)

    # Add edges
    for i in range(n_nodes):
        node_i = list_pairs[i]
        for j in range(i + 1, n_nodes):
            node_j = list_pairs[j]
            # There is an edge between two nodes if their coordinates differ for less than max_distance
            if abs(node_i[1] - node_j[1]) <= max_distance and abs(node_i[2] - node_j[2]) <= max_distance:
                graph.add_edge(node_i[0], node_j[0])

    print(f"Graph R built. Nodes: {graph.number_of_nodes()} - Edges: {graph.number_of_edges()}")

    return graph


def build_graph_test() -> Graph:
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5])
    g.add_edges_from([(0, 1), (0, 3), (0, 4), (0, 5), (1, 4), (1, 2), (2, 4), (2, 5), (3, 4), (4, 5)])
    return g


def test_counting_triangles(dict_graphs: dict):
    print("\n\n--- TEST COUNTING TRIANGLES ---")

    time_evaluations = 1000
    for graph_name, graph in dict_graphs.items():
        print(f"\nEvaluating graph {graph_name}...")

        our_time = math.inf
        nx_time = math.inf
        for i in range(time_evaluations):
            start = timeit.default_timer()
            counting_triangles_ldv(graph)
            end = timeit.default_timer()
            our_time = min(end - start, our_time)

            start = timeit.default_timer()
            sum(nx.triangles(graph).values()) / 3
            end = timeit.default_timer()
            nx_time = min(end - start, nx_time)

        print(" - Triangles counted")
        print(f"\tOurs: {counting_triangles_ldv(graph)}")
        print(f"\tNetworkX: {sum(nx.triangles(graph).values()) / 3}")
        print(" - Execution times")
        print(f"\tOurs: {our_time * 1000} ms")
        print(f"\tNetworkX: {nx_time * 1000} ms")


def test_clustering_coefficients(dict_graphs: dict):
    print("\n\n--- TEST CLUSTERING COEFFICIENTS ---")

    time_evaluations = 1000
    for graph_name, graph in dict_graphs.items():
        print(f"\nEvaluating graph {graph_name}...")

        our_time = math.inf
        nx_time = math.inf
        for i in range(time_evaluations):
            start = timeit.default_timer()
            clustering_coefficients(graph)
            end = timeit.default_timer()
            our_time = min(end - start, our_time)

            start = timeit.default_timer()
            nx.clustering(graph)
            end = timeit.default_timer()
            nx_time = min(end - start, nx_time)

        print(" - Coefficients calculated")
        print(f"\tOurs: {clustering_coefficients(graph)}")
        print(f"\tNetworkX: {nx.clustering(graph)}")
        print(" - Execution times")
        print(f"\tOurs: {our_time * 1000} ms")
        print(f"\tNetworkX: {nx_time * 1000} ms")


def test():
    # Test on a handcrafted graph
    g = nx.Graph()
    g.add_node("1")
    g.add_node("2")
    g.add_node("3")
    g.add_node("4")
    g.add_node("5")
    g.add_edge("1", "2")
    g.add_edge("2", "3")
    g.add_edge("3", "1")
    g.add_edge("2", "4")
    g.add_edge("3", "4")
    g.add_edge("3", "5")
    g.add_edge("4", "5")
    print(counting_triangles(g))
    print(counting_triangles_ldv(g))
    print(sum(nx.triangles(g).values()) / 3)

    print("\n----\n")

    g = nx.complete_graph(5)
    print(counting_triangles(g))
    print(counting_triangles_ldv(g))
    print(sum(nx.triangles(g).values()) / 3)

    nx.clustering(g)


if __name__ == '__main__':
    # function()
    if LOAD_GRAPHS_FROM_FILES:
        print("Loading graphs from files...")
        graph_P = nx.read_yaml(FILE_GRAPH_P)
        graph_R = nx.read_yaml(FILE_GRAPH_R)
        print("Graphs loaded")
    else:
        graph_P, graph_R = build_graphs()

    graph_test = build_graph_test()

    dict_graphs = {"P": graph_P, "R": graph_R, "Test": graph_test}

    test_counting_triangles(dict_graphs)
    test_clustering_coefficients(dict_graphs)
