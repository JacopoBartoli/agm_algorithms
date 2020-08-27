import matplotlib.pyplot as plt
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
TIME_EVALUATIONS = 100


# ------------------------------------------------------------
#       COUNTING TRIANGLES AND CLUSTERING COEFF. SECTION
# ------------------------------------------------------------

def counting_triangles_naive(graph: Graph) -> int:
    """
    Implementation of the algorithm for counting triangles using the naive (brute-force) approach
    """

    nodes = list(graph.nodes)
    n_nodes = len(nodes)

    count_triangles = 0
    for i in range(n_nodes):
        v = nodes[i]
        for j in range(i+1, n_nodes):
            u = nodes[j]
            for k in range(j+1, n_nodes):
                w = nodes[k]
                if u in graph.adj[v] and w in graph.adj[v] and w in graph.adj[u]:
                    count_triangles += 1

    return count_triangles


def counting_triangles_nbrs_pairs(graph: Graph) -> int:
    """
    Implementation of the algorithm for counting triangles using the Enumerating of Neighbors Pairs approach
    """
    count_triangles = 0
    for v in graph.nodes:
        list_nbrs = list(graph.neighbors(v))
        n_nbrs = len(list_nbrs)
        for i in range(n_nbrs):
            u = list_nbrs[i]
            for j in range(i+1, n_nbrs):
                w = list_nbrs[j]
                if w in graph.adj[u]:
                    count_triangles += 1
    return int(count_triangles/3)


def counting_triangles_ldv(graph: Graph) -> int:
    """
    Implementation of the algorithm for counting triangles using the Low-Degree Vertices approach
    """
    count_triangles = 0
    for v in graph.nodes:
        v_degree = graph.degree[v]

        list_nbrs = list(graph.neighbors(v))
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


def clustering_coefficients_set_intersection(graph: Graph) -> dict:
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


def clustering_coefficients_list_intersection(graph: Graph) -> dict:
    coefficients = {}
    for v in graph.nodes:
        v_neighbors = list(graph.neighbors(v))
        v_neighbors_len = len(v_neighbors)

        if v_neighbors_len > 1:
            n_triangles = 0
            for u in v_neighbors:
                u_neighbors = list(graph.neighbors(u))
                intersection = get_list_intersection(v_neighbors, u_neighbors)
                n_triangles += len(intersection)

            coeff = n_triangles / (v_neighbors_len * (v_neighbors_len - 1))
            coefficients[v] = coeff
        else:
            coefficients[v] = 0

    return coefficients


def get_list_intersection(l1: list, l2: list) -> list:
    """
    Calculates the intersection between two lists
    Complexity: O(|l1|log|l1| + |l2|log|l2|)
    """
    l1 = sorted(l1)
    l2 = sorted(l2)

    i = 0
    j = 0
    intersection = []
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            i += 1
        elif l1[i] > l2[j]:
            j += 1
        else:
            intersection.append(l1[i])
            i += 1
            j += 1
    return intersection


# ----------------------------------
#       GRAPH BUILDING SECTION
# ----------------------------------

def build_graphs() -> tuple:
    graph_P = build_graph_P()
    graph_R = build_graph_R()

    plot_graph(graph_P, "graph_P", "spring", True)
    plot_graph(graph_R, "graph_R", "random", False)

    return graph_P, graph_R


def plot_graph(graph: Graph, name: str, layout: str = "spring", draw_labels: bool = True):

    if layout == "spring":
        pos = nx.layout.spring_layout(graph)
    elif layout == "random":
        pos = nx.layout.random_layout(graph)

    plt.subplots(1, 1, figsize=(8, 8))
    edge_colors = [d*1.5 for d in nx.get_edge_attributes(graph, "weight").values()]
    edge_cmap = plt.cm.YlOrRd
    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color="#008ae6", alpha=0.8)
    if draw_labels:
        labels = nx.get_node_attributes(graph, "label")
        nx.draw_networkx_labels(graph, pos, labels, font_size=9)
    # nx.draw_networkx_labels(graph, pos, font_size=8, font_family="sans-serif", ax=ax)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors,
                                   edge_cmap=edge_cmap,
                                   width=2)

    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=min(nx.get_edge_attributes(graph, "weight").values()),
                                                                  vmax=max(nx.get_edge_attributes(graph, "weight").values())))

    sm._A = []
    plt.colorbar(sm)
    plt.axis('off')
    plt.savefig(os.path.join(DATA_FOLDER, name + ".png"))


def load_data_from_json() -> list:
    """
    Load and clean from the JSON file data about COVID cases for each province.
    After loading the file, it performs some cleaning operations to take only the data relative to the 107 italian provinces and
    to the latest day available
    """

    print("Reading JSON data...")
    # Read the data. Each item of the datastore array represent a province.
    with open(FILE_JSON_DATA) as f:
        datastore = json.load(f)

    # Clean data: Filter data relative to the latest day available
    latest_day = datastore[-1]["data"]
    datastore = [data for data in datastore if data["data"] == latest_day]

    # Clean data: Discard the data where the field "lat" is null (None)
    # Those data refer to COVID cases of people that are currently out of their region of residence or cases of people not localized yet
    # We are only interested into provinces data
    datastore = [data for data in datastore if data["lat"] is not None]

    print("JSON data loaded")

    return datastore


def build_graph_P() -> Graph:
    """
    Creates the graph of provinces (graph P)
    """

    print("Building graph P...")

    max_distance = 0.8
    graph = generate_graph_P_nodes()
    add_edges_sorting(graph, max_distance)
    add_weights(graph)

    print(f"Graph P built. Nodes: {graph.number_of_nodes()} - Edges: {graph.number_of_edges()}")

    return graph


def build_graph_R() -> Graph:
    """
    Creates a graph using random pairs (x, y) of doubles (graph R)
    """

    print("Building graph R...")

    n_nodes = 2000
    max_distance = 0.08
    graph = generate_graph_R_nodes(n_nodes)
    add_edges_sorting(graph, max_distance)
    add_weights(graph)

    print(f"Graph R built. Nodes: {graph.number_of_nodes()} - Edges: {graph.number_of_edges()}")

    return graph


def generate_graph_P_nodes() -> Graph:
    """
    Creates a graph of provinces with only nodes (without edges) starting from the JSON file.
    Required to avoid code duplication and to easy test the performance of the graph generation algorithms.
    """
    datastore = load_data_from_json()

    graph = nx.Graph()
    for data in datastore:
        graph.add_node(data["codice_provincia"], x=data["lat"], y=data["long"], label=data["sigla_provincia"])

    return graph


def generate_graph_R_nodes(n_nodes: int) -> Graph:
    """
    Creates a random graph with only nodes (without edges) starting from the JSON file.
    Required to avoid code duplication and to easy test the performance of the graph generation algorithms.
    """
    graph = nx.Graph()

    # Generate n_nodes random pairs (x,y) of doubles with x in[30,50) and y in [10,20)
    list_pairs = [(i, np.random.uniform(30, 50), np.random.uniform(10, 20)) for i in range(n_nodes)]
    for key, x, y in list_pairs:
        graph.add_node(key, x=x, y=y, label=key)

    return graph


def add_edges_naive(graph: Graph, max_distance: float):
    """
    Add edges between each pair of nodes in the graph if their distance along both axis (x, y) is less than max_distance.
    It compares all the possible pairs of nodes
    Complexity: O(n^2) with n = number of nodes
    """

    nodes = list(graph.nodes(data=True))
    n_nodes = len(nodes)

    for i in range(n_nodes):
        v_id, v_attr = nodes[i]
        for j in range(i + 1, n_nodes):
            u_id, u_attr = nodes[j]
            # There is an edge between two nodes if their coordinates differ for less than max_distance
            if abs(v_attr["x"] - u_attr["x"]) <= max_distance and abs(v_attr["y"] - u_attr["y"]) <= max_distance:
                graph.add_edge(v_id, u_id)


def add_edges_sorting(graph: Graph, max_distance: float):
    """
    Add edges between each pair of nodes in the graph if their distance along both axis (x, y) is less than max_distance.
    It sorts the list of nodes to reduce the computational cost.
    Complexity: O(n*log(n) + m) with n = number of nodes and m = number of edges
    """

    nodes = list(graph.nodes(data=True))
    nodes.sort(key=lambda n: n[1]['x'])
    n_nodes = len(nodes)

    for i in range(n_nodes):
        v_id, v_attr = nodes[i]

        # Check if there must be an edge between v and the previous nodes in the list
        # If a node distant more than max_distance along x is found stop the loop (the subsequent nodes cannot be connected to v)
        end_loop = False
        j = i - 1
        while j > 0 and not end_loop:
            u_id, u_attr = nodes[j]
            if abs(v_attr["x"] - u_attr["x"]) <= max_distance:
                if abs(v_attr["y"] - u_attr["y"]) <= max_distance:
                    graph.add_edge(v_id, u_id)
            else:
                end_loop = True
            j -= 1

        # Check if there must be an edge between v and the following nodes in the list
        end_loop = False
        j = i + 1
        while j < n_nodes and not end_loop:
            u_id, u_attr = nodes[j]
            if abs(v_attr["x"] - u_attr["x"]) <= max_distance:
                if abs(v_attr["y"] - u_attr["y"]) <= max_distance:
                    graph.add_edge(v_id, u_id)
            else:
                end_loop = True
            j += 1


def add_weights(graph: Graph):
    """
    Assign a weight to each edge in the graph based on the distance between its nodes
    """
    for edge in graph.edges:
        v = graph.nodes[edge[0]]
        u = graph.nodes[edge[1]]
        graph[edge[0]][edge[1]]["weight"] = get_node_distance(v, u)


def get_node_distance(v, u):
    return math.sqrt((v["x"]-u["x"]) ** 2 + (v["y"]-u["y"]) ** 2)


# ----------------------------------
#       TEST FUNCTIONS SECTION
# ----------------------------------

def test_graph_generation():
    print("\n\n--- TEST BUILDING GRAPHS ---")

    print(f"\nEvaluating graph P...")
    naive_time = math.inf
    sorting_time = math.inf
    graph_P = generate_graph_P_nodes()
    max_distance = 0.8
    for i in tqdm(range(TIME_EVALUATIONS)):
        graph = graph_P.copy()
        start = timeit.default_timer()
        add_edges_naive(graph, max_distance)
        end = timeit.default_timer()
        naive_time = min(end - start, naive_time)

        graph = graph_P.copy()
        start = timeit.default_timer()
        add_edges_sorting(graph, max_distance)
        end = timeit.default_timer()
        sorting_time = min(end - start, sorting_time)

    print(" - Execution times")
    print(f"\tNaive: {naive_time * 1000} ms")
    print(f"\tSorting: {sorting_time * 1000} ms")

    print(f"\nEvaluating graph R...")
    naive_time = math.inf
    sorting_time = math.inf
    graph_R = generate_graph_R_nodes(2000)
    max_distance = 0.08
    for i in tqdm(range(TIME_EVALUATIONS)):
        graph = graph_R.copy()
        start = timeit.default_timer()
        add_edges_naive(graph, max_distance)
        end = timeit.default_timer()
        naive_time = min(end - start, naive_time)

        graph = graph_R.copy()
        start = timeit.default_timer()
        add_edges_sorting(graph, max_distance)
        end = timeit.default_timer()
        sorting_time = min(end - start, sorting_time)

    print(" - Execution times")
    print(f"\tNaive: {naive_time * 1000} ms")
    print(f"\tSorting: {sorting_time * 1000} ms")


def test_counting_triangles(dict_graphs: dict):
    print("\n\n--- TEST COUNTING TRIANGLES ---")

    for graph_name, graph in dict_graphs.items():
        print(f"\nEvaluating graph {graph_name}...")

        #naive_time = math.inf
        nbrs_time = math.inf
        ldv_time = math.inf
        nx_time = math.inf
        for i in tqdm(range(TIME_EVALUATIONS)):
            #start = timeit.default_timer()
            #counting_triangles_naive(graph)
            #end = timeit.default_timer()
            #naive_time = min(end - start, naive_time)

            start = timeit.default_timer()
            counting_triangles_nbrs_pairs(graph)
            end = timeit.default_timer()
            nbrs_time = min(end - start, nbrs_time)

            start = timeit.default_timer()
            counting_triangles_ldv(graph)
            end = timeit.default_timer()
            ldv_time = min(end - start, ldv_time)

            start = timeit.default_timer()
            sum(nx.triangles(graph).values()) / 3
            end = timeit.default_timer()
            nx_time = min(end - start, nx_time)

        print(" - Triangles counted")
        #print(f"\tNaive: {counting_triangles_naive(graph)}")
        print(f"\tNeighbors Pairs: {counting_triangles_nbrs_pairs(graph)}")
        print(f"\tLDV: {counting_triangles_ldv(graph)}")
        print(f"\tNetworkX: {sum(nx.triangles(graph).values()) / 3}")
        print(" - Execution times")
        # print(f"\tNaive: {naive_time * 1000} ms")
        print(f"\tNeighbors Pairs: {nbrs_time * 1000} ms")
        print(f"\tLDV: {ldv_time * 1000} ms")
        print(f"\tNetworkX: {nx_time * 1000} ms")


def test_clustering_coefficients(dict_graphs: dict):
    print("\n\n--- TEST CLUSTERING COEFFICIENTS ---")

    for graph_name, graph in dict_graphs.items():
        print(f"\nEvaluating graph {graph_name}...")

        set_time = math.inf
        list_time = math.inf
        nx_time = math.inf
        for i in tqdm(range(TIME_EVALUATIONS)):
            start = timeit.default_timer()
            clustering_coefficients_set_intersection(graph)
            end = timeit.default_timer()
            set_time = min(end - start, set_time)

            start = timeit.default_timer()
            clustering_coefficients_list_intersection(graph)
            end = timeit.default_timer()
            list_time = min(end - start, list_time)

            start = timeit.default_timer()
            nx.clustering(graph)
            end = timeit.default_timer()
            nx_time = min(end - start, nx_time)

        print(" - Coefficients calculated")
        print(f"\tSet Intersection: {clustering_coefficients_set_intersection(graph)}")
        print(f"\tList Intersection: {clustering_coefficients_list_intersection(graph)}")
        print(f"\tNetworkX: {nx.clustering(graph)}")
        print(" - Execution times")
        print(f"\tSet Intersection: {set_time * 1000} ms")
        print(f"\tList Intersection: {list_time * 1000} ms")
        print(f"\tNetworkX: {nx_time * 1000} ms")


if __name__ == '__main__':
    graph_P, graph_R = build_graphs()

    dict_graphs = {"P": graph_P, "R": graph_R}

    test_graph_generation()
    test_counting_triangles(dict_graphs)
    test_clustering_coefficients(dict_graphs)
