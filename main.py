import networkx as nx
from networkx import Graph
from tqdm import tqdm
import numpy as np
import json
import os

DATA_FOLDER = "data"
FILE_JSON_DATA = os.path.join(DATA_FOLDER, "dpc-covid19-ita-province.json")
FILE_GRAPH_P = os.path.join(DATA_FOLDER, "graph_P.yml")
FILE_GRAPH_R = os.path.join(DATA_FOLDER, "graph_R.yml")

LOAD_GRAPHS_FROM_FILES = True  # Flag to load graphs from saved file instead of build them from JSON data

def low_degree_vertex(graph):
    nodes=list(graph.nodes)
    triangle_count=0
    # For each nodes, we need to iterate on its neighbors and for each couple of them we need to find if there is an edge between that connect the pair.
    # The next cycle iterates on the graph nodes.
    # In this algorithm we consider only the nodes that have higher degree of v.
    for v in nodes:
        for u in graph.neighbors(v):
            for w in graph.neighbors(v):
                if(u!=w):
                    if(graph.degree[u]>graph.degree[v] and graph.degree[w]>graph.degree[v]):
                        if(u in graph.neighbors(w)):
                            triangle_count=triangle_count+1
    return triangle_count

def clustering_coefficient(graph):
    nodes=list(graph.nodes)
    coefficient=list()
    for v in nodes:
        v_neighbors=set(graph.neighbors(v))
        appo=0
        for u in graph.neighbors(v):
            u_neighbors=set(graph.neighbors(u))
            appo=appo+len(v_neighbors.intersection(u_neighbors))
        coefficient.append((2*appo)/(len(graph.neighbors(v))*(len(graph.neighbors(v))-1)))
    return coefficient



def function():
    # Preprocessing data.
    # Read the data.
    # Each item of the datastore array represent a province.
    with open('./../../Documenti/COVID-19/dati-json/dpc-covid19-ita-province.json') as f:
        datastore = json.load(f)

    # In the json file there are some provinces labeled with an unusual "codice_provincia" and
    # in this provinces the field values of "denominazione_provincia" is equal to "In fase di definizione/aggiornamento".
    # I suppose that i need to discharge this provinces.

    # Discharge the provinces where the field "denominazione_provincia" is equal to 'In fase di definizione/aggiornamento'
    for data in datastore:
        if (data["denominazione_provincia"] == 'In fase di definizione/aggiornamento'):
            datastore.remove(data)

    # Create the province's graph.
    distance = 0.8;
    graph = nx.Graph();

    # Add nodes.
    for data in datastore:
        graph.add_node(data["codice_provincia"], **data)

    # Add edges.
    for x in tqdm(datastore):
        for y in datastore:
            if (x != y):
                if (abs(x["lat"] - y["lat"]) <= distance and abs(x["long"] - y["long"]) <= distance):
                    graph.add_edge(x["codice_provincia"], y["codice_provincia"])

    # Migliorabile utilizzando qualche cosa come i vicini

    print(graph.number_of_nodes(), graph.number_of_edges())


    # Generate 2000 double (x,y) with x in[30,50) and y in [10,20)
    # Shall the elements be distinct?
    # Instead of a list we can create a dict in order to emulate the former graph creation.
    double_list = [(i, np.random.uniform(30, 50), np.random.uniform(10, 20)) for i in range(0, 2000)]

    # Create an handcrafted graph with the doubles.
    handcrafted_graph = nx.Graph()
    for item in double_list:
        handcrafted_graph.add_node(item)

    # Add edges to the graph
    distance = 0.08
    for x in tqdm(double_list):
        for y in double_list:
            if (x[0] != y[0]):
                if (abs(x[1] - y[1]) <= distance and abs(x[2] - y[2] <= distance)):
                    handcrafted_graph.add_edge(x, y);

    print(handcrafted_graph.number_of_nodes(), handcrafted_graph.number_of_edges())


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
        for j in range(i+1, n_provinces):
            data_j = datastore[j]
            # There is an edge between two nodes/provinces if their coordinates differ for less than max_distance
            if abs(data_i["lat"] - data_j["lat"]) <= max_distance and abs(data_i["long"] - data_j["long"]) <= max_distance:
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
        for j in range(i+1, n_nodes):
            node_j = list_pairs[j]
            # There is an edge between two nodes if their coordinates differ for less than max_distance
            if abs(node_i[1] - node_j[1]) <= max_distance and abs(node_i[2] - node_j[2]) <= max_distance:
                graph.add_edge(node_i[0], node_j[0])

    print(f"Graph R built. Nodes: {graph.number_of_nodes()} - Edges: {graph.number_of_edges()}")

    return graph


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
                for j in range(i+1, n_nbrs):
                    w = list_nbrs[j]
                    w_degree = graph.degree[w]
                    if w_degree > v_degree or (w_degree == v_degree and v < w):
                        # If it exists and edge connecting u and w we found a triangle
                        if u in graph.adj[w]:
                            count_triangles += 1

    return count_triangles


if __name__ == '__main__':
    # function()
    if LOAD_GRAPHS_FROM_FILES:
        print("Loading graphs from files...")
        graph_P = nx.read_yaml(FILE_GRAPH_P)
        graph_R = nx.read_yaml(FILE_GRAPH_R)
        print("Graphs loaded")
    else:
        graph_P, graph_R = build_graphs()

    print(f"Apo: {low_degree_vertex(graph_P)}")
    print(f"Jason: {counting_triangles_ldv(graph_P)}")
    print(f"NetworkX: {sum(nx.triangles(graph_P).values())/3}")

