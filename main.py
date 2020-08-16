# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import networkx as nx
from tqdm import tqdm
import numpy as np
import json

def low_degree_vertex(graph):
    nodes=list(graph.nodes)
    triangle_count=0
    #For each nodes, we need to iterate on its neighbors and for each couple of them we need to find if there is an edge between that connect the pair.
    #The next cycle iterates on the graph nodes.
    #In this algorithm we consider only the nodes that have higher degree of v.
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    function()

