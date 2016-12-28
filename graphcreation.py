# encoding: utf-8

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as neighbors
import sklearn.metrics.pairwise as smp
from sklearn.preprocessing import minmax_scale
import time
plt.style.use('ggplot')


def load_embeddings(file_name):
    data = pd.read_csv(file_name, sep='\t', header=None)
    vocabulary = list(data.iloc[:, 0])
    embeddings = data.iloc[:, 1:].values
    embeddings = minmax_scale(embeddings)
    return vocabulary, embeddings


def opt_epsilon(embeddings, metric, metric_param2=1):
    """
    Returns the smallest value of epsilon where the graph is connected, i.e the length
    of the longest edge in a minimal spanning tree (MST) of the fully connected graph
    """
    # Building the fully connected graph
    if metric == "cosine":
        adjacency = smp.cosine_distances(embeddings)
    elif metric == "exp":
        adjacency = np.exp(smp.euclidean_distances(embeddings)**2 / (2*metric_param2))
    else:
        return 'Wrong metric: must be "cosine" or "exp"'
    graph = nx.from_numpy_matrix(adjacency)
    # Computing the MST
    T = nx.minimum_spanning_tree(graph)
    # Longest edge in the MST
    max_weight = max(d.get('weight', 1) for u, v, d in T.edges(data=True))
    return max_weight


def get_adjacency(embeddings, metric, graph_type, graph_param, metric_param2=1):
    if metric == "cosine":
        lambda_metric = lambda x, y: smp.paired_cosine_distances(x.reshape(1, -1), y.reshape(1, -1))
    elif metric == "exp":
        lambda_metric = lambda x, y: np.exp((smp.paired_euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))**2) / (2*metric_param2))
    else:
        return 'Wrong metric: must be "cosine" or "exp"'

    if graph_type == "knn":
        adjacency = neighbors.kneighbors_graph(embeddings, graph_param, metric=lambda_metric,
                                               mode='distance', include_self=True).toarray()
        to_fill = ((adjacency == 0) * (adjacency.T != 0))
        adjacency[to_fill] = adjacency.T[to_fill]
    elif graph_type == "eps":
        if graph_param == "opt":
            # Choosing epsilon such that the graph is safely connected
            graph_param = opt_epsilon(embeddings, metric, metric_param2)
        adjacency = neighbors.radius_neighbors_graph(embeddings, graph_param, metric=lambda_metric,
                                                     mode='distance', include_self=True).toarray()
    elif graph_type == 'lsh_knn':
        lshf = neighbors.LSHForest()
        lshf.fit(embeddings)
        adjacency = lshf.kneighbors_graph(embeddings, graph_param, mode='distance').toarray()
        to_fill = ((adjacency == 0) * (adjacency.T != 0))
        adjacency[to_fill] = adjacency.T[to_fill]
    elif graph_type == 'lsh_eps':
        lshf = neighbors.LSHForest()
        lshf.fit(embeddings)
        adjacency = lshf.radius_neighbors_graph(embeddings, graph_param, mode='distance').toarray()
    else:
        return "Wrong parameters"
    return adjacency


def get_graph(adjacency):
    graph = nx.from_numpy_matrix(adjacency)
    return graph
