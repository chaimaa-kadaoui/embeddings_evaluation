# encoding: utf-8

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as neighbors
import sklearn.metrics.pairwise as smp
from sklearn.preprocessing import minmax_scale
plt.style.use('ggplot')


def load_embeddings(file_name):
    data = pd.read_csv(file_name, sep='\t', header=None)
    vocabulary = list(data.iloc[:, 0])
    embeddings = data.iloc[:, 1:].values
    embeddings = minmax_scale(embeddings)
    return vocabulary, embeddings


def get_adjacency(embeddings, metric, graph_type, graph_param, metric_param2=1):
    if metric == "cosine":
        metric = lambda x, y: smp.paired_cosine_distances(x.reshape(1, -1), y.reshape(1, -1))
    elif metric == "exp":
        metric = lambda x, y: np.exp((smp.paired_euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))**2) / (2*metric_param2))
    else:
        return "Wrong metric"

    if graph_type == "knn":
        adjacency = neighbors.kneighbors_graph(embeddings, graph_param, metric=metric,
                                               mode='distance', include_self=True).toarray()
        to_fill = ((adjacency == 0) * (adjacency.T != 0))
        adjacency[to_fill] = adjacency.T[to_fill]
    elif graph_type == "eps":
        adjacency = neighbors.radius_neighbors_graph(embeddings, graph_param, metric=metric,
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
