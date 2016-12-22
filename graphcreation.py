# encoding: utf-8

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
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

def get_similarity(embeddings, metric, sigma2=1):
    similarity = dist.squareform(dist.pdist(embeddings, metric))
    if metric == "cosine":
        return 1 - similarity
    elif metric == "sqeuclidean":
        return np.exp(-(similarity)/(2*sigma2))
    else:
        return "Wrong metric"

def get_adjacency(embeddings, metric, graph_type, graph_param, metric_param2=1):

    if metric == "cosine":
        metric = lambda x, y: smp.paired_cosine_distances(x.reshape(1, -1), y.reshape(1, -1))
    elif metric == "sqeuclidean":
        metric = lambda x, y: np.exp((smp.paired_euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))**2) / (2*metric_param2))
    else:
        return "Wrong metric"

    if graph_type == "knn":
        adjacency = neighbors.kneighbors_graph(embeddings, graph_param, metric=metric,
                                               mode='distance', include_self=True)
    elif graph_type == "eps":
        adjacency = neighbors.radius_neighbors_graph(embeddings, graph_param, metric=metric,
                                               mode='distance', include_self=True)
    else:
        return "Wrong parameters"
    to_fill = ((adjacency == 0) * (adjacency.T != 0))
    adjacency[to_fill] = adjacency.T[to_fill]
    adjacency = adjacency.toarray()
    return adjacency

def get_graph(adjacency):
    graph = nx.from_numpy_matrix(adjacency)
    nx.draw(graph)
    return graph