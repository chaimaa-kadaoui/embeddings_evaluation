import graphcreation as gc
import networkx as nx
import community
import numpy as np
import pandas as pd
from time import time
from os import path
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections

plt.style.use("bmh")
mpl.rcParams['figure.figsize'] = (23.5, 11)


def get_file_path(half_size, dim, emb_name):
    """Get the desired file_path for a specific embedding name and dimension and half_size"""

    file_name = '_'.join([emb_name, 'window_half_size='+half_size, 'd='+dim])
    file_ext = '.gz'
    if emb_name == 'GloVe':
        file_ext = '.txt'+file_ext
    file_path = path.join('embeddings', emb_name, file_name+file_ext)
    return file_path


def degree_histogram(G, normed=True):
    """
    Generating degree histogram with the mean
    """
    # Sorting degrees in descending order
    degree_seq = sorted([d for n, d in G.degree().items()], reverse=True)

    degree_count = collections.Counter(degree_seq)
    deg, count = zip(*degree_count.items())
    # If normed, the degrees frequencies are displayed instead of the counts
    if normed:
        count = tuple([c / G.number_of_nodes() for c in count])
    # Computing the mean
    mean_degree = np.asarray(deg).dot(np.asarray(count)) / sum(np.asarray(count))
    # Generating the histogram
    fig, ax = plt.subplots()
    plt.bar(deg, count, width=0.8, color='b', alpha=0.4)
    ax.axvline(mean_degree, color='b', linewidth=2, label='Mean: {:0.1f}'.format(mean_degree))
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    ax.legend(loc='upper right')
    plt.tight_layout()
    #plt.savefig("degree_histogram.png")
    plt.show()


def diameter(graph):
    """Returns the dimater of the graph, for more than 1 components, returns the maximum diameter"""

    graphs = list(nx.connected_component_subgraphs(graph))
    return max(list(map(nx.diameter, graphs)))


def coefficients(graph, key, weights=None):
    """Returns the clustering coefficients of the graph and updates the global DataFrame all_coeffs
    We can use weights to compute the coefficients"""

    global all_coeffs
    coeffs = nx.clustering(graph, weight=weights)
    coeffs = pd.Series(list(coeffs.values()))
    all_coeffs[key] = coeffs
    return coeffs.mean()


def plot_clustering(all_coeffs):
    """"For each pair of dimension and half_size, plots the histogram of clustering coefficients for the 3 embeddings"""

    names = ["GloVe", "HPCA", "Word2Vec"]
    pars = ["2_50", "2_200", "5_50", "5_200"]
    for i in range(4):
        plt.hist(all_coeffs.iloc[:, 3*i:3*(i+1)].values, label=names)
        plt.title("Histogram of Clustering Coefficients with halfsize_dim="+pars[i], size=24)
        plt.xlabel("Clustering coefficients", size=14)
        plt.ylabel("Number of nodes", size=14)
        plt.legend()
        plt.savefig(path.join("results", "coeffs_" + pars[i] + ".png"), format="png")
        plt.close()


def community_detection(graph):
    """Uses the library community to partition the graph into communities
    The partitions are chosen so as to maximize the modularity of the graph
    Returns the partition dictionary and its corresponding 1-d array"""

    partition = community.best_partition(graph)
    commu = np.array(list(partition.values()))
    return commu, partition


def count_communities(commu):
    """For a given embedding_dim_half_size, returns the histogram of the population of communities"""

    cnt = pd.Series([])
    histo = commu.value_counts()  # An array with the population of each community
    for i in range(10):
        # We compute how many communities have between 100*i and 100*(i+1) members
        check = (histo >= 100*i)&(histo < 100*(i+1))
        if i == 9:
            check = (histo >= 100*i)
        cnt[100*i] = check.sum()
    return cnt


def plot_communities(all_commu):
    """"For each pair of dimension and half_size, plots the histogram of communities' sizes for the 3 embeddings
    We use the function "count_communities" defined above"""

    names = ["GloVe", "HPCA", "Word2Vec"]
    pars = ["2_50", "2_200", "5_50", "5_200"]
    for i in range(4):
        commu = all_commu.iloc[:, 3*i:3*(i+1)]
        cnt = commu.apply(count_communities)
        cnt.plot(kind="bar", label=names)
        plt.title("Histogram of communities' size with halfsize_dim="+pars[i], size=24)
        plt.xlabel("Number of words", size=18)
        plt.ylabel("Number of communities", size=18)
        plt.legend()
        plt.savefig(path.join("results", "communities_" + pars[i] + ".png"), format="png")
        plt.close()


def plot_results(results, col_idx, ylabel, ylim):
    """Plots a specific column of the results"""

    all_colors = [parameter['color'] for parameter in plt.rcParams['axes.prop_cycle']]
    names = ["GloVe", "HPCA", "Word2Vec"]
    pars = ["2_50", "2_200", "5_50", "5_200"]
    width = 0.1
    fig, ax = plt.subplots()
    ind = np.arange(len(names))

    for i in range(len(pars)):
        vals = list(results.iloc[3 * i:3 * (i + 1), col_idx])
        ax.bar(ind+width*i, vals, width, color=all_colors[i], label=pars[i])

    plt.xlabel('Embeddings')
    plt.ylabel(ylabel)
    plt.title(ylabel+' of embeddings')
    plt.ylim(ylim)
    plt.xticks(ind + 2*width, names)
    plt.legend(loc="lower right", framealpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    half_sizes = ['2', '5']
    dims = ['50', '200']
    emb_names = ['GloVe', 'HPCA', 'Word2Vec']

    all_coeffs = pd.DataFrame([])
    all_commu = pd.DataFrame([])
    results = pd.DataFrame([])

    for half_size in half_sizes:
        for dim in dims:
            for name in emb_names:
                key = '_'.join([name, half_size, dim])
                print(key)
                graph = nx.read_graphml(path.join("graphs_knn_5", key+".graphml"))
                nb_comp = len(list(nx.connected_components(graph)))
                results.loc[key, "nb_comp"] = nb_comp
                diam = diameter(graph)
                results.loc[key, "diameter"] = diam
                coeffs = coefficients(graph, key)
                results.loc[key, "clustering_coeff"] = coeffs
                commu, partition = community_detection(graph)
                modularity = community.modularity(partition, graph)
                results.loc[key, "nb_communities"] = commu.max()
                results.loc[key, "modularity"] = modularity
                all_commu[key] = commu
                results.to_csv(path.join("results", "results.csv"))
                all_coeffs.to_csv(path.join("results", "coefficients.csv"), index=False)
                all_commu.to_csv(path.join("results", "communities.csv"), index=False)
    plot_clustering(all_coeffs)
    plot_communities(all_commu)
