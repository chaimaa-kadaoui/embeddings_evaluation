#import graphcreation as gc
import networkx as nx
#import community
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
    graphs = list(nx.connected_component_subgraphs(graph))
    return max(list(map(nx.diameter, graphs)))


def coefficients(graph, key, weights=None):
    global all_coeffs
    # weights = "weight"
    coeffs = nx.clustering(graph, weight=weights)
    coeffs = pd.Series(list(coeffs.values()))
    all_coeffs[key] = coeffs
    # plt.hist(coeffs)
    # plt.xlabel("Clustering coefficients", size=14)
    # plt.ylabel("Number of nodes", size=14)
    # plt.title("Histogram of clustering coefficients for "+key, size=18)
    # plt.savefig(path.join("results", "coefficients_"+key+".png"), format="png")
    # plt.close()
    return coeffs.mean()


def plot_clustering(all_coeffs):
    names = ["GloVe", "HPCA", "Word2Vec"]
    pars = ["2_50", "2_200", "5_50", "5_200"]
    for i in range(4):
        n, bins, patches = plt.hist(all_coeffs.iloc[:, 3*i:3*(i+1)].values, normed=True, label=names)
        # plt.close()
        # for j, values in enumerate(n):
        #     plt.plot(bins[:-1], values, '--', marker='o', linewidth=1, label=names[j])
        #     # plt.yscale('log', nonposy='clip')
        plt.title("Histogram of Clustering Coefficients with halfsize_dim="+pars[i], size=24)
        plt.xlabel("Clustering coefficients", size=14)
        plt.ylabel("Number of nodes", size=14)
        plt.legend()
        plt.savefig(path.join("results", "coeffs_" + pars[i] + ".png"), format="png")
        plt.close()


def community_detection(graph):
    partition = community.best_partition(graph)
    commu = np.array(list(partition.values()))
    return commu, partition


def community_evaluation(graph, partition):
    partition = pd.Series(partition)
    list_commus = partition.unique()
    seq = []
    for idx in list_commus:
        part_i = partition[partition == idx].index
        seq.append(list(part_i))
    cov = nx.algorithms.community.quality.coverage(graph, seq)
    perf = nx.algorithms.community.quality.performance(graph, seq)
    return cov, perf


def count_communities(commu):
    cnt = pd.Series([])
    histo = commu.value_counts()
    for i in range(10):
        check = (histo >= 100*i)&(histo < 100*(i+1))
        if i == 9:
            check = (histo >= 100*i)
        cnt[100*i] = check.sum()
    return cnt


def plot_communities(all_commu):
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


if __name__ == "__main__":
    half_sizes = ['2', '5']
    dims = ['50', '200']
    emb_names = ['GloVe', 'HPCA', 'Word2Vec']

    metric = "cosine"
    graph_type = "lsh_knn"
    graph_param = 5

    all_coeffs = pd.DataFrame([])
    all_commu = pd.DataFrame([])

    for half_size in half_sizes:
        for dim in dims:
            for name in emb_names:
                start = time()
                key = '_'.join([name, half_size, dim])
                print(key)
                file_path = get_file_path(half_size, dim, name)
                voc, embs = gc.load_embeddings(file_path)
                adj = gc.get_adjacency(embs, metric, graph_type, graph_param)
                graph = gc.get_graph(adj)
                print("Number of connected components:", len(list(nx.connected_components(graph))))
                # diam = diameter(graph)
                # print(diam)
                # coeffs = coefficients(graph, key)
                # print(coeffs)
                commu, partition = community_detection(graph)
                print(commu.max())
                cov, perf = community_evaluation(graph, partition)
                print(cov, perf)
                all_commu[key] = commu
        #         break
        #     break
        # break

    # all_coeffs.to_csv(path.join("results", "coefficients.csv"), index=False)
    # plot_clustering(all_coeffs)
    # all_commu.to_csv(path.join("results", "communities.csv"), index=False)
    # plot_communities(all_commu)