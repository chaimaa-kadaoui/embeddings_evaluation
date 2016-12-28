import graphcreation as gc
import networkx as nx
import community
import numpy as np
import pandas as pd
from time import time
from os import path
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("bmh")
mpl.rcParams['figure.figsize'] = (23.5, 11)

def get_file_path(half_size, dim, emb_name):
    file_name = '_'.join([emb_name, 'window_half_size='+half_size, 'd='+dim])
    file_ext = '.gz'
    if emb_name == 'GloVe':
        file_ext = '.txt'+file_ext
    file_path = path.join('embeddings', emb_name, file_name+file_ext)
    return file_path


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
                # adj = gc.get_adjacency(embs, metric, graph_type, graph_param)
                adj = np.load(path.join("adjacencies", key+".npy"))
                graph = gc.get_graph(adj)
                labels = {idx: voc[idx] for idx in range(len(voc))}
                graph = nx.relabel_nodes(graph, labels)
                print("Number of connected components:", len(list(nx.connected_components(graph))))
                # diam = diameter(graph)
                # print(diam)
                # coeffs = coefficients(graph, key)
                # print(coeffs)
                commu, partition = community_detection(graph)
                print(commu.max())
                print(community.modularity(partition, graph))
                all_commu[key] = commu
        #         break
        #     break
        # break

    # all_coeffs.to_csv(path.join("results", "coefficients.csv"), index=False)
    # plot_clustering(all_coeffs)
    # all_commu.to_csv(path.join("results", "communities.csv"), index=False)
    # plot_communities(all_commu)