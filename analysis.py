import graphcreation as gc
import networkx as nx
import community
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import csv

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


def degree_histogram(G, filename, normed):
    """
    Generates degree histogram with the mean
    """
    # Sorting degrees in descending order and counting
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
    plt.savefig("degree_histogram_" + filename + ".png")


def degree_log_log(list_graphs, list_keys, half_size, dim, cdf):
    """
    Plots the degree distribution for each graph in 'list_graphs' with a log-log
    scale on the same figure. If 'cdf' is True, the cumulative function distribution
    is plotted.
    """
    plt.figure()
    plt.ylabel("Log probability")
    plt.xlabel("Log degree")
    i = 0
    for graph in list_graphs:
        # Sorting degrees in descending order and counting
        degree_seq = sorted([d for n, d in graph.degree().items()], reverse=True)
        degree_count = collections.Counter(degree_seq)
        deg, count = zip(*degree_count.items())
        # Computing the degrees frequencies
        count = np.array([c / graph.number_of_nodes() for c in count])
        deg = np.asarray(deg)
        sorted_indexes = np.argsort(deg)
        count = count[sorted_indexes]
        # If cdf, the cumulative distribution function is displayed
        if cdf:
            # Cumulative distribution function computation
            cdf_values = np.cumsum(count[::-1])[::-1]
            plt.plot(np.log(deg[sorted_indexes]), np.log(cdf_values), label=list_keys[i])
            plt.title("Cumulative degree distribution in log-log scale")
            keyword = 'cdf'
        # If not cdf, the distribution is displayed
        else:
            plt.plot(np.log(deg[sorted_indexes]), np.log(count), label=list_keys[i])
            plt.title("Degree distribution in log-log scale")
            keyword = 'distrib'
        i += 1
    plt.legend()
    plt.tight_layout()
    plt.savefig("degree_" + keyword + "_loglog_" + half_size + '_' + dim + ".png")


def degree_analysis(half_sizes, dims, emb_names, dir):
    """
    Analysis of the degree distribution: calls function 'degree_histogram'
    and 'degree_log_log' on each graph
    """
    for half_size in half_sizes:
        for dim in dims:
            list_graphs, list_keys = [], []
            for name in emb_names:
                key = '_'.join([name, half_size, dim]) + '.graphml'
                key = '/'.join([dir, key])
                print(key)
                list_keys.append(key)
                # Reading the graph
                graph = nx.read_graphml(key)
                list_graphs.append(graph)
                filename = (key.rsplit('/', 1)[-1]).rsplit('.', 1)[0]
                # Creation of the histogram
                degree_histogram(graph, filename, normed=True)
            print(list_keys)
            # Creation of the log-log plots for the 3 graphs corresponding
            # to the same half size and dimension
            degree_log_log(list_graphs, list_keys, half_size, dim, cdf=True)


def centrality_degree(G, nb_nodes):
    """
    Finds (at least) the 'nb_nodes' most central nodes using the degree
    """
    # Sorting the degree sequence and finding the 'nb_nodes' largest values
    degree_seq = sorted([d for n, d in G.degree().items()], reverse=True)
    max_degrees = degree_seq[0:nb_nodes]
    # Finding the nodes corresponding to these max degrees
    central_nodes = [(n, int(d)) for n, d in G.degree().items() if d in max_degrees]
    return central_nodes


def centrality_nx(G, nb_nodes, type):
    if type == 'closeness':
        scores = nx.closeness_centrality(G)
    elif type == 'betweenness':
        scores = nx.betweenness_centrality(G)
    else:
        print('Error: type must be "closeness" or "betweenness".')
    sorted_scores = sorted([c for n, c in scores.items()], reverse=True)
    max_scores = sorted_scores[0:nb_nodes]
    central_nodes = [(n, c) for n, c in scores.items() if c in max_scores]
    return central_nodes


def centrality_analysis(half_sizes, dims, emb_names, dir):
    """
    Analysis of the centrality in all graphs
    """
    for half_size in half_sizes:
        for dim in dims:
            for name in emb_names:
                key = '_'.join([name, half_size, dim]) + '.graphml'
                key = '/'.join([dir, key])
                print(key)
                # Reading the graph
                graph = nx.read_graphml(key)
                key = (key.rsplit('/', 1)[-1]).rsplit('.', 1)[0]
                # Finding (at least) the 'nb_nodes' most central nodes
                nb_nodes = 10
                # central_nodes = centrality_degree(graph, nb_nodes)
                # # Writing results in a .csv file
                # with open('centrality_degree.csv', 'a') as result:
                #     result_csv = csv.writer(result)
                #     result_csv.writerow([key, ' '])
                #     for row in central_nodes:
                #         result_csv.writerow(row)
                type = 'closeness'
                central_nodes = centrality_nx(graph, nb_nodes, type)
                # Writing results in a .csv file
                filename = 'centrality_' + type + '.csv'
                with open(filename, 'a') as result:
                    result_csv = csv.writer(result)
                    result_csv.writerow([key, ' '])
                    for row in central_nodes:
                        result_csv.writerow(row)


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

def get_top_10(group):
   top = group[["centrality", "words"]].sort_values("centrality", ascending=False)["words"]
   top = list(top)[:10]
   return top


def top_words_community(graph, commus):
    """Returns the top 10 words (by closeness centrality) for the 5 most crowded communities"""

    global top_words
    # We would like to take only the 5 most crowded communities
    counts = pd.get_dummies(commus).cumsum().iloc[-1, :].sort_values(ascending=False)
    top_commus = list(counts.index)[:5]
    commu_filtered = commus[commus.isin(top_commus)]
    indexes = list(commu_filtered.index)  # The indexes of the nodes who belong to the top 5 communities
    # We get the degree centrality of nodes
    centrality = nx.algorithms.centrality.closeness_centrality(graph)
    words = np.array(list(centrality.keys()))[indexes]
    centrality = np.array(list(centrality.values()))[indexes]
    # We want to take the 10 most important words by community and store them in a list
    df = pd.DataFrame([words, commu_filtered, centrality], index=["words", "community", "centrality"]).T
    top_words[key] = df.groupby("community").apply(get_top_10).reset_index(drop=True)


def count_communities(commu):
    """For a given embedding_dim_half_size, returns the histogram of the population of communities"""

    cnt = pd.Series([])
    histo = commu.value_counts()  # An array with the population of each community
    for i in range(10):
        # We compute how many communities have between 100*i and 100*(i+1) members
        check = (histo >= 100*i)&(histo < 100*(i+1))
        if i == 9:
            check = (histo >= 100*i)
        cnt[100*i] = 100.0 * check.sum() / commu.max()
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
        plt.ylabel("Percentage of communities", size=18)
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
    dir = 'graphs_knn_5'
    degree_analysis(half_sizes, dims, emb_names, dir)
    centrality_analysis(half_sizes, dims, emb_names, dir)

    results = pd.DataFrame([])
    top_words = pd.DataFrame([])

    for half_size in half_sizes:
        for dim in dims:
            for name in emb_names:
                key = '_'.join([name, half_size, dim])
                print(key)
                graph = nx.read_graphml(path.join("graphs_knn_5", key + ".graphml"))
                nb_comp = len(list(nx.connected_components(graph)))
                results.loc[key, "nb_comp"] = nb_comp
                diam = diameter(graph)
                results.loc[key, "diameter"] = diam
                coeffs = coefficients(graph, key)
                results.loc[key, "clustering_coeff"] = coeffs
                weights = nx.get_edge_attributes(graph, 'weight')
                edges = [edge for edge, val in weights.items() if val < 0]
                graph.remove_edges_from(edges)
                commu, partition = community_detection(graph)
                modularity = community.modularity(partition, graph)
                results.loc[key, "nb_communities"] = commu.max()
                results.loc[key, "modularity"] = modularity
                all_commu[key] = commu
                top_words_community(graph, all_commu[key])
                results.to_csv(path.join("results", "results.csv"))
                all_coeffs.to_csv(path.join("results", "coefficients.csv"), index=False)
                all_commu.to_csv(path.join("results", "communities.csv"), index=False)
                top_words.to_csv(path.join("results", "top_words.csv"), index=False)
    plot_clustering(all_coeffs)
    plot_communities(all_commu)
