from os import path
import graphcreation as gc
import networkx as nx


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

if __name__ == "__main__":
    half_sizes = ['2', '5']
    dims = ['50', '200']
    emb_names = ['GloVe', 'HPCA', 'Word2Vec']

    metric = "cosine"
    graph_type = "lsh_knn"
    graph_param = 5

    all_vocs = {}
    all_embs = {}
    all_adjs = {}
    all_graphs = {}
    all_diams = {}

    for half_size in half_sizes:
        for dim in dims:
            for name in emb_names:
                key = '_'.join([name, half_size, dim])
                print(key)
                file_path = get_file_path(half_size, dim, name)
                voc, embs = gc.load_embeddings(file_path)
                adj = gc.get_adjacency(embs, metric, graph_type, graph_param)
                graph = gc.get_graph(adj)
                print("Number of connected components:", len(list(nx.connected_components(graph))))
                diam = diameter(graph)
                print(diam)
                all_vocs[key] = voc
                all_embs[key] = embs
                all_adjs[key] = adj
                all_graphs[key] = graph
                all_diams[key] = diam
        #     break
        # break
