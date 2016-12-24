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
    return nx.diameter(graph)

if __name__ == "__main__":
    half_sizes = ['2', '5']
    dims = ['50', '200']
    emb_names = ['GloVe', 'HPCA', 'Word2Vec']

    half_size = half_sizes[0]
    dim = dims[0]

    metric = "cosine"
    graph_type = "lsh_knn"
    graph_param = 5

    all_vocs = {}
    all_embs = {}
    all_adjs = {}
    all_graphs = {}
    all_diams = {}

    for name in emb_names:
        print(name)
        file_path = get_file_path(half_size, dim, name)
        voc, embs = gc.load_embeddings(file_path)
        adj = gc.get_adjacency(embs, metric, graph_type, graph_param)
        graph = gc.get_graph(adj)
        diam = diameter(graph)
        print(diam)
        key = '_'.join([name, half_size, dim])
        all_vocs[key] = voc
        all_embs[key] = embs
        all_adjs[key] = adj
        all_graphs[key] = graph
        all_diams[key] = diam

    print(all_diams)