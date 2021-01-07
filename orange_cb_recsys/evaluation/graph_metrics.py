from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
import networkx as nx


def nx_degree_centrality(g: NXFullGraph):
    return nx.degree_centrality(g.graph)


def nx_closeness_centrality(g: NXFullGraph):
    return nx.closeness_centrality(g.graph)


def nx_dispersion(g: NXFullGraph):
    return nx.dispersion(g.graph)
