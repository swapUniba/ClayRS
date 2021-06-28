import networkx as nx
from abc import ABC, abstractmethod
import pandas as pd

from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph


class FeatureSelection(ABC):
    @abstractmethod
    def perform(self, graph, ratings: pd.DataFrame, user_exogenous_properties=None, item_exogenous_properties=None):
        raise NotImplementedError


class NXFSPageRank(FeatureSelection):
    def perform(self, graph: nx.Graph, ratings: pd.DataFrame, user_exogenous_properties=None, item_exogenous_properties=None):
        rank = nx.pagerank(graph)
        new_ratings = pd.DataFrame()
        for rk in rank.keys():
            to_append = pd.DataFrame()
            if rk in ratings[['from_id']].values:
                to_append = ratings[ratings['from_id'] == rk]
            elif rk in ratings[['to_id']].values:
                to_append = ratings[ratings['to_id'] == rk]
            new_ratings = new_ratings.append(to_append)
        if new_ratings.empty:
            return graph
        new_graph = NXFullGraph(new_ratings)
        return new_graph
