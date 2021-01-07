from typing import List, Dict
import networkx as nx
import pandas as pd
import numpy as np
from abc import abstractmethod

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
from orange_cb_recsys.recsys.graphs import Graph
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.recsys.graphs.graph import FullGraph
from orange_cb_recsys.recsys.graphs.tripartite_graphs import NXTripartiteGraph
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.feature_selection import FeatureSelection


class PageRankAlg(RankingAlgorithm):
    def __init__(self, graph: FullGraph = None, personalized: bool = True):
        super().__init__('', '')
        self.__personalized = personalized
        self.__fullgraph: FullGraph = graph

    @property
    def fullgraph(self):
        return self.__fullgraph

    def set_fullgraph(self, graph):
        self.__fullgraph = graph

    @property
    def personalized(self):
        return self.__personalized

    def set_personalized(self, personalized):
        self.__personalized = personalized

    @abstractmethod
    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        raise NotImplemented

    def clean_rank(self, rank: Dict, user_id: str,
                        remove_from_nodes: bool = True,
                        remove_profile: bool = True,
                        remove_properties: bool = True) -> Dict:
        extracted_profile = self.extract_profile(user_id)
        new_rank = {k: rank[k] for k in rank.keys()}
        for k in rank.keys():
            if remove_from_nodes and self.__fullgraph.is_from_node(k):
                new_rank.pop(k)
            if remove_profile and self.__fullgraph.is_to_node(k) and k in extracted_profile.keys():
                new_rank.pop(k)
            if remove_properties and not self.__fullgraph.is_to_node(k) and not self.__fullgraph.is_from_node(k):
                new_rank.pop(k)
        return new_rank

    def extract_profile(self, user_id: str) -> Dict:
        adj = self.__fullgraph.get_adj(user_id)
        profile = {}
        #logger.info('unpack %s', str(adj))
        for a in adj:
            #logger.info('unpack %s', str(a))
            edge_data = self.__fullgraph.get_edge_data(user_id, a)
            profile[a] = edge_data['weight']
            logger.info('unpack %s, %s', str(a), str(profile[a]))
        return profile #{t: w for (f, t, w) in adj}


class NXPageRank(PageRankAlg):

    def __init__(self, graph: NXFullGraph = None, personalized: bool = False):
        super().__init__(graph=graph, personalized=personalized)

    def predict(self, user_id: str,
                ratings: pd.DataFrame,                      # not used
                recs_number: int,
                candidate_item_id_list: List = None,        # not used
                feature_selection_algorithm: FeatureSelection = None):
        if self.fullgraph is None:
            return {}
        if feature_selection_algorithm is not None:
            self.set_fullgraph(feature_selection_algorithm.perform(self.fullgraph.graph, ratings=ratings))

        print(self.fullgraph.graph)
        # run the pageRank
        if self.personalized:
            profile = self.extract_profile(user_id)
            scores = nx.pagerank(self.fullgraph.graph, personalization=profile)
        else:
            scores = nx.pagerank(self.fullgraph.graph)
        # clean the results removing user nodes, selected user profile and eventually properties
        scores = self.clean_rank(scores, user_id)
        ks = list(scores.keys())
        ks = ks[:recs_number]
        new_scores = {k: scores[k] for k in scores.keys() if k in ks}

        return new_scores
