from typing import List, Dict
import networkx as nx
import pandas as pd
from abc import abstractmethod

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.recsys.graphs.graph import FullGraph
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
            if remove_from_nodes and self.__fullgraph.is_user_node(k):
                new_rank.pop(k)
            if remove_profile and self.__fullgraph.is_item_node(k) and k in extracted_profile.keys():
                new_rank.pop(k)
            if remove_properties and self.__fullgraph.is_property_node(k):
                new_rank.pop(k)
        return new_rank

    def extract_profile(self, user_id: str) -> Dict:
        succ = self.__fullgraph.get_successors(user_id)
        profile = {}
        # logger.info('unpack %s', str(adj))
        for a in succ:
            # logger.info('unpack %s', str(a))
            link_data = self.__fullgraph.get_link_data(user_id, a)
            profile[a] = link_data['weight']
            logger.info('unpack %s, %s', str(a), str(profile[a]))
        return profile  # {t: w for (f, t, w) in adj}


class NXPageRank(PageRankAlg):

    def __init__(self, graph: NXFullGraph = None, personalized: bool = False):
        super().__init__(graph=graph, personalized=personalized)

    def predict(self, user_id: str,
                ratings: pd.DataFrame,  # not used
                recs_number: int,
                candidate_item_id_list: List = None,  # not used
                feature_selection_algorithm: FeatureSelection = None):
        if self.fullgraph is None:
            return {}
        if feature_selection_algorithm is not None:
            self.set_fullgraph(feature_selection_algorithm.perform(self.fullgraph._graph, ratings=ratings))

        # run the pageRank
        if self.personalized:
            profile = self.extract_profile(user_id)
            scores = nx.pagerank(self.fullgraph._graph.to_undirected(), personalization=profile)
        else:
            scores = nx.pagerank(self.fullgraph._graph)
        # clean the results removing user nodes, selected user profile and eventually properties
        scores = self.clean_rank(scores, user_id)
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        ks = list(scores.keys())
        ks = ks[:recs_number]
        new_scores = {k: scores[k] for k in scores.keys() if k in ks}

        return new_scores
