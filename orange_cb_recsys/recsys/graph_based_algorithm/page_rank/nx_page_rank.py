from typing import List
import pandas as pd
import networkx as nx

from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.feature_selection import FeatureSelectionAlgorithm
from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.feature_selection_handler import \
    FeatureSelectionHandler
from orange_cb_recsys.recsys.graphs import NXFullGraph
from orange_cb_recsys.utils.const import recsys_logger

from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.page_rank import PageRankAlg


class NXPageRank(PageRankAlg):
    """
    Page Rank algorithm based on the networkx implementation

    The PageRank can be 'personalized', in this case the PageRank will be calculated with Priors.
    Also, since it's a graph based algorithm, it can be done feature selection to the graph before calculating
    any prediction.

    Args:
        personalized (bool): boolean value that specifies if the page rank must be calculated with Priors
            considering the user profile as personalization vector. Default is False
        feature_selection (FeatureSelectionAlgorithm): a FeatureSelection algorithm if the graph needs to be reduced

    """

    def __init__(self, personalized: bool = False, feature_selection: FeatureSelectionAlgorithm = None):
        super().__init__(personalized, feature_selection)

    def rank(self, user_id: str, graph: NXFullGraph, recs_number: int = None, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All items will be ranked.

        One can specify which items must be ranked with the filter_list parameter,
        in this case ONLY items in the filter_list which are present in the graph will be ranked.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items which are present in the graph will be used to calculate the rank.

        If a feature selection algorithm is passed in the constructor, it is performed before calculating
        any prediction

        Most of the time the rank is calculated by calling the predict() method and sorting the ratings
        predicted, but it's abstract since some algorithm may implement some optimizations to calculate
        the rank.

        Args:
            user_id (str): id of the user of which predictions will be calculated
            graph (FullGraph): a FullGraph containing users, items and eventually other categories of nodes
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to predict, if None all unrated items will be predicted
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the score predicted, sorted in descending order by the 'rating' column
        """
        recsys_logger.info("Calculating rank")

        columns = ["to_id", "score"]
        score_frame = pd.DataFrame(columns=columns)

        if graph is None:
            return score_frame
        if self.feature_selection is not None:
            user_target_nodes = list(graph.user_nodes)
            # only items recommendable to the user will be considered in the feature selection process
            if filter_list is None:
                item_target_nodes = [node for node in graph.item_nodes if node not in graph.get_successors(user_id)]
            else:
                item_target_nodes = [node for node in graph.item_nodes if node not in graph.get_successors(user_id) and
                                     node in filter_list]
            graph = FeatureSelectionHandler(self.feature_selection).\
                process_feature_selection_on_fullgraph(graph, user_target_nodes, item_target_nodes)

        # run the pageRank
        if self.personalized:
            # the personalization vector is formed by the nodes that the user voted with their weight
            # + all the other nodes in the graph with weight as the min weight given by the user
            # (This because if a node isn't specified in the personalization vector will have 0 score in page rank)
            profile = self.extract_profile(graph, user_id)
            pers = {node: profile[node] if node in profile else min(set(profile.values()))
                    for node in graph._graph.nodes}

            # pagerank_scipy faster than pagerank or pagerank_numpy
            scores = nx.pagerank_scipy(graph._graph, personalization=pers)
        else:
            # pagerank_scipy faster than pagerank or pagerank_numpy
            scores = nx.pagerank_scipy(graph._graph)

        # clean the results removing user nodes, selected user profile and eventually properties
        if filter_list is not None:
            nodes = list(scores.keys())
            wrapped_index = [nodes.index(filter_str) for filter_str in filter_list if graph.node_exists(filter_str)]
            wrapped_filter = [nodes[i] for i in wrapped_index]
            scores = self.filter_result(scores, wrapped_filter)
        else:
            scores = self.clean_result(graph, scores, user_id)

        score_frame.to_id = [node.value for node in scores.keys()]
        score_frame.score = scores.values()

        score_frame.sort_values(by=["score"], ascending=False, inplace=True)

        rank = score_frame[:recs_number]

        return rank
