from typing import List, Set, Dict, Any
import networkx as nx

from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Interaction
from orange_cb_recsys.recsys.graphs import NXBipartiteGraph

from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.page_rank import PageRank
from orange_cb_recsys.recsys.graphs.graph import UserNode, ItemNode
from orange_cb_recsys.utils.const import get_progbar


class NXPageRank(PageRank):
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
    def __init__(self, alpha: Any = 0.85, personalized: bool = False, max_iter: Any = 100, tol: Any = 1.0e-6,
                 nstart: Any = None, weight: bool = True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.nstart = nstart
        self.weight = 'weight' if weight is True else None

        super().__init__(personalized)

    def rank(self, all_users: Set[str], graph: NXBipartiteGraph, recs_number: int = None,
             filter_dict: Dict[str, Set] = None) -> List[Interaction]:

        # scores will contain pagerank scores
        scores = None
        all_rank_interaction_list = []

        with get_progbar(all_users) as pbar:

            for user_id in pbar:
                pbar.set_description(f"Computing rank for {user_id}")

                filter_list = None
                if filter_dict is not None:
                    filter_list = [ItemNode(item_to_rank) for item_to_rank in filter_dict.pop(user_id)]

                user_node = UserNode(user_id)

                # run the pageRank
                if self._personalized is True:
                    # the personalization vector is formed by the nodes that the user voted with their weight
                    # + all the other nodes in the graph with weight as the min weight given by the user
                    # (This because if a node isn't specified in the personalization vector will have 0 score in page
                    # rank)
                    succ = graph.get_successors(user_node)
                    profile = {scored_node: graph.get_link_data(user_node, scored_node).get('weight')
                               for scored_node in succ
                               if graph.get_link_data(user_node, scored_node).get('weight') is not None}

                    pers = {node: profile[node] if node in profile else min(set(profile.values()))
                            for node in graph._graph.nodes}

                    scores = nx.pagerank(graph._graph, personalization=pers, alpha=self.alpha, max_iter=self.max_iter,
                                         tol=self.tol, nstart=self.nstart, weight=self.weight)

                # if scores is None it means this is the first time we are running normal pagerank
                # for all the other users the pagerank won't be computed again
                elif scores is None:
                    scores = nx.pagerank(graph._graph, alpha=self.alpha, max_iter=self.max_iter,
                                         tol=self.tol, nstart=self.nstart, weight=self.weight)

                # clean the results removing user nodes, selected user profile and eventually properties
                user_scores = self.filter_result(graph, scores, filter_list, user_node)

                # Build the item_score dict (key is item_id, value is rank score predicted)
                # and order the keys in descending order
                item_score_dict = dict(zip([node.value for node in user_scores.keys()], user_scores.values()))
                ordered_item_ids = sorted(item_score_dict, key=item_score_dict.get, reverse=True)

                # we only save the top-n items_ids corresponding to top-n recommendations
                # (if recs_number is None ordered_item_ids will contain all item_ids as the original list)
                ordered_item_ids = ordered_item_ids[:recs_number]

                # we construct the output data
                single_rank_interaction_list = [Interaction(user_id, item_id, item_score_dict[item_id])
                                                for item_id in ordered_item_ids]

                all_rank_interaction_list.extend(single_rank_interaction_list)

        return all_rank_interaction_list
