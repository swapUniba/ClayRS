from __future__ import annotations
import itertools
from typing import List, Set, Any, Union, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings
    from clayrs.recsys.graphs import NXBipartiteGraph
    from clayrs.recsys.methodology import Methodology

from clayrs.content_analyzer.ratings_manager.ratings import Interaction
from clayrs.recsys.graph_based_algorithm.page_rank.page_rank import PageRank
from clayrs.recsys.graphs.graph import UserNode, ItemNode
from clayrs.recsys.methodology import TestRatingsMethodology
from clayrs.utils.context_managers import get_iterator_parallel


class NXPageRank(PageRank):
    """
    Page Rank algorithm based on the networkx implementation.
    Please note that it can only be used for instantiated NXGraphs

    The PageRank can be ***personalized***, in this case the PageRank will be calculated with a personalization vector
    made by items in the user profile weighted by the score given to them.

    Args:
        alpha: Damping parameter for PageRank, default=0.85.
        personalized: Boolean value that specifies if the page rank must be calculated considering the user profile
            as personalization vector. Default is False
        max_iter: Maximum number of iterations in power method eigenvalue solver.
        tol: Error tolerance used to check convergence in power method solver.
        nstart: Starting value of PageRank iteration for each node.
        weight: Boolean value which tells the algorithm if weight of the edges must be considered or not.
            Default is True
    """

    def __init__(self, alpha: Any = 0.85, personalized: bool = False, max_iter: Any = 100, tol: Any = 1.0e-6,
                 nstart: Any = None, weight: bool = True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.nstart = nstart
        self.weight = weight

        super().__init__(personalized)

    def rank(self, all_users: Set[str], graph: NXBipartiteGraph, test_set: Ratings,
             recs_number: int = None, methodology: Union[Methodology, None] = TestRatingsMethodology(),
             num_cpus: int = 0) -> List[Interaction]:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items for the user will be ranked (or only items in the filter list, if specified).

        One can specify which items must be ranked for each user with the `filter_dict` parameter,
        in this case every user is mapped with a list of items for which a ranking score must be computed.
        Otherwise, **ALL** unrated items will be ranked for each user.

        Args:
            all_users: Set of user id for which a recommendation list must be generated
            graph: A NX graph previously instantiated
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            test_set: Ratings object which represents the ground truth of the split considered
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Returns:
            List of Interactions object in a descending order w.r.t the 'score' attribute, representing the ranking for
                a single user
        """

        def compute_single_rank(user_id):

            # nonlocal keyword allows to modify the score variable
            nonlocal scores

            user_node = UserNode(user_id)

            filter_list = None
            if methodology is not None:
                filter_list = set(ItemNode(item_to_rank) for item_to_rank in
                                  methodology.filter_single(user_id, train_set, test_set))

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
                        for node in graph.to_networkx().nodes}

                scores = nx.pagerank(graph.to_networkx(), personalization=pers, alpha=self.alpha,
                                     max_iter=self.max_iter, tol=self.tol, nstart=self.nstart, weight=weight)

            # if scores is None it means this is the first time we are running normal pagerank
            # for all the other users the pagerank won't be computed again
            elif scores is None:
                scores = nx.pagerank(graph.to_networkx(), alpha=self.alpha, max_iter=self.max_iter,
                                     tol=self.tol, nstart=self.nstart, weight=weight)

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

            return user_id, single_rank_interaction_list

        # scores will contain pagerank scores
        scores = None
        all_rank_interaction_list = []
        weight = 'weight' if self.weight is True else None
        train_set = graph.to_ratings()

        with get_iterator_parallel(num_cpus,
                                   compute_single_rank, all_users,
                                   progress_bar=True, total=len(all_users)) as pbar:

            pbar.set_description("Prepping rank...")

            for user_id, user_rank in pbar:
                all_rank_interaction_list.append(user_rank)
                pbar.set_description(f"Computing rank for user {user_id}")

        return list(itertools.chain.from_iterable(all_rank_interaction_list))

    def __str__(self):
        return "NXPageRank"

    def __repr__(self):
        return f"NXPageRank(alpha={self.alpha}, personalized={self._personalized}, max_iter={self.max_iter}, " \
               f"tol={self.tol}, nstart={self.nstart}, weight={self.weight})"
