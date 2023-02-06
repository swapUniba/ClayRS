from __future__ import annotations
from typing import Set, Any, TYPE_CHECKING, Optional

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings
    from clayrs.recsys.graphs import NXBipartiteGraph
    from clayrs.recsys.methodology import Methodology

from clayrs.recsys.graph_based_algorithm.page_rank.page_rank import PageRank
from clayrs.recsys.graphs.graph import UserNode, ItemNode
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

    def rank(self, graph: NXBipartiteGraph, train_set: Ratings, test_set: Ratings, user_id_list: Set[str],
             recs_number: Optional[int], methodology: Methodology, num_cpus: int):
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

        def compute_single_rank(user_tuple):

            # nonlocal keyword allows to modify the score variable
            nonlocal scores

            user_id, user_idx = user_tuple
            user_node = UserNode(user_id)

            filter_list = set(ItemNode(item_to_rank)
                              for item_to_rank in
                              train_set.item_map.convert_seq_int2str(methodology.filter_single(user_idx,
                                                                                               train_set,
                                                                                               test_set)))

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
                        for node in networkx_graph.nodes}

                scores = nx.pagerank(networkx_graph, personalization=pers, alpha=self.alpha,
                                     max_iter=self.max_iter, tol=self.tol, nstart=self.nstart, weight=weight)

            # if scores is None it means this is the first time we are running normal pagerank
            # for all the other users the pagerank won't be computed again
            elif scores is None:
                scores = nx.pagerank(networkx_graph, alpha=self.alpha, max_iter=self.max_iter,
                                     tol=self.tol, nstart=self.nstart, weight=weight)

            # clean the results removing user nodes, selected user profile and eventually properties
            user_scores = self.filter_result(graph, scores, filter_list, user_node)

            if len(user_scores) == 0:
                return user_id, np.array([])  # if no item to predict, empty rank is returned

            user_scores_arr = np.array(list(user_scores.items()))

            sorted_scores_idxs = np.argsort(user_scores_arr[:, 1])[::-1][:recs_number]
            user_scores_arr = user_scores_arr[sorted_scores_idxs]

            user_col = np.full((user_scores_arr.shape[0], 1), user_id)
            uir_rank = np.append(user_col, user_scores_arr, axis=1)

            return user_id, uir_rank

        # scores will contain pagerank scores
        scores = None
        all_rank_uirs_list = []
        weight = 'weight' if self.weight is True else None
        networkx_graph = graph.to_networkx()
        user_idxs_list = train_set.user_map.convert_seq_str2int(list(user_id_list))

        with get_iterator_parallel(num_cpus,
                                   compute_single_rank, zip(user_id_list, user_idxs_list),
                                   progress_bar=True, total=len(user_id_list)) as pbar:

            pbar.set_description("Prepping rank...")

            for user_id, user_rank in pbar:
                all_rank_uirs_list.append(user_rank)
                pbar.set_description(f"Computing rank for user {user_id}")

        return all_rank_uirs_list

    def __str__(self):
        return "NXPageRank"

    def __repr__(self):
        return f"NXPageRank(alpha={self.alpha}, personalized={self._personalized}, max_iter={self.max_iter}, " \
               f"tol={self.tol}, nstart={self.nstart}, weight={self.weight})"
