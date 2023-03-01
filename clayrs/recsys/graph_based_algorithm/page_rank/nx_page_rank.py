from __future__ import annotations
from typing import Set, Any, TYPE_CHECKING, Optional, List

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings
    from clayrs.recsys.graphs import NXBipartiteGraph
    from clayrs.recsys.methodology import Methodology

from clayrs.recsys.graph_based_algorithm.page_rank.page_rank import PageRank
from clayrs.recsys.graphs.graph import UserNode, ItemNode, PropertyNode
from clayrs.utils.context_managers import get_iterator_parallel


class NXPageRank(PageRank):
    r"""
    Page Rank algorithm based on the networkx implementation.
    Please note that it can only be used for instantiated NXGraphs

    The PageRank can be ***personalized***, in this case the PageRank will be calculated with a personalization vector
    which ***depends by the specific user***.

    !!! info "Personalized PageRank"

        In the PageRank, the *random surfer* has probability $\alpha$ of following one of the out links of the node it's
        in, and probability $1-\alpha$ of visiting another node which is not necessarily linked with an out link
        to the node it's in.

        * In the **classical Page Rank**, all nodes have uniform probability of being picked by the *random surfer* when
          not following the out links
        * In the **personalized Page Rank**, we assign a different probability to certain nodes depending on heuristics
          when not following the out links

        In the recommendation task, the idea is to assign a **higher probability** to item nodes which are *relevant*
        to the user, so that the Page Rank algorithm assigns **higher score** to item nodes *close* to relevant items
        for the user

    Several weighting schemas can be applied in order to customize the personalization vector of a user:

    * 80/20: 80% prob. that the random surfer ends up in a relevant item node, 20% that it will end up in any other node
      **(default)**
    * 60/20/20: 60% prob. that the random surfer ends up in a relevant item node, 20% that it will end up in a property
      node linked to a relevant item, 20% that it will end up in any other node
    * 40/40/20: 40% prob. that the random surfer ends up in a relevant item node, 40% that it will end up in a property
      node linked to a relevant item, 20% that it will end up in any other node
    * ...

    It's important to note that the weight assigned are then **normalized** across each node category: this means that
    if 80% prob. is assigned to relevant items, then this probability is *shared* among all relevant items (i.e.
    80% divided by the total number of relevant items for the user)

    Args:
        alpha: Damping parameter for PageRank, default=0.85.
        personalized: Boolean value that specifies if the page rank must be calculated considering the user profile
            as personalization vector. Default is False
        max_iter: Maximum number of iterations in power method eigenvalue solver.
        tol: Error tolerance used to check convergence in power method solver.
        nstart: Starting value of PageRank iteration for each node.
        weight: Boolean value which tells the algorithm if weight of the edges must be considered or not.
            Default is True
        relevance_threshold: Threshold which separates relevant and non-relevant items for a user. Can be set globally,
            but if None the relevance threshold for each user is computed considering the mean rating given by the user
            in the train set
        rel_items_weight: Probability that the random surfer will end up in a relevant item node when not following the
            out links of a node. This probability will be normalized and divided by the total number of relevant items
            for the user. If None, the `default_nodes_weight` probability will be assigned
        rel_items_prop_weight: Probability that the random surfer will end up in a property node linked to a relevant
            item when not following the out links of a node. This probability will be normalized and divided by the
            total number of property nodes linked to relevant items. If None, the `default_nodes_weight` probability
            will be assigned
        default_nodes_weight: Probability that the random surfer will end up in a node which is not a relevant item
            or a property linked to a relevant item when not following the out links of a node.
            If `rel_items_weight` is None, then also relevant item nodes will be considered as *default nodes*.
            If `rel_items_prop_weight` is None, then also property nodes linked to relevant items will be considered
            as *default nodes*
    """

    def __init__(self, alpha: Any = 0.85,
                 personalized: bool = False,
                 max_iter: Any = 100,
                 tol: Any = 1.0e-6,
                 nstart: Any = None,
                 weight: bool = True,
                 relevance_threshold: float = None,
                 rel_items_weight: Optional[float] = 0.8,
                 rel_items_prop_weight: Optional[float] = None,
                 default_nodes_weight: Optional[float] = 0.2):

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.nstart = nstart
        self.weight = weight

        # basically when the user set a weight to be 0, the personalization vector
        # should take that into consideration. But if the user set None, the weights will also be
        # set automatically to 0, but in this case we should give them the 'default_nodes_weight'!
        # That's why we have these other variables
        self._strict_rel_items_weight = rel_items_weight == 0
        self._strict_rel_items_prop_weight = rel_items_prop_weight == 0

        super().__init__(personalized, relevance_threshold, rel_items_weight, rel_items_prop_weight,
                         default_nodes_weight)

    def rank(self, graph: NXBipartiteGraph, train_set: Ratings, test_set: Ratings, user_id_list: Set[str],
             recs_number: Optional[int], methodology: Methodology, num_cpus: int) -> List[np.ndarray]:
        """
        Rank the top-n recommended items for the user. If the `recs_number` parameter is set to None,
        All unrated items for the user will be ranked among all those selected by the `methodology` parameter.

        The train set contains basically the interactions modelled in the graph, and it is needed by the methodology
        object

        Args:
            graph: A graph which models interactions of users and items
            train_set: a Ratings object containing interactions between users and items
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            test_set: Ratings object which represents the ground truth of the split considered
            user_id_list: List of users for which you want to compute ranking. The list should contain user id as
                strings and NOT user ids mapped to their integers
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            List of uir matrices for each user, where each uir contains predicted interactions between users and unseen
                items sorted in a descending way w.r.t. the third dimension which is the ranked score
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

                user_ratings = train_set.get_user_interactions(user_idx)
                user_relevance_threshold = self._relevance_threshold or np.nanmean(user_ratings[:, 2])

                pers_dict = {}

                relevant_items = user_ratings[np.where(user_ratings[:, 2] >= user_relevance_threshold)][:, 1]
                relevant_items = [ItemNode(item_node)
                                  for item_node in train_set.item_map.convert_seq_int2str(relevant_items.astype(int))]

                # If the prob is > 0 then add relevant nodes to personalization vector. (rel_items_weight True)
                # But also if the prob is 0 and the user explicitly set this prob to 0, add relevant nodes to
                # personalization vector (strict_rel_items_weight is True)
                # But if the prob is 0 and the user set None, we will skip this and add relevant nodes with the
                # default_nodes_weight! (rel_items_weight not True and strict_rel_items_weight not True)
                if self._rel_items_weight or self._strict_rel_items_weight:
                    pers_dict.update({item_node: self._rel_items_weight / len(relevant_items)
                                      for item_node in relevant_items})

                # If the prob is > 0 then add relevant props to personalization vector. (rel_items_prop_weight True)
                # But also if the prob is 0 and the user explicitly set this prob to 0, add relevant props to
                # personalization vector (strict_rel_items_prop_weight is True)
                # But if the prob is 0 and the user set None, we will skip this and add relevant props with the
                # default_nodes_weight! (rel_items_prop_weight False and strict_rel_items_prop_weight False)
                if self._rel_items_prop_weight or self._strict_rel_items_prop_weight:
                    relevant_props = set()
                    for item_node in relevant_items:
                        relevant_item_properties = filter(lambda n: isinstance(n, PropertyNode),
                                                          graph.get_successors(item_node))
                        relevant_props.update(relevant_item_properties)

                    pers_dict.update({prop_node: self._rel_items_prop_weight / len(relevant_props)
                                      for prop_node in relevant_props})

                # all nodes that are not present up until now in the personalization vector, will be added
                # with probability 'default_nodes_weight'
                other_nodes = networkx_graph.nodes - pers_dict.keys()
                pers_dict.update({node: self._default_nodes_weight / len(other_nodes) for node in other_nodes})

                scores = nx.pagerank(networkx_graph, personalization=pers_dict, alpha=self.alpha,
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
               f"tol={self.tol}, nstart={self.nstart}, weight={self.weight}, " \
               f"relevance_threshold={self._relevance_threshold}, rel_items_weight={self._rel_items_weight}, " \
               f"rel_items_prop_weight={self._rel_items_prop_weight}, " \
               f"default_nodes_weight={self._default_nodes_weight})"
