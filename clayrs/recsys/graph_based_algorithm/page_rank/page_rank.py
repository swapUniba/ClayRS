from __future__ import annotations
from typing import List, Set, TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings
    from clayrs.recsys.graphs import NXBipartiteGraph
    from clayrs.recsys.methodology import Methodology

from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm


class PageRank(GraphBasedAlgorithm):
    """
    Abstract class that contains the main methods and attributes for any PageRank algorithm.

    Every PageRank algorithm should also offer the possibility to compute the `personalized` page rank with different
    weights for different types of nodes

    Args:
        personalized: Boolean value that specifies if the page rank must be calculated considering the user profile
            as personalization vector. Default is False
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

    def __init__(self,
                 personalized: bool = False,
                 relevance_threshold: float = None,
                 rel_items_weight: float = 0.8,
                 rel_items_prop_weight: Optional[float] = None,
                 default_nodes_weight: Optional[float] = None
                 ):

        super().__init__()

        self._personalized = personalized
        self._relevance_threshold = relevance_threshold

        rel_weight, prop_weight, other_weight = self.check_weights(rel_items_weight,
                                                                   rel_items_prop_weight,
                                                                   default_nodes_weight)
        self._rel_items_weight = rel_weight
        self._rel_items_prop_weight = prop_weight
        self._default_nodes_weight = other_weight

    @staticmethod
    def check_weights(rel_items_weight, rel_items_prop_weight, default_nodes_weight):
        """
        Simple static method which checks that the weights respect the following constraints:

        * Each *probability weight* should be between $0$ (included) and $1$ (included)
        * The sum of all *probability weights* should be exactly one (if all *probability weights* are set)

        If some *probability weights* are None, then they will be set accordingly (so that the sum of all *probability
        weights* is 1)
        """

        all_probs_weight = [rel_items_weight, rel_items_prop_weight, default_nodes_weight]

        # check that single probs are between 0 and 1
        for prob_weight in all_probs_weight:
            if prob_weight is not None and not 0 <= prob_weight <= 1:
                raise ValueError(f"All weight probabilities should be in range [0, 1]! Got {prob_weight}")

        total_prob = sum(filter(None, all_probs_weight))
        n_weight_wo_prob = all_probs_weight.count(None)
        if total_prob > 1:
            # we go here if the user set all weight probs but sum is < 1 or > 1
            raise ValueError(f"The sum of the weight probabilities should be 1, got {total_prob} instead!")

        elif n_weight_wo_prob > 0:

            prob_to_assign = (1 - total_prob) / n_weight_wo_prob

            if rel_items_weight is None:
                rel_items_weight = prob_to_assign

            if rel_items_prop_weight is None:
                rel_items_prop_weight = prob_to_assign

            if default_nodes_weight is None:
                default_nodes_weight = prob_to_assign

        return rel_items_weight, rel_items_prop_weight, default_nodes_weight

    def predict(self, graph: NXBipartiteGraph, train_set: Ratings, test_set: Ratings, user_id_list: Set[str],
                methodology: Methodology, num_cpus: int) -> List[np.ndarray]:
        """
        PageRank is not a Prediction Score Algorithm, so if this method is called,
        a NotPredictionAlg exception is raised

        Raises:
            NotPredictionAlg: exception raised since the PageRank algorithm is not a score prediction algorithm
        """
        raise NotPredictionAlg("PageRank is not a Score Prediction Algorithm!")
