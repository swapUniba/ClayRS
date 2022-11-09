from __future__ import annotations
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Interaction, Ratings
    from clayrs.recsys.graphs import NXBipartiteGraph
    from clayrs.recsys.methodology import Methodology

from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm
from clayrs.recsys.methodology import TestRatingsMethodology


class PageRank(GraphBasedAlgorithm):
    """
    Abstract class that contains the main methods and attributes for any PageRank algorithm.

    Every PageRank algorithm can be `personalized`, in this case the PageRank will be calculated with a personalization
    vector made by items in the user profile weighted by the score given to them.

    Args:
        personalized (bool): boolean value that specifies if the page rank must be calculated considering the user
            profile as personalization vector. Default is False
    """

    def __init__(self, personalized: bool = False):

        self._personalized = personalized
        super().__init__()

    def predict(self, all_users: Set[str], graph: NXBipartiteGraph, test_set: Ratings,
                methodology: Methodology = TestRatingsMethodology(),
                num_cpus: int = 1) -> List[Interaction]:
        """
        PageRank is not a Prediction Score Algorithm, so if this method is called,
        a NotPredictionAlg exception is raised

        Raises:
            NotPredictionAlg: exception raised since the PageRank algorithm is not a score prediction algorithm
        """
        raise NotPredictionAlg("PageRank is not a Score Prediction Algorithm!")
