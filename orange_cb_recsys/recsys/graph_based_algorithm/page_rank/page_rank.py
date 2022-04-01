from typing import List, Set, Dict

from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Interaction
from orange_cb_recsys.recsys.graphs import NXBipartiteGraph

from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm


class PageRank(GraphBasedAlgorithm):
    """
    Abstract class that contains the main methods and attributes for any PageRank algorithm.

    Every PageRank algorithm can be 'personalized', in this case the PageRank will be calculated with Priors.
    Also, since it's a graph based algorithm, it can be done feature selection to the graph before calculating
    any prediction.

    Args:
        personalized (bool): boolean value that specifies if the page rank must be calculated with Priors
            considering the user profile as personalization vector. Default is False
        feature_selection (FeatureSelectionAlgorithm): a FeatureSelection algorithm if the graph needs to be reduced
    """

    def __init__(self, personalized: bool = False):

        self._personalized = personalized
        super().__init__()

    def predict(self, all_users: Set[str], graph: NXBipartiteGraph,
                filter_dict: Dict[str, Set] = None) -> List[Interaction]:
        """
        PageRank is not a Prediction Score Algorithm, so if this method is called,
        a NotPredictionAlg exception is raised

        Raises:
            NotPredictionAlg
        """
        raise NotPredictionAlg("PageRank is not a Score Prediction Algorithm!")
