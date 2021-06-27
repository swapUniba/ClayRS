from typing import List
import pandas as pd
from orange_cb_recsys.recsys.graphs.graph import FullGraph

from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.graph_based_algorithm import GraphBasedAlgorithm
from orange_cb_recsys.utils.feature_selection import FeatureSelection


class PageRankAlg(GraphBasedAlgorithm):
    """
    Abstract class that contains the main methods and attributes for any PageRank algorithm.

    Every PageRank algorithm can be 'personalized', in this case the PageRank will be calculated with Priors.
    Also, since it's a graph based algorithm, it can be done feature selection to the graph before calculating
    any prediction.

    Args:
        personalized (bool): boolean value that specifies if the page rank must be calculated with Priors
            considering the user profile as personalization vector. Default is False
        feature_selection (FeatureSelection): a FeatureSelection algorithm if the graph needs to be reduced
    """
    def __init__(self, personalized: bool = False, feature_selection: FeatureSelection = None):
        self.__personalized = personalized
        super().__init__(feature_selection)

    @property
    def personalized(self):
        return self.__personalized

    @personalized.setter
    def personalized(self, personalized: bool):
        self.__personalized = personalized

    def predict(self, user_id: str, graph: FullGraph, filter_list: List[str] = None) -> pd.DataFrame:
        """
        PageRank is not a Prediction Score Algorithm, so if this method is called,
        a NotPredictionAlg exception is raised

        Raises:
            NotPredictionAlg
        """
        raise NotPredictionAlg("PageRank is not a Score Prediction Algorithm!")
