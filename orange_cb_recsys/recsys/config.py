from abc import ABC

from orange_cb_recsys.recsys.algorithm import Algorithm
from orange_cb_recsys.recsys.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm
from orange_cb_recsys.recsys.graphs.graph import FullGraph

import pandas as pd


class RecSysConfig(ABC):
    """
    Configuration for the recommender system.
    Every type of RecSys needs to specify an algorithm to use.

    Args:
        algorithm (Algorithm): algorithm to use
    """

    def __init__(self, algorithm: Algorithm):
        self.__alg = algorithm

    @property
    def algorithm(self):
        return self.__alg


class ContentBasedConfig(RecSysConfig):
    """
    Configuration for a content-based recommender system
    Args:
        algorithm (ContentBasedAlgorithm): content-based algorithm to use
        rating_frame (pd.DataFrame): DataFrame containing all the ratings of the users
        items_directory (str): Path to the directory in which the items are stored
        users_directory (str): Path to the directory in which the users are stored. It's optional since
            many content-based algorithms don't use users information
    """
    def __init__(self,
                 algorithm: ContentBasedAlgorithm,
                 rating_frame: pd.DataFrame,
                 items_directory: str,
                 users_directory: str = None):
        self.__rating_frame = rating_frame
        self.__items_directory = items_directory
        self.__users_directory = users_directory
        super().__init__(algorithm)
        # Pass arguments to the algorithm
        algorithm.initialize(rating_frame, items_directory, users_directory)

    @property
    def users_directory(self):
        return self.__users_directory

    @property
    def items_directory(self):
        return self.__items_directory

    @property
    def rating_frame(self):
        return self.__rating_frame


class GraphBasedConfig(RecSysConfig):
    """
    Configuration for a graph-based recommender system
    Args:
        algorithm (GraphBasedAlgorithm): graph-based algorithm to use
        graph (FullGraph): graph to use for calculating prediction and recommendation
    """
    def __init__(self,
                 algorithm: GraphBasedAlgorithm,
                 graph: FullGraph):
        self.__graph = graph
        super().__init__(algorithm)
        # Pass arguments to the algorithm
        algorithm.initialize(graph)

    @property
    def graph(self):
        return self.__graph
