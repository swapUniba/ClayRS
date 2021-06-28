import abc
from typing import Dict, List

from orange_cb_recsys.recsys.algorithm import Algorithm

from orange_cb_recsys.utils.feature_selection import FeatureSelection

from orange_cb_recsys.recsys.graphs.graph import FullGraph

import pandas as pd


class GraphBasedAlgorithm(Algorithm):
    """
    Abstract class for the graph-based algorithms

    Like every subclass of Algorithm, it must implements the 'initialize(...)' method where one must pass
    important parameters for the usage of this specific type of Algorithm

    Said method must be called right after the instantiation of the Algorithm

    Args:
    feature_selection (FeatureSelection): a FeatureSelection algorithm if the graph needs to be reduced
    """
    def __init__(self, feature_selection: FeatureSelection = None):
        self.__feature_selection: FeatureSelection = feature_selection

    @property
    def feature_selection(self):
        return self.__feature_selection

    @feature_selection.setter
    def feature_selection(self, feature_selection: FeatureSelection):
        self.__feature_selection = feature_selection

    def clean_result(self, graph: FullGraph, result: Dict, user_id: str,
                     remove_users: bool = True,
                     remove_profile: bool = True,
                     remove_properties: bool = True) -> Dict:
        """
        Cleans the result from all the nodes that are not requested.

        It's possible to remove:
        * user nodes (remove_users),
        * item nodes rated by the user (remove_profile),
        * property nodes (remove_properties).

        This produces a cleaned result with only the desired nodes inside of it.
        Args:
            result (dict): dictionary representing the result (keys are nodes and values are their score prediction)
            user_id (str): id of the user used to extract his profile
            remove_users (bool): boolean value, set to true if 'User' nodes need to be removed from the result dict
            remove_profile (bool): boolean value, set to true if 'Item' nodes rated by the user
                need to be removed from the result dict
            remove_properties (bool): boolean value, set to true if 'Property' nodes need to be removed from the
                result dict
        Returns:
            new_result (dict): dictionary representing the cleaned result
        """
        def is_valid(node: object, user_profile):
            valid = True
            if remove_users and graph.is_user_node(node) or \
                remove_profile and node in user_profile or \
                remove_properties and graph.is_property_node(node):

                valid = False

            return valid

        extracted_profile = self.extract_profile(graph, user_id)
        new_result = {k: result[k] for k in result.keys() if is_valid(k, extracted_profile)}

        return new_result

    @staticmethod
    def filter_result(result: Dict, filter_list: List[str]) -> Dict:
        """
        Method which filters the result dict returning only items that are also in the filter_list

        Args:
            result (dict): dictionary representing the result (keys are nodes and values are their score prediction)
            filter_list (list): list of the items to predict, if None all unrated items will be predicted
        """

        filtered_result = {k: result[k] for k in result.keys() if k in filter_list}

        return filtered_result

    @staticmethod
    def extract_profile(graph: FullGraph, user_id: str) -> Dict:
        """
        Extracts the user profile (the items that the user rated, or in general the nodes with a link to the user).

        Returns a dictionary containing the successor nodes as keys and the weights in the graph for the edges between the user node
        and his successors as values

        EXAMPLE::
             graph: i1 <---0.2--- u1 ---0.4---> i2

            > print(extract_profile('u1'))
            > {'i1': 0.2, 'i2': 0.4}

        Args:
            user_id (str): id for the user for which the profile will be extracted
        Returns:
            profile (dict): dictionary with item successor nodes to the user as keys and weights of the edge
                connecting them in the graph as values
        """
        succ = graph.get_successors(user_id)
        profile = {}
        for a in succ:
            link_data = graph.get_link_data(user_id, a)
            profile[a] = link_data['weight']
        return profile  # {t: w for (f, t, w) in adj}

    @abc.abstractmethod
    def predict(self, user_id: str, graph: FullGraph, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Abstract method that predicts how much a user will like unrated items.

        One can specify which items must be predicted with the filter_list parameter,
        in this case ONLY items in the filter_list which are present in the graph will be predicted.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items which are present in the graph will be predicted.

        If a feature selection algorithm is passed in the constructor, it is performed before calculating
        any prediction

        Args:
            user_id (str): id of the user of which predictions will be calculated
            graph (FullGraph): a FullGraph containing users, items and eventually other categories of nodes
            filter_list (list): list of the items to predict, if None all unrated items will be predicted
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the score predicted
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, user_id: str, graph: FullGraph, recs_number: int = None, filter_list: List[str] = None) -> pd.DataFrame:
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
        raise NotImplementedError
