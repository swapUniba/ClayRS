import abc
import gc
from copy import deepcopy
from typing import Union, Dict, List

import pandas as pd
from abc import ABC

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Rank, Prediction
from orange_cb_recsys.recsys.methodology import TestRatingsMethodology
from orange_cb_recsys.recsys.graphs.graph import FullDiGraph

from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import UserSkipAlgFit, NotFittedAlg
from orange_cb_recsys.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm
from orange_cb_recsys.recsys.methodology import Methodology
from orange_cb_recsys.utils.const import logger, get_progbar


class RecSys(ABC):
    """
    Abstract class for a Recommender System

    There exists various type of recommender systems, content-based, graph-based, etc. so extend this class
    if another type must be implemented into the framework.

    Every recommender system do its prediction based on a rating frame, containing interactions between
    users and items

    Args:
        rating_frame (pd.DataFrame): a dataframe containing interactions between users and items
    """

    def __init__(self, algorithm: Union[ContentBasedAlgorithm, GraphBasedAlgorithm]):
        self.__alg = algorithm

    @property
    def algorithm(self):
        return self.__alg

    @abc.abstractmethod
    def rank(self, test_set: pd.DataFrame, n_recs: int = None) -> Rank:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, test_set: pd.DataFrame) -> Prediction:
        raise NotImplementedError


class ContentBasedRS(RecSys):
    """
    Class for recommender systems which use the items' content in order to make predictions,
    some algorithms may also use users' content

    Every CBRS differ from each other based the algorithm chosen

    Args:
        algorithm (ContentBasedAlgorithm): the content based algorithm that will be used in order to
            rank or make score prediction
        train_set (pd.DataFrame): a DataFrame containing interactions between users and items
        items_directory (str): the path of the items serialized by the Content Analyzer
        users_directory (str): the path of the users serialized by the Content Analyzer
    """

    def __init__(self,
                 algorithm: ContentBasedAlgorithm,
                 train_set: Ratings,
                 items_directory: str,
                 users_directory: str = None):

        super().__init__(algorithm)
        self.__train_set = train_set
        self.__items_directory = items_directory
        self.__users_directory = users_directory
        self._user_fit_dic = {}

    @property
    def algorithm(self):
        """
        The content based algorithm chosen
        """
        alg: ContentBasedAlgorithm = super().algorithm
        return alg

    @property
    def train_set(self):
        return self.__train_set

    @property
    def items_directory(self):
        """
        Path of the serialized items
        """
        return self.__items_directory

    @property
    def users_directory(self):
        """
        Path of the serialized users
        """
        return self.__users_directory

    def fit(self):
        """
        Method that divides the train set into as many parts as
        there are different users. then it proceeds with the fit
        for each user and saves the result in the dictionary "user_fit_dic"

        """
        items_to_load = set(self.train_set.item_id_column)
        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, items_to_load)

        with get_progbar(set(self.train_set.user_id_column)) as pbar:

            pbar.set_description("Fitting algorithm")
            for user_id in pbar:
                user_train = self.train_set.get_user_interactions(user_id)

                try:
                    user_alg = deepcopy(self.algorithm)
                    user_alg.process_rated(user_train, loaded_items_interface)
                    user_alg.fit()
                    self._user_fit_dic[user_id] = user_alg
                except UserSkipAlgFit as e:
                    warning_message = str(e) + f"\nNo algorithm will be fitted for the user {user_id}"
                    logger.warning(warning_message)
                    self._user_fit_dic[user_id] = None

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return self

    def rank(self, test_set: Ratings, n_recs: int = None, user_id_list: List = None,
             methodology: Union[Methodology, None] = TestRatingsMethodology()) -> Rank:
        """
        Method used to calculate ranking for the user in test set

        If the recs_number is specified, then the rank will contain the top-n items for the users.
        Otherwise the rank will contain all unrated items of the particular users

        if the items evaluated are present for each user, the filter list is calculated, and
        score prediction is executed only for the items inside the filter list.
        Otherwise, score prediction is executed for all unrated items of the particular user

        Args:
            test_set: set of users for which to calculate the rank
            n_recs: number of the top items that will be present in the ranking

        Returns:
            concat_rank: list of the items ranked for each user

        """
        if len(self._user_fit_dic) == 0:
            raise NotFittedAlg("Algorithm not fit! You must call the fit() method first, or fit_rank().")

        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, set())

        rank = []

        with get_progbar(all_users) as pbar:

            for user_id in pbar:
                user_id = str(user_id)
                pbar.set_description(f"Computing rank for {user_id}")
                user_train = self.train_set.get_user_interactions(user_id)

                filter_list = None
                if methodology is not None:
                    filter_list = set(methodology.filter_single(user_id, self.train_set, test_set))

                user_fitted_alg = self._user_fit_dic.get(user_id)
                if user_fitted_alg is not None:
                    user_rank = user_fitted_alg.rank(user_train, loaded_items_interface,
                                                     n_recs, filter_list=filter_list)
                else:
                    user_rank = []
                    logger.warning(f"No algorithm fitted for user {user_id}! It will be skipped")

                rank.extend(user_rank)

        rank = Rank.from_list(rank)

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return rank

    def predict(self, test_set: Ratings, user_id_list: List = None,
                methodology: Union[Methodology, None] = TestRatingsMethodology()) -> Prediction:
        """
        Method to call when score prediction must be done for the users in test set

        If the items evaluated are present for each user, the filter list is calculated, and
        score prediction is executed only for the items inside the filter list.
        Otherwise, score prediction is executed for all unrated items of the particular user

        Args:
            test_set: set of users for which to calculate the predictions

        Returns:
            concat_score_preds: prediction for each user

        """
        if len(self._user_fit_dic) == 0:
            raise NotFittedAlg("Algorithm not fit! You must call the fit() method first, or fit_rank().")

        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, set())

        pred = []

        with get_progbar(all_users) as pbar:

            for user_id in pbar:
                user_id = str(user_id)
                pbar.set_description(f"Computing predictions for {user_id}")

                user_train = self.train_set.get_user_interactions(user_id)

                filter_list = None
                if methodology is not None:
                    filter_list = set(methodology.filter_single(user_id, self.train_set, test_set))

                user_fitted_alg = self._user_fit_dic.get(user_id)
                if user_fitted_alg is not None:
                    user_pred = user_fitted_alg.predict(user_train, loaded_items_interface,
                                                        filter_list=filter_list)
                else:
                    user_pred = []
                    logger.warning(f"No algorithm fitted for user {user_id}! It will be skipped")

                pred.extend(user_pred)

        pred = Prediction.from_list(pred)

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return pred

    def fit_predict(self, test_set: Ratings, user_id_list: List = None,
                    methodology: Union[Methodology, None] = TestRatingsMethodology()) -> Prediction:
        """
        The method fits the algorithm and then calculates the prediction for each user

        Args:
            test_set: set of users for which to calculate the prediction

        Returns:
            prediction: prediction for each user

        """
        self.fit()
        return self.predict(test_set, user_id_list, methodology)

    def fit_rank(self, test_set: Ratings, n_recs: int = None, user_id_list: List = None,
                 methodology: Union[Methodology, None] = TestRatingsMethodology()) -> Rank:
        """
        The method fits the algorithm and then calculates the rank for each user

        Args:
            test_set: set of users for which to calculate the rank
            n_recs: number of the top items that will be present in the ranking

        Returns:
            rank: ranked items for each user

        """
        self.fit()
        return self.rank(test_set, n_recs, user_id_list, methodology)


class GraphBasedRS(RecSys):
    """
    Class for recommender systems which use a graph in order to make predictions

    Every graph based recommender system differ from each other based the algorithm chosen

    Args:
        algorithm (GraphBasedAlgorithm): the graph based algorithm that will be used in order to
            rank or make score prediction
        graph (FullGraph): a FullGraph containing interactions
    """

    def __init__(self,
                 algorithm: GraphBasedAlgorithm,
                 graph: FullDiGraph):

        self.__graph = graph
        super().__init__(algorithm)

    @property
    def users(self):
        return self.__graph.user_nodes

    @property
    def graph(self):
        """
        The graph containing interactions
        """
        return self.__graph

    @property
    def algorithm(self):
        """
        The content based algorithm chosen
        """
        alg: GraphBasedAlgorithm = super().algorithm
        return alg

    def predict(self, test_set: Ratings, user_id_list: List = None,
                methodology: Union[Methodology, None] = TestRatingsMethodology()):
        """
        Method used to predict the rating of the users

        If the items evaluated are present for each user, the filter list is calculated, and
        score prediction is executed only for the items inside the filter list.
        Otherwise, score prediction is executed for all unrated items of the particular user

        Args:
            test_set: set of users for which to calculate the predictions

        Returns:
            concate_score_preds: list of predictions for each user

        """
        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        filter_dict: Union[Dict, None] = None
        if methodology is not None:
            train_set = self.graph.to_ratings()
            filter_dict = methodology.filter_all(train_set, test_set, result_as_iter_dict=True)

        total_predict_list = self.algorithm.predict(all_users, self.graph, filter_dict)

        total_predict = Prediction.from_list(total_predict_list)

        return total_predict

    def rank(self, test_set: Ratings, n_recs: int = None, user_id_list: List = None,
             methodology: Union[Methodology, None] = TestRatingsMethodology()):
        """
        Method used to rank the rating of the users

        If the items evaluated are present for each user, the filter list is calculated, and
        score prediction is executed only for the items inside the filter list.
        Otherwise, score prediction is executed for all unrated items of the particular user

        Args:
            test_set:  set of users for which to calculate the rank
            n_recs:  number of the top items that will be present in the ranking

        Returns:
            concate_rank: list of the items ranked for each user

        """
        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        filter_dict: Union[Dict, None] = None
        if methodology is not None:
            train_set = self.graph.to_ratings()
            filter_dict = methodology.filter_all(train_set, test_set, result_as_iter_dict=True)

        total_rank_list = self.algorithm.rank(all_users, self.graph, n_recs, filter_dict)

        total_rank = Rank.from_list(total_rank_list)

        return total_rank
