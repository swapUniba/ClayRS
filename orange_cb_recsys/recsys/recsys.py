import abc
from itertools import chain
from typing import Union, Iterable, Dict, Optional
from typing import Set

import pandas as pd
from abc import ABC

from orange_cb_recsys.recsys.methodology import TestRatingsMethodology
from orange_cb_recsys.recsys.algorithm import Algorithm
from orange_cb_recsys.recsys.graphs.graph import FullGraph

from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import UserSkipAlgFit, NotFittedAlg
from orange_cb_recsys.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm
from orange_cb_recsys.recsys.methodology import Methodology
from orange_cb_recsys.utils.const import logger, progbar


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

    def __init__(self, algorithm: Algorithm):
        self.__alg = algorithm

    @property
    def algorithm(self):
        return self.__alg

    @property
    @abc.abstractmethod
    def users(self):
        """
        Users of the recommender systems (users that have at least one interaction in the rating_frame)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, test_set: pd.DataFrame, n_recs: int = None):
        # test_set has columns = from_id, to_id, score
        # from_id should only values that are only present in self.users, otherwise exception is thrown.
        # It's not necessary that all users are present in 'from_id' column of the test_set, but if a user is in
        # 'test_set' must also exist in self.users.
        #
        # If test_set is a DataFrame with only the 'from_id' column, then for all users in 'from_id' all unranked items
        # must be ranked.
        # If test_set is a DataFrame with the 'from_id' column and 'to_id' column, then for all users in 'from_id' the
        # rank method of the algorithm must be called passing as a filter list all items in its test_set.
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, test_set: pd.DataFrame):
        # test_set has columns = from_id, to_id, score
        raise NotImplementedError

    # @abc.abstractmethod
    # def fit_predict(self, user_id: str, filter_list: List[str] = None):
    #     """
    #     Method to call when score prediction must be done for the specified user_id
    #
    #     By default, score prediction will be done for all unrated items by the user, unless the
    #     filter_list parameter is specified.
    #
    #     If the filter_list parameter is specified, then score_prediction is executed only for the
    #     items inside the filter_list
    #     """
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None):
    #     """
    #     Method to call when ranking must be calculated for the specified user_id
    #
    #     If the recs_number parameter is specified, only the top-n recommendation will be returned.
    #     Otherwise ranking of all the unrated items by the user will be returned
    #
    #     If the filter_list parameter is specified, then the ranking is calculated only for the
    #     items inside the filter list
    #     """
    #     raise NotImplementedError
    #
    # def multiple_fit_rank(self, user_id_list: List[str], recs_number: int = None, filter_list: List[str] = None):
    #     """
    #     Method used to calculate ranking for a list of users
    #
    #     The method fits the algorithm (when eligible) and then calculates the rank.
    #
    #     If the recs_number is specified, then the rank will contain the top-n items for the particular user.
    #     Otherwise the rank will contain all unrated items of the particular user
    #
    #     If the filter_list parameter is specified, score prediction is executed only for the items
    #     inside the filter list. Otherwise, score prediction is executed for all unrated items of the
    #     particular user
    #
    #     Args:
    #         user_id_list (List[str]): list of all the users ids of which ranking must be calculated
    #         recs_number (int): number of the top items that will be present in the ranking
    #         filter_list (list): list of the items to rank, if None all unrated items of the particular user
    #             will be ranked
    #     Returns:
    #         pd.DataFrame: DataFrame containing ranking for all users specified
    #     """
    #     rank_list = []
    #     for user_id in user_id_list:
    #         rank = self.fit_rank(user_id, recs_number, filter_list)
    #
    #         rank_list.append(rank)
    #
    #     concat_rank = pd.concat(rank_list)
    #     return concat_rank
    #
    # def multiple_fit_predict(self, user_id_list: List[str], filter_list: List[str] = None):
    #     """
    #     Method used to calculate score prediction for a list of users
    #
    #     The method fits the algorithm (when eligible) and then calculates the prediction
    #
    #     If the filter_list parameter is specified, score prediction is executed only for the items
    #     inside the filter list. Otherwise, score prediction is executed for all unrated items of the
    #     particular user
    #
    #     Args:
    #         user_id_list (List[str]): list of all the users ids of which score prediction must be calculated
    #         filter_list (list): list of the items to score predict, if None all unrated items of the particular user
    #             will be score predicted
    #     Returns:
    #         pd.DataFrame: DataFrame containing score predictions for all users specified
    #     """
    #     score_preds_list = []
    #     for user_id in user_id_list:
    #     #         score_preds = self.fit_predict(user_id, filter_list)
    #     #
    #     #         score_preds_list.append(score_preds)
    #     #
    #     #     concat_score_preds = pd.concat(score_preds_list)
    #     return concat_score_preds
    #
    # @abc.abstractmethod
    # def _eval_fit_predict(self, train_ratings: pd.DataFrame, test_set_items: List[str]):
    #     """
    #     Private method used by the evaluation module
    #
    #     It calculates score prediction on the test set items based on the train ratings items
    #     """
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def _eval_fit_rank(self, train_ratings: pd.DataFrame, test_set_items: List[str]):
    #     """
    #     Private method used by the evaluation module
    #
    #     It calculates ranking on the test set items based on the train ratings items
    #     """
    #     raise NotImplementedError


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
                 train_set: pd.DataFrame,
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
    def users(self):
        return set(self.train_set['from_id'])

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
        items_to_load = set(self.train_set['to_id'].values)
        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, items_to_load)

        for user_id in progbar(set(self.train_set['from_id']), prefix="Fitting algorithm:"):
            user_train = self.train_set[self.train_set['from_id'] == user_id]

            try:
                user_alg = self.algorithm.copy()
                user_alg.process_rated(user_train, loaded_items_interface)
                user_alg.fit()
                self._user_fit_dic[user_id] = user_alg
            except UserSkipAlgFit as e:
                warning_message = str(e) + f"\nNo algorithm will be fitted for the user {user_id}"
                logger.warning(warning_message)
                self._user_fit_dic[user_id] = None

    def rank(self, test_set: Union[pd.DataFrame, Iterable], n_recs: int = None,
             methodology: Methodology = TestRatingsMethodology()) -> pd.DataFrame:
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

        all_users = set(test_set)
        items_to_load = None
        filter_dict: Optional[Dict] = None
        if hasattr(test_set, "columns") and 'to_id' in test_set.columns:
            filter_dict = methodology.filter_all(self.train_set, test_set, result_as_dict=True)
            items_to_load = set(chain(*filter_dict.values()))
            all_users = set(filter_dict.keys())

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, items_to_load)

        rank_list = []

        for user_id in progbar(all_users, prefix="Computing rank for user {}:", substitute_with_current=True):
            user_train = self.train_set[self.train_set['from_id'] == user_id]
            user_seen_items = set(user_train['to_id'])

            filter_list = None
            if filter_dict is not None:
                filter_list = filter_dict.get(user_id)

            user_fitted_alg = self._user_fit_dic.get(user_id)
            if user_fitted_alg is not None:
                rank = user_fitted_alg.rank(user_seen_items, loaded_items_interface,
                                            n_recs, filter_list=filter_list)
                rank.insert(0, 'from_id', user_id)
            else:
                rank = pd.DataFrame({'from_id': [], 'to_id': [], 'score': []})
                logger.warning(f"No algorithm fitted for user {user_id}! It will be skipped")

            rank_list.append(rank)

        concat_rank = pd.concat(rank_list)
        return concat_rank

    def predict(self, test_set: Union[pd.DataFrame, Iterable], methodology: Methodology = TestRatingsMethodology()):
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

        all_users = set(test_set)
        items_to_load = None
        filter_dict: Optional[Dict] = None
        if hasattr(test_set, "columns") and 'to_id' in test_set.columns:
            filter_dict: Dict = methodology.filter_all(self.train_set, test_set, result_as_dict=True)
            items_to_load = set(chain(*filter_dict.values()))
            all_users = set(filter_dict.keys())

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, items_to_load)

        pred_list = []

        for user_id in progbar(all_users, prefix="Computing score predictions for user {}:",
                               substitute_with_current=True):
            user_train = self.train_set[self.train_set['from_id'] == user_id]
            user_seen_items = set(user_train['to_id'])

            filter_list = None
            if filter_dict is not None:
                filter_list = filter_dict.get(user_id)

            user_fitted_alg = self._user_fit_dic.get(user_id)
            if user_fitted_alg is not None:
                pred = user_fitted_alg.predict(user_seen_items, loaded_items_interface,
                                               filter_list=filter_list)
                pred.insert(0, 'from_id', user_id)
            else:
                pred = pd.DataFrame({'from_id': [], 'to_id': [], 'score': []})
                logger.warning(f"No algorithm fitted for user {user_id}! It will be skipped")

            pred_list.append(pred)

        concat_pred = pd.concat(pred_list)
        return concat_pred

    def fit_predict(self, test_set: Union[pd.DataFrame, Iterable],
                    methodology: Methodology = TestRatingsMethodology()):
        """
        The method fits the algorithm and then calculates the prediction for each user

        Args:
            test_set: set of users for which to calculate the prediction

        Returns:
            prediction: prediction for each user

        """
        self.fit()
        prediction = self.predict(test_set, methodology)
        return prediction

    def fit_rank(self, test_set: Union[pd.DataFrame, Iterable], n_recs: int = None,
                 methodology: Methodology = TestRatingsMethodology()):
        """
        The method fits the algorithm and then calculates the rank for each user

        Args:
            test_set: set of users for which to calculate the rank
            n_recs: number of the top items that will be present in the ranking

        Returns:
            rank: ranked items for each user

        """
        self.fit()
        rank = self.rank(test_set, n_recs, methodology)
        return rank

    # @Handler_EmptyFrame
    # def fit_predict(self, user_id: str, filter_list: List[str] = None):
    #     """
    #     Method used to predict the rating of the user specified
    #
    #     If the filter_list parameter is specified, score prediction is executed only for the items
    #     inside the filter list. Otherwise, score prediction is executed for all unrated items of the
    #     user
    #
    #     The method fits the algorithm and then calculates the prediction
    #
    #     Args:
    #         user_id (str): user_id of the user
    #         filter_list (list): list of the items to score predict, if None all unrated items will
    #             be score predicted
    #     Returns:
    #         pd.DataFrame: DataFrame containing score predictions for the user
    #     """
    #     # Extracts ratings of the user
    #     user_ratings = self.rating_frame[self.rating_frame['from_id'] == user_id]
    #
    #     alg = self.algorithm
    #
    #     # Process rated items
    #     alg.process_rated(user_ratings, self.items_directory)
    #
    #     # Fit
    #     alg.fit()
    #
    #     # Predict
    #     prediction = alg.predict(user_ratings, self.items_directory, filter_list)
    #
    #     prediction.insert(0, 'from_id', [user_id for _ in range(len(prediction))])
    #
    #     return prediction
    #
    # @Handler_EmptyFrame
    # def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None):
    #     """
    #     Method used to calculate ranking for the specified user
    #
    #     The method fits the algorithm and then calculates the rank.
    #
    #     If the recs_number is specified, then the rank will contain the top-n items for the user.
    #     Otherwise the rank will contain all unrated items of the user
    #
    #     If the filter_list parameter is specified, score prediction is executed only for the items
    #     inside the filter list. Otherwise, score prediction is executed for all unrated items of the
    #     particular user
    #
    #     Args:
    #         user_id (str): user_id of the user
    #         recs_number (int): number of the top items that will be present in the ranking
    #         filter_list (list): list of the items to rank, if None all unrated items will be ranked
    #     Returns:
    #         pd.DataFrame: DataFrame containing ranking for the user specified
    #     """
    #     # Extracts ratings of the user
    #     user_ratings = self.rating_frame[self.rating_frame['from_id'] == user_id]
    #
    #     alg = self.algorithm
    #
    #     # Process rated items
    #     alg.process_rated(user_ratings, self.items_directory)
    #
    #     # Fit
    #     alg.fit()
    #
    #     # Rank
    #     rank = alg.rank(user_ratings, self.items_directory, recs_number, filter_list)
    #
    #     rank.insert(0, 'from_id', [user_id for _ in range(len(rank))])
    #
    #     return rank
    #
    # def _eval_fit_predict(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
    #     user_id = user_ratings_train.from_id.iloc[0]
    #
    #     rs_eval = ContentBasedRS(self.algorithm, user_ratings_train, self.items_directory, self.users_directory)
    #     score_frame = rs_eval.fit_predict(user_id, filter_list=test_items_list)
    #     return score_frame
    #
    # def _eval_fit_rank(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
    #     user_id = user_ratings_train.from_id.iloc[0]
    #
    #     rs_eval = ContentBasedRS(self.algorithm, user_ratings_train, self.items_directory, self.users_directory)
    #     score_frame = rs_eval.fit_rank(user_id, filter_list=test_items_list)
    #     return score_frame


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
                 graph: FullGraph):

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

    def predict(self, test_set: pd.DataFrame, methodology: Methodology = TestRatingsMethodology()):
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
        train_set = self.graph.convert_to_dataframe(only_values=True)

        all_users = set(test_set)
        filter_frame = None
        if hasattr(test_set, "columns") and 'to_id' in test_set.columns:
            filter_frame = methodology.filter_all(train_set, test_set)
            all_users = set(filter_frame['from_id'].values)

        pred_list = []

        for user_id in progbar(all_users, prefix="Computing rank for user {}:", substitute_with_current=True):

            filter_list = None
            if filter_frame is not None:
                filter_list = list(filter_frame[filter_frame['from_id'] == user_id]['to_id'])

            pred = self.algorithm.predict(user_id, self.graph, filter_list=filter_list)
            pred.insert(0, 'from_id', user_id)

            pred_list.append(pred)

        concat_pred = pd.concat(pred_list)
        return concat_pred

    def rank(self, test_set: pd.DataFrame, n_recs: int = None, methodology: Methodology = TestRatingsMethodology()):
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
        train_set = self.graph.convert_to_dataframe(only_values=True)

        all_users = set(test_set)
        filter_frame = None
        if hasattr(test_set, "columns") and 'to_id' in test_set.columns:
            filter_frame = methodology.filter_all(train_set, test_set)
            all_users = set(filter_frame['from_id'].values)

        rank_list = []

        for user_id in progbar(all_users, prefix="Computing score prediction for user {}:",
                               substitute_with_current=True):
            filter_list = None
            if filter_frame is not None:
                filter_list = list(filter_frame[filter_frame['from_id'] == user_id]['to_id'])

            rank = self.algorithm.rank(user_id, self.graph, n_recs, filter_list=filter_list)
            rank.insert(0, 'from_id', user_id)

            rank_list.append(rank)

        concat_rank = pd.concat(rank_list)
        return concat_rank

    # def fit_predict(self, user_id: str, filter_list: List[str] = None) -> pd.DataFrame:
    #     """
    #     Method used to predict the rating of the user specified
    #
    #     If the filter_list parameter is specified, score prediction is executed only for the items
    #     inside the filter list. Otherwise, score prediction is executed for all unrated items of the
    #     user
    #
    #     Args:
    #         user_id (str): user_id of the user
    #         filter_list (list): list of the items to score predict, if None all unrated items will
    #             be score predicted
    #     Returns:
    #         pd.DataFrame: DataFrame containing score predictions for the user
    #     """
    #     alg = self.algorithm
    #
    #     prediction = alg.predict(user_id, self.graph, filter_list)
    #
    #     prediction.insert(0, 'from_id', [user_id for i in range(len(prediction))])
    #
    #     return prediction
    #
    # def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None) -> pd.DataFrame:
    #     """
    #     Method used to calculate ranking for the specified user
    #
    #     If the recs_number is specified, then the rank will contain the top-n items for the user.
    #     Otherwise the rank will contain all unrated items of the user
    #
    #     If the filter_list parameter is specified, score prediction is executed only for the items
    #     inside the filter list. Otherwise, score prediction is executed for all unrated items of the
    #     particular user
    #
    #     Args:
    #         user_id (str): user_id of the user
    #         recs_number (int): number of the top items that will be present in the ranking
    #         filter_list (list): list of the items to rank, if None all unrated items will be ranked
    #     Returns:
    #         pd.DataFrame: DataFrame containing ranking for the user specified
    #     """
    #     alg = self.algorithm
    #
    #     rank = alg.rank(user_id, self.graph, recs_number, filter_list)
    #
    #     rank.insert(0, 'from_id', [user_id for i in range(len(rank))])
    #
    #     return rank
    #
    # def _eval_fit_predict(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
    #     user_id = user_ratings_train.from_id.iloc[0]
    #
    #     eval_graph: FullGraph = self.graph.copy()
    #
    #     # we remove the interaction between user and items in the training set in order
    #     # to make items unknown to the user
    #     for idx, row in user_ratings_train.iterrows():
    #         eval_graph.remove_link(row['from_id'], row['to_id'])
    #
    #     rs_eval = GraphBasedRS(self.algorithm, eval_graph)
    #     score_frame = rs_eval.fit_predict(user_id, filter_list=test_items_list)
    #
    #     return score_frame
    #
    # def _eval_fit_rank(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
    #     user_id = user_ratings_train.from_id.iloc[0]
    #
    #     eval_graph: FullGraph = self.graph.copy()
    #
    #     # we remove the interaction between user and items in the training set in order
    #     # to make items unknown to the user
    #     for idx, row in user_ratings_train.iterrows():
    #         eval_graph.remove_link(row['from_id'], row['to_id'])
    #
    #     rs_eval = GraphBasedRS(self.algorithm, eval_graph)
    #
    #     score_frame = rs_eval.fit_rank(user_id, filter_list=test_items_list)
    #
    #     return score_frame
