import abc



import pandas as pd
from typing import List
from abc import ABC

#from orange_cb_recsys.evaluation.exceptions import PartitionError
from orange_cb_recsys.recsys.graphs.graph import FullGraph

from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import Handler_EmptyFrame
from orange_cb_recsys.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm

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

    def __init__(self, rating_frame: pd.DataFrame):
        self.__rating_frame = rating_frame

    @property
    def rating_frame(self):
        """
        The DataFrame containing interactions between users and items
        """
        return self.__rating_frame

    @property
    @abc.abstractmethod
    def users(self):
        """
        Users of the recommender systems (users that have at least one interaction in the rating_frame)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_predict(self, user_id: str, filter_list: List[str] = None):
        """
        Method to call when score prediction must be done for the specified user_id

        By default, score prediction will be done for all unrated items by the user, unless the
        filter_list parameter is specified.

        If the filter_list parameter is specified, then score_prediction is executed only for the
        items inside the filter_list
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None):
        """
        Method to call when ranking must be calculated for the specified user_id

        If the recs_number parameter is specified, only the top-n recommendation will be returned.
        Otherwise ranking of all the unrated items by the user will be returned

        If the filter_list parameter is specified, then the ranking is calculated only for the
        items inside the filter list
        """
        raise NotImplementedError

    def multiple_fit_rank(self, user_id_list: List[str], recs_number: int = None, filter_list: List[str] = None):
        """
        Method used to calculate ranking for a list of users

        The method fits the algorithm (when eligible) and then calculates the rank.

        If the recs_number is specified, then the rank will contain the top-n items for the particular user.
        Otherwise the rank will contain all unrated items of the particular user

        If the filter_list parameter is specified, score prediction is executed only for the items
        inside the filter list. Otherwise, score prediction is executed for all unrated items of the
        particular user

        Args:
            user_id_list (List[str]): list of all the users ids of which ranking must be calculated
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to rank, if None all unrated items of the particular user
                will be ranked
        Returns:
            pd.DataFrame: DataFrame containing ranking for all users specified
        """
        rank_list = []
        for user_id in user_id_list:
            rank = self.fit_rank(user_id, recs_number, filter_list)

            rank_list.append(rank)

        concat_rank = pd.concat(rank_list)
        return concat_rank

    def multiple_fit_predict(self, user_id_list: List[str], filter_list: List[str] = None):
        """
        Method used to calculate score prediction for a list of users

        The method fits the algorithm (when eligible) and then calculates the prediction

        If the filter_list parameter is specified, score prediction is executed only for the items
        inside the filter list. Otherwise, score prediction is executed for all unrated items of the
        particular user

        Args:
            user_id_list (List[str]): list of all the users ids of which score prediction must be calculated
            filter_list (list): list of the items to score predict, if None all unrated items of the particular user
                will be score predicted
        Returns:
            pd.DataFrame: DataFrame containing score predictions for all users specified
        """
        score_preds_list = []
        for user_id in user_id_list:
            score_preds = self.fit_predict(user_id, filter_list)

            score_preds_list.append(score_preds)

        concat_score_preds = pd.concat(score_preds_list)
        return concat_score_preds

    @abc.abstractmethod
    def _eval_fit_predict(self, train_ratings: pd.DataFrame, test_set_items: List[str]):
        """
        Private method used by the evaluation module

        It calculates score prediction on the test set items based on the train ratings items
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_fit_rank(self, train_ratings: pd.DataFrame, test_set_items: List[str]):
        """
        Private method used by the evaluation module

        It calculates ranking on the test set items based on the train ratings items
        """
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
                 train_set: pd.DataFrame,
                 items_directory: str,
                 users_directory: str = None):

        # frame_to_concat = []
        # for user in set(rating_frame['from_id']):
        #     user_frame = rating_frame.query('from_id == @user')
        #     valid_user_frame = remove_not_existent_items(user_frame, items_directory)
        #     frame_to_concat.append(valid_user_frame)
        #
        # valid_rating_frame = pd.concat(frame_to_concat)
        super().__init__(train_set)

        self.__algorithm = algorithm
        self.__items_directory = items_directory
        self.__users_directory = users_directory

    @property
    def algorithm(self):
        """
        The content based algorithm chosen
        """
        return self.__algorithm

    @property
    def users(self):
        return set(self.rating_frame['from_id'])

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

    @Handler_EmptyFrame
    def fit_predict(self, user_id: str, filter_list: List[str] = None):
        """
        Method used to predict the rating of the user specified

        If the filter_list parameter is specified, score prediction is executed only for the items
        inside the filter list. Otherwise, score prediction is executed for all unrated items of the
        user

        The method fits the algorithm and then calculates the prediction

        Args:
            user_id (str): user_id of the user
            filter_list (list): list of the items to score predict, if None all unrated items will
                be score predicted
        Returns:
            pd.DataFrame: DataFrame containing score predictions for the user
        """
        # Extracts ratings of the user
        user_ratings = self.rating_frame[self.rating_frame['from_id'] == user_id]

        alg = self.algorithm

        # Process rated items
        alg.process_rated(user_ratings, self.items_directory)

        # Fit
        alg.fit()

        # Predict
        prediction = alg.predict(user_ratings, self.items_directory, filter_list)

        prediction.insert(0, 'from_id', [user_id for _ in range(len(prediction))])

        return prediction

    @Handler_EmptyFrame
    def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None):
        """
        Method used to calculate ranking for the specified user

        The method fits the algorithm and then calculates the rank.

        If the recs_number is specified, then the rank will contain the top-n items for the user.
        Otherwise the rank will contain all unrated items of the user

        If the filter_list parameter is specified, score prediction is executed only for the items
        inside the filter list. Otherwise, score prediction is executed for all unrated items of the
        particular user

        Args:
            user_id (str): user_id of the user
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to rank, if None all unrated items will be ranked
        Returns:
            pd.DataFrame: DataFrame containing ranking for the user specified
        """
        # Extracts ratings of the user
        user_ratings = self.rating_frame[self.rating_frame['from_id'] == user_id]

        alg = self.algorithm

        # Process rated items
        alg.process_rated(user_ratings, self.items_directory)

        # Fit
        alg.fit()

        # Rank
        rank = alg.rank(user_ratings, self.items_directory, recs_number, filter_list)

        rank.insert(0, 'from_id', [user_id for _ in range(len(rank))])

        return rank

    def _eval_fit_predict(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
        user_id = user_ratings_train.from_id.iloc[0]

        rs_eval = ContentBasedRS(self.algorithm, user_ratings_train, self.items_directory, self.users_directory)
        score_frame = rs_eval.fit_predict(user_id, filter_list=test_items_list)
        return score_frame

    def _eval_fit_rank(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
        user_id = user_ratings_train.from_id.iloc[0]

        rs_eval = ContentBasedRS(self.algorithm, user_ratings_train, self.items_directory, self.users_directory)
        score_frame = rs_eval.fit_rank(user_id, filter_list=test_items_list)
        return score_frame


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
        self.__algorithm = algorithm
        self.__graph = graph
        super().__init__(rating_frame=graph.convert_to_dataframe())

    @property
    def algorithm(self):
        """
        The content based algorithm chosen
        """
        return self.__algorithm

    @property
    def rating_frame(self):
        return self.__graph.convert_to_dataframe()

    @property
    def users(self):
        return self.__graph.user_nodes

    @property
    def graph(self):
        """
        The graph containing interactions
        """
        return self.__graph

    def fit_predict(self, user_id: str, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method used to predict the rating of the user specified

        If the filter_list parameter is specified, score prediction is executed only for the items
        inside the filter list. Otherwise, score prediction is executed for all unrated items of the
        user

        Args:
            user_id (str): user_id of the user
            filter_list (list): list of the items to score predict, if None all unrated items will
                be score predicted
        Returns:
            pd.DataFrame: DataFrame containing score predictions for the user
        """
        alg = self.algorithm

        prediction = alg.predict(user_id, self.graph, filter_list)

        prediction.insert(0, 'from_id', [user_id for i in range(len(prediction))])

        return prediction

    def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method used to calculate ranking for the specified user

        If the recs_number is specified, then the rank will contain the top-n items for the user.
        Otherwise the rank will contain all unrated items of the user

        If the filter_list parameter is specified, score prediction is executed only for the items
        inside the filter list. Otherwise, score prediction is executed for all unrated items of the
        particular user

        Args:
            user_id (str): user_id of the user
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to rank, if None all unrated items will be ranked
        Returns:
            pd.DataFrame: DataFrame containing ranking for the user specified
        """
        alg = self.algorithm

        rank = alg.rank(user_id, self.graph, recs_number, filter_list)

        rank.insert(0, 'from_id', [user_id for i in range(len(rank))])

        return rank

    def _eval_fit_predict(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
        user_id = user_ratings_train.from_id.iloc[0]

        eval_graph: FullGraph = self.graph.copy()

        # we remove the interaction between user and items in the training set in order
        # to make items unknown to the user
        for idx, row in user_ratings_train.iterrows():
            eval_graph.remove_link(row['from_id'], row['to_id'])

        rs_eval = GraphBasedRS(self.algorithm, eval_graph)
        score_frame = rs_eval.fit_predict(user_id, filter_list=test_items_list)

        return score_frame

    def _eval_fit_rank(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
        user_id = user_ratings_train.from_id.iloc[0]

        eval_graph: FullGraph = self.graph.copy()

        # we remove the interaction between user and items in the training set in order
        # to make items unknown to the user
        for idx, row in user_ratings_train.iterrows():
            eval_graph.remove_link(row['from_id'], row['to_id'])

        rs_eval = GraphBasedRS(self.algorithm, eval_graph)

        score_frame = rs_eval.fit_rank(user_id, filter_list=test_items_list)

        return score_frame

