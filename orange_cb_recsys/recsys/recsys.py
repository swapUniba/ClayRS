import abc

import pandas as pd
from typing import List
from abc import ABC

from orange_cb_recsys.recsys.graphs.graph import FullGraph

from orange_cb_recsys.recsys.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import Handler_EmptyFrame
from orange_cb_recsys.recsys.graph_based_algorithm import GraphBasedAlgorithm


class RecSys(ABC):

    def __init__(self, rating_frame: pd.DataFrame):
        self.__rating_frame = rating_frame

    @property
    def rating_frame(self):
        return self.__rating_frame

    @abc.abstractmethod
    def fit_predict(self, user_id: str, filter_list: List[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_fit_predict(self, train_ratings: pd.DataFrame, test_set_items: List[str]):
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_fit_rank(self, train_ratings: pd.DataFrame, test_set_items: List[str]):
        raise NotImplementedError


class ContentBasedRS(RecSys):

    def __init__(self,
                 algorithm: ContentBasedAlgorithm,
                 rating_frame: pd.DataFrame,
                 items_directory: str,
                 users_directory: str = None):

        # frame_to_concat = []
        # for user in set(rating_frame['from_id']):
        #     user_frame = rating_frame.query('from_id == @user')
        #     valid_user_frame = remove_not_existent_items(user_frame, items_directory)
        #     frame_to_concat.append(valid_user_frame)
        #
        # valid_rating_frame = pd.concat(frame_to_concat)
        super().__init__(rating_frame)

        self.__algorithm = algorithm
        self.__items_directory = items_directory
        self.__users_directory = users_directory

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def items_directory(self):
        return self.__items_directory

    @property
    def users_directory(self):
        return self.__users_directory

    @Handler_EmptyFrame
    def fit_predict(self, user_id: str, filter_list: List[str] = None):
        """
        Method used to predict the rating of the user passed for all unrated items or for the items passed
        in the filter_list parameter.

        The method fits the algorithm and then calculates the prediction

        Args:
            user_id (str): user_id of the user
            filter_list (list): list of the items to rank, if None all unrated items will be used to
                calculate the rank
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted
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

        prediction.insert(0, 'from_id', [user_id for i in range(len(prediction))])

        return prediction

    @Handler_EmptyFrame
    def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None):
        """
        Method used to rank for a particular user all unrated items or the items specified in
        the filter_list parameter.

        The method fits the algorithm and then calculates the rank.

        If the recs_number is specified, then the rank will contain the top-n items for the user.

        Args:
            user_id (str): user_id of the user
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to rank, if None all unrated items will be used to
                calculate the rank
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted, sorted in descending order by the 'rating' column
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

        rank.insert(0, 'from_id', [user_id for i in range(len(rank))])

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
    def __init__(self,
                 algorithm: GraphBasedAlgorithm,
                 graph: FullGraph):
        self.__algorithm = algorithm
        self.__graph = graph
        super().__init__(rating_frame=graph.convert_to_dataframe())

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def rating_frame(self):
        return self.__graph.convert_to_dataframe()

    @property
    def graph(self):
        return self.__graph

    def fit_predict(self, user_id: str, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method used to predict the rating of the user passed for all unrated items which are present in the graph
        or for the items passed in the filter_list parameter which are also present in the graph.

        The method fits the algorithm and then calculates the prediction, even though in the case of the
        graph-based recommendation the fit process is non-existent

        Args:
            user_id (str): user_id of the user
            filter_list (list): list of the items to rank, if None all unrated items will be used to
                calculate the rank
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted
        """

        alg = self.algorithm

        prediction = alg.predict(user_id, self.graph, filter_list)

        prediction.insert(0, 'from_id', [user_id for i in range(len(prediction))])

        return prediction

    def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method used to rank for a particular user all unrated items which are present in the graph or the items
        specified in the filter_list parameter which are also present in the graph.

        The method fits the algorithm and then calculates the prediction, even though in the case of the
        graph-based recommendation the fit process is non-existent

        If the recs_number is specified, then the rank will contain the top-n items for the user.

        Args:
            user_id (str): user_id of the user
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to rank, if None all unrated items will be used to
                calculate the rank
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted, sorted in descending order by the 'rating' column
        """
        alg = self.algorithm

        rank = alg.rank(user_id, self.graph, recs_number, filter_list)

        rank.insert(0, 'from_id', [user_id for i in range(len(rank))])

        return rank

    def _eval_fit_predict(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
        user_id = user_ratings_train.from_id.iloc[0]

        eval_graph: FullGraph = self.graph.copy()

        for idx, row in user_ratings_train.iterrows():
            eval_graph.remove_link(row['from_id'], row['to_id'])

        rs_eval = GraphBasedRS(self.algorithm, eval_graph)
        score_frame = rs_eval.fit_predict(user_id, filter_list=test_items_list)

        return score_frame

    def _eval_fit_rank(self, user_ratings_train: pd.DataFrame, test_items_list: List[str]):
        user_id = user_ratings_train.from_id.iloc[0]

        eval_graph: FullGraph = self.graph.copy()

        for idx, row in user_ratings_train.iterrows():
            eval_graph.remove_link(row['from_id'], row['to_id'])

        rs_eval = GraphBasedRS(self.algorithm, eval_graph)

        score_frame = rs_eval.fit_rank(user_id, filter_list=test_items_list)

        return score_frame
