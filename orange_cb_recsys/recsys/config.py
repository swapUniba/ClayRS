from orange_cb_recsys.recsys.algorithm import RankingAlgorithm, ScorePredictionAlgorithm

from orange_cb_recsys.utils.load_ratings import load_ratings

import pandas as pd


class RecSysConfig:
    """
    Configuration for the recommender system
    Args:
        users_directory (str): Path to the directory in which the users are stored
        items_directory (str): Path to the directory in which the items are stored
        score_prediction_algorithm (ScorePredictionAlgorithm): Score prediction algorithm to use
        ranking_algorithm (RankingAlgorithm): Ranking algorithm to use
        rating_frame: Can be the path to the directory in which the ratings .csv is stored, or a DataFrame
            that contains the ratings
    """
    def __init__(self, users_directory: str,
                 items_directory: str,
                 score_prediction_algorithm: ScorePredictionAlgorithm = None,
                 ranking_algorithm: RankingAlgorithm = None,
                 rating_frame=None):
        self.__users_directory: str = users_directory
        self.__items_directory: str = items_directory

        self.__score_prediction_algorithm: ScorePredictionAlgorithm = score_prediction_algorithm
        self.__ranking_algorithm: RankingAlgorithm = ranking_algorithm

        if self.__score_prediction_algorithm is None and self.__ranking_algorithm is None:
            raise ValueError("You must set at least one algorithm")

        if type(rating_frame) is str:
            self.__rating_frame = load_ratings(rating_frame)
        else:
            self.__rating_frame = rating_frame

        self.__rating_frame['score'] = pd.to_numeric(self.__rating_frame["score"], downcast="float")

    @property
    def users_directory(self):
        return self.__users_directory

    @property
    def items_directory(self):
        return self.__items_directory

    @property
    def score_prediction_algorithm(self):
        return self.__score_prediction_algorithm

    @property
    def ranking_algorithm(self):
        return self.__ranking_algorithm

    @property
    def rating_frame(self):
        return self.__rating_frame

    @users_directory.setter
    def users_directory(self, users_directory: str):
        self.__users_directory = users_directory

    @ranking_algorithm.setter
    def ranking_algorithm(self, ranking_algorithm: str):
        self.__ranking_algorithm = ranking_algorithm

    @score_prediction_algorithm.setter
    def score_prediction_algorithm(self, score_prediction_algorithm: str):
        self.__score_prediction_algorithm = score_prediction_algorithm

    @items_directory.setter
    def items_directory(self, items_directory: str):
        self.__items_directory = items_directory

    @rating_frame.setter
    def rating_frame(self, rating_frame: str):
        self.__rating_frame = rating_frame
