import re

import pandas as pd
from typing import List


from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import load_content_instance


class RecSys:
    """
    Class that represent a recommender system
    Args:
        config (RecSysConfig): Configuration of the recommender system
    """

    def __init__(self, config: RecSysConfig):
        self.__config: RecSysConfig = config

    def fit_predict(self, user_id: str, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method which predicts the ratings of a user for the unrated items.

        One can specify which items must be predicted with the filter_list parameter,
        in this case ONLY items in the filter_list will be predicted.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be predicted.

        Args:
            user_id (str): user for which the predictions will be calculated
            filter_list (list): list of items that will be predicted. If None,
                all items will be predicted
        """

        # calculate predictions
        logger.info("Computing predictions")
        score_frame = self.__config.algorithm.fit_predict(user_id, filter_list)

        return score_frame

    def fit_ranking(self, user_id: str, recs_number: int = None, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method which predicts the ratings of a user for the unrated items and ranks them.

        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All items will be ranked.

        One can specify which items must be ranked with the filter_list parameter,
        in this case ONLY items in the filter_list will be used to calculate the rank.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be used to calculate the rank.

        Args:
            user_id (str): user for which the rank will be calculated
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of items that will be ranked. If None,
                all items will be ranked
        """
        logger.info("Computing ranking")
        score_frame = self.__config.algorithm.fit_rank(user_id, recs_number, filter_list)

        return score_frame

    def fit_eval_predict(self, user_id, user_ratings: pd.DataFrame, test_set: pd.DataFrame):
        """
        Computes predicted ratings, or ranking (according to algorithm chosen in the config)
        user ratings will be used as train set to fit the algorithm.
        If the algorithm is score_prediction the rating for the item in the test set will
        be predicted

        Args:
            user_id: user for which predictions will be computed
            user_ratings: train set
            test_set:
        Returns:
            score_frame (DataFrame): result frame whose columns are: to_id, rating
        """
        logger.info("Loading items")
        item_to_predict_id_list = [item for item in test_set.to_id]  # unrated items list
        items = [load_content_instance(self.__config.items_directory, re.sub(r'[^\w\s]', '', item_id))
                 for item_id in item_to_predict_id_list]

        logger.info("Loaded %d items" % len(items))

        # calculate predictions
        logger.info("Computing predictions")
        score_frame = self.__config.score_prediction_algorithm.predict(user_id, items, user_ratings,
                                                                       self.__config.items_directory)

        return score_frame

    def fit_eval_ranking(self, user_id, user_ratings: pd.DataFrame, test_set_items, recs_number):
        """
        Computes a ranking of specified length,
        using as training set the ratings provided by the user

        Args:
            user_id:
            user_ratings (pd.DataFrame): Training set
            test_set_items (pd.DataFrame)
            recs_number (int): Number of recommendations to provide
        """
        user_ratings = user_ratings.sort_values(['to_id'], ascending=True)
        score_frame = self.__config.ranking_algorithm.predict(user_id, user_ratings, recs_number,
                                                              self.__config.items_directory,
                                                              test_set_items)
        return score_frame
