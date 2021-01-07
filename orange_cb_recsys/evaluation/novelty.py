import math
from collections import Counter

import pandas as pd

from orange_cb_recsys.evaluation.metrics import Metric


class Novelty(Metric):
    """
    Novelty

    .. image:: metrics_img/novelty.png
    \n\n
    where:
    - hits is a set of predicted items
    - Popularity(i) = % users who rated item i


    Args:
        num_of_recs: number of recommendation
            produced for each user
    """
    def __init__(self, num_of_recs):
        self.__num_of_recs = num_of_recs

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Calculates the novelty score

        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            novelty (float): Novelty score
        """
        total_ratings = len(truth.index)
        ratings_by_item = Counter(truth[['to_id']].values.flatten())
        users = set(predictions[['from_id']].values.flatten())

        users_log_popularity = 0
        for user in users:
            user_recs = set(predictions.query('from_id == @user')[['to_id']].values.flatten())
            user_log_popularity = 0
            for item in user_recs:
                item_pop = (ratings_by_item[item] + 1) / total_ratings
                user_log_popularity += math.log2(item_pop)
            users_log_popularity += user_log_popularity

        novelty = - (users_log_popularity / (len(users) * self.__num_of_recs))

        return novelty
