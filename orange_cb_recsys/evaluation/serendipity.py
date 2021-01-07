import pandas as pd

from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.evaluation.utils import popular_items


class Serendipity(Metric):
    """
    Serendipity

    .. image:: metrics_img/serendipity.png


    Args:
        num_of_recs: number of recommendation
            produced for each user
    """
    def __init__(self, num_of_recs: int):
        self.__num_of_recs = num_of_recs

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Calculates the serendipity score: unexpected recommendations, surprisingly and interesting items a user
        might not have otherwise discovered
        
        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            serendipity (float): The serendipity value
        """

        most_popular_items = popular_items(score_frame=truth)
        users = set(predictions[['from_id']].values.flatten())

        pop_ratios_sum = 0
        for user in users:
            recommended_items = predictions.query('from_id == @user')[['to_id']].values.flatten()
            pop_items_count = 0
            for item in recommended_items:
                if item not in most_popular_items:
                    pop_items_count += 1

            pop_ratios_sum += pop_items_count / self.__num_of_recs

        serendipity = pop_ratios_sum / len(users)

        return serendipity
