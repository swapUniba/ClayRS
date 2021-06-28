import pandas as pd

from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.utils import popular_items


class Serendipity(RankingNeededMetric):
    """
    Serendipity

    .. image:: metrics_img/serendipity.png


    Args:
        num_of_recs: number of recommendation
            produced for each user
    """
    def __init__(self, top_n: int):
        self.__top_n = top_n

    def perform(self, splt: Split) -> pd.DataFrame:
        raise NotImplementedError("Serendipity not yet implemented!")

    # TO TEST
    def OLD_perform(self, split: Split) -> float:
        """
        Calculates the serendipity score: unexpected recommendations, surprisingly and interesting items a user
        might not have otherwise discovered
        
        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            serendipity (float): The serendipity value
        """
        pred = split.pred
        truth = split.truth

        most_popular_items = popular_items(score_frame=truth)
        users = set(pred['from_id'])

        pop_ratios_sum = 0
        for user in users:
            recommended_items = list(pred.query('from_id == @user')['to_id'])
            pop_items_count = 0
            for item in recommended_items:
                if item not in most_popular_items:
                    pop_items_count += 1

            pop_ratios_sum += pop_items_count

        serendipity = pop_ratios_sum / len(users)

        return serendipity
