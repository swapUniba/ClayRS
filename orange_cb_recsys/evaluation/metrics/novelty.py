from collections import Counter

import pandas as pd
import numpy as np

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric


class Novelty(RankingNeededMetric):
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
    def __init__(self, top_n: int = None):
        self.__top_n = top_n

    def __str__(self):
        return "Novelty"

    def perform(self, split: Split):
        raise NotImplementedError("Novelty not yet implemented!")

    # TO TEST
    def OLD_perform(self, split: Split) -> pd.DataFrame:
        """
        Calculates the novelty score

        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            novelty (float): Novelty score
        """
        pred = split.pred
        truth = split.truth

        if self.__top_n:
            pred = pred.groupby('from_id').head(self.__top_n)

        pop = Counter(list(truth['to_id']))
        user_list = set(pred['from_id'])

        split_result = {'from_id': [], str(self): []}

        for user in set(pred.from_id):
            user_ranking_pred = list(pred.query('from_id == @user')['to_id'])

            self_information = 0
            for item in user_ranking_pred:
                if pop[item] != 0:
                    self_information += np.sum(-np.log2(pop[item] / len(user_list)))

            split_result['from_id'].append(user)
            split_result[str(self)].append(self_information / len(user_ranking_pred))

        novelty_system = sum(split_result[str(self)]) / len(user_list)

        split_result['from_id'].append('sys')
        split_result[str(self)].append(novelty_system)

        return pd.DataFrame(split_result)
