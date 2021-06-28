from abc import abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.metrics import ScoresNeededMetric


class ErrorMetric(ScoresNeededMetric):

    def perform(self, split: Split) -> pd.DataFrame:
        """
        Method that execute the prediction metric computation

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating
        """
        pred = split.pred
        truth = split.truth

        split_result = {'from_id': [], str(self): []}

        for user in set(truth.from_id):
            user_predictions = pred.loc[split.pred['from_id'] == user]
            user_truth = truth.loc[split.truth['from_id'] == user]

            user_predictions = user_predictions[['to_id', 'score']]
            user_truth = user_truth[['to_id', 'score']]

            valid = user_predictions.merge(user_truth, on='to_id',
                                           suffixes=('_pred', '_truth'))

            if not valid.empty:
                result = self._calc_metric(valid['score_pred'], valid['score_truth'])
            else:
                result = np.nan

            split_result['from_id'].append(user)
            split_result[str(self)].append(result)

        split_result['from_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)

    @abstractmethod
    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        raise NotImplementedError


class MSE(ErrorMetric):
    """
    RMSE
    .. image:: metrics_img/rmse.png
    \n\n
    Where T is the test set and r' is the actual score give by user u to item i
    """

    def __str__(self):
        return "MSE"

    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        return mean_squared_error(truth_scores, pred_scores)


class RMSE(ErrorMetric):
    """
    RMSE
    .. image:: metrics_img/rmse.png
    \n\n
    Where T is the test set and r' is the actual score give by user u to item i
    """

    def __str__(self):
        return "RMSE"

    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        return mean_squared_error(truth_scores, pred_scores, squared=False)


class MAE(ErrorMetric):
    """
    MAE
    .. image:: metrics_img/mae.png
    """

    def __str__(self):
        return "MAE"

    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        return mean_absolute_error(truth_scores, pred_scores)
