from abc import abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.evaluation.metrics.metrics import Metric


class ErrorMetric(Metric):
    """
    Abstract class for error metrics.
    An Error Metric evaluates 'how wrong' the recommender system was in predicting a rating

    Obviously the recommender system must be able to do score prediction in order to be evaluated in one of these
    metrics
    """

    def perform(self, split: Split) -> pd.DataFrame:
        pred = split.pred
        truth = split.truth

        split_result = {'from_id': [], str(self): []}

        for user in set(truth.from_id):
            user_predictions = pred[pred['from_id'] == user]
            user_truth = truth[truth['from_id'] == user]

            zipped_score_list = [(float(pred_score), float(user_truth[user_truth['to_id'] == item_id]['score'].values))
                                 for item_id, pred_score in zip(user_predictions['to_id'], user_predictions['score'])
                                 if not user_truth[user_truth['to_id'] == item_id].empty]

            if len(zipped_score_list) != 0:
                pred_score_list = pd.Series([a_tuple[0] for a_tuple in zipped_score_list])
                truth_score_list = pd.Series([a_tuple[1] for a_tuple in zipped_score_list])
                result = self._calc_metric(pred_score_list, truth_score_list)
            else:
                result = np.nan

            split_result['from_id'].append(user)
            split_result[str(self)].append(result)

        split_result['from_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)

    @abstractmethod
    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        """
        Private method that must be implemented by every error metric specifying how to calculate the metric
        for a single user given a Series of ratings taken from the ground truth and a Series of ratings predicted by the
        recommender system.

        Both Series must be in relation 1 to 1 between each other, meaning that if row 1 of the 'truth_scores' parameter
        contains the rating of the 'iPhone' item, row 1 of the 'pred_scores' parameter must contain the predicted rating
        of the 'iPhone' item, so both Series must be ordered in the same manner and must be filtered in order to exclude
        items which are in the ground truth of the user but are not predicted, as well as items which are predicted
        but aren't present in the ground truth. Debug the unittest cases for examples

        Args:
            truth_scores (pd.Series): Pandas Series which contains rating of the user taken from its ground truth. Has a
                1 to 1 relationship with the 'pred_scores' Series
            pred_scores (pd.Series): Pandas Series which contains rating predicted by the recommender system. Has a
                1 to 1 relationship with the 'truth_scores' Series
        """
        raise NotImplementedError


class MSE(ErrorMetric):
    """
    The MSE (Mean Squared Error) metric is calculated as such for the **single user**:

    .. math:: MSE_u = \sum_{i \in T_u} \\frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u|}

    |
    Where:

    - :math:`T_u` is the *test set* of the user :math:`u`
    - :math:`r_{u, i}` is the actual score give by user :math:`u` to item :math:`i`
    - :math:`\hat{r}_{u, i}` is the predicted score give by user :math:`u` to item :math:`i`

    And it is calculated as such for the **entire system**:

    .. math::
        MSE_sys = \sum_{u \in T} \\frac{MSE_u}{|T|}
    |
    Where:

    - :math:`T` is the *test set*
    - :math:`MSE_u` is the MSE calculated for user :math:`u`

    There may be cases in which some items of the *test set* of the user could not be predicted (eg. A CBRS was chosen
    and items were not present locally, a methodology different than *TestRatings* was chosen).

    In those cases the :math:`MSE_u` formula becomes

    .. math:: MSE_u = \sum_{i \in T_u} \\frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u| - unk}
    |
    Where:

    - **unk** (*unknown*) is the number of items of the *user test set* that could not be predicted

    If no items of the user test set has been predicted (:math:`|T_u| - unk = 0`), then:

    .. math:: MSE_u = NaN
    """

    def __str__(self):
        return "MSE"

    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        return mean_squared_error(truth_scores, pred_scores)


class RMSE(ErrorMetric):
    """
    The RMSE (Root Mean Squared Error) metric is calculated as such for the **single user**:

    .. math:: RMSE_u = \sqrt{\sum_{i \in T_u} \\frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u|}}

    |
    Where:

    - :math:`T_u` is the *test set* of the user :math:`u`
    - :math:`r_{u, i}` is the actual score give by user :math:`u` to item :math:`i`
    - :math:`\hat{r}_{u, i}` is the predicted score give by user :math:`u` to item :math:`i`

    And it is calculated as such for the **entire system**:

    .. math::
        RMSE_sys = \sum_{u \in T} \\frac{RMSE_u}{|T|}
    |
    Where:

    - :math:`T` is the *test set*
    - :math:`RMSE_u` is the RMSE calculated for user :math:`u`

    There may be cases in which some items of the *test set* of the user could not be predicted (eg. A CBRS was chosen
    and items were not present locally, a methodology different than *TestRatings* was chosen).

    In those cases the :math:`RMSE_u` formula becomes

    .. math:: RMSE_u = \sqrt{\sum_{i \in T_u} \\frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u| - unk}}
    |
    Where:

    - **unk** (*unknown*) is the number of items of the *user test set* that could not be predicted

    If no items of the user test set has been predicted (:math:`|T_u| - unk = 0`), then:

    .. math:: RMSE_u = NaN
    """

    def __str__(self):
        return "RMSE"

    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        return mean_squared_error(truth_scores, pred_scores, squared=False)


class MAE(ErrorMetric):
    """
    The MAE (Mean Absolute Error) metric is calculated as such for the **single user**:

    .. math:: MAE_u = \sum_{i \in T_u} \\frac{|r_{u,i} - \hat{r}_{u,i}|}{|T_u|}

    |
    Where:

    - :math:`T_u` is the *test set* of the user :math:`u`
    - :math:`r_{u, i}` is the actual score give by user :math:`u` to item :math:`i`
    - :math:`\hat{r}_{u, i}` is the predicted score give by user :math:`u` to item :math:`i`

    And it is calculated as such for the **entire system**:

    .. math::
        MAE_sys = \sum_{u \in T} \\frac{MAE_u}{|T|}
    |
    Where:

    - :math:`T` is the *test set*
    - :math:`MAE_u` is the MAE calculated for user :math:`u`

    There may be cases in which some items of the *test set* of the user could not be predicted (eg. A CBRS was chosen
    and items were not present locally, a methodology different than *TestRatings* was chosen).

    In those cases the :math:`MAE_u` formula becomes

    .. math:: MAE_u = \sum_{i \in T_u} \\frac{|r_{u,i} - \hat{r}_{u,i}|}{|T_u| - unk}
    |
    Where:

    - **unk** (*unknown*) is the number of items of the *user test set* that could not be predicted

    If no items of the user test set has been predicted (:math:`|T_u| - unk = 0`), then:

    .. math:: MAE_u = NaN
    """

    def __str__(self):
        return "MAE"

    def _calc_metric(self, truth_scores: pd.Series, pred_scores: pd.Series):
        return mean_absolute_error(truth_scores, pred_scores)
