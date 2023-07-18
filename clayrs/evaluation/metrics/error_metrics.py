from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np
import numpy_indexed as npi
from sklearn.metrics import mean_absolute_error, mean_squared_error

if TYPE_CHECKING:
    from clayrs.recsys.partitioning import Split

from clayrs.evaluation.metrics.metrics import Metric, handler_different_users


class ErrorMetric(Metric):
    """
    Abstract class for error metrics.
    An Error Metric evaluates 'how wrong' the recommender system was in predicting a rating
    """

    @handler_different_users
    def perform(self, split: Split) -> pd.DataFrame:
        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}

        # users in truth and pred of the split to evaluate must be the same!
        user_idx_truth = truth.unique_user_idx_column
        user_idx_pred = pred.user_map.convert_seq_str2int(truth.unique_user_id_column)

        for uidx_pred, uidx_truth in zip(user_idx_pred, user_idx_truth):
            user_predictions_idxs = pred.get_user_interactions(uidx_pred, as_indices=True)
            user_truth_idxs = truth.get_user_interactions(uidx_truth, as_indices=True)

            user_truth_items = truth.item_id_column[user_truth_idxs]

            user_prediction_items = pred.item_id_column[user_predictions_idxs]
            user_prediction_scores = pred.score_column[user_predictions_idxs]

            idx_truth_in_pred = npi.indices(user_prediction_items, user_truth_items, missing=-1)
            idx_truth_not_in_pred = np.where(idx_truth_in_pred == -1)

            user_pred_common_idxs = np.delete(idx_truth_in_pred, idx_truth_not_in_pred)
            user_truth_common_idxs = np.delete(user_truth_idxs, idx_truth_not_in_pred)

            common_truth_scores = truth.score_column[user_truth_common_idxs]
            common_prediction_scores = user_prediction_scores[user_pred_common_idxs]

            if len(common_prediction_scores) != 0:
                result = self._calc_metric(common_truth_scores, common_prediction_scores)
            else:
                result = np.nan

            split_result['user_id'].append(uidx_truth)
            split_result[str(self)].append(result)

        split_result['user_id'] = list(truth.user_map.convert_seq_int2str(split_result['user_id']))

        split_result['user_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)

    @abstractmethod
    def _calc_metric(self, truth_scores: list, pred_scores: list):
        """
        Private method that must be implemented by every error metric specifying how to calculate the metric
        for a single user given a list of ratings taken from the ground truth and a list of ratings predicted by the
        recommender system.

        Both lists must be in relation 1 to 1 between each other, meaning that if row 1 of the 'truth_scores' parameter
        contains the rating of the 'iPhone' item, row 1 of the 'pred_scores' parameter must contain the predicted rating
        of the 'iPhone' item, so both lists must be ordered in the same manner and must be filtered in order to exclude
        items which are in the ground truth of the user but are not predicted, as well as items which are predicted
        but aren't present in the ground truth. Debug the unittest cases for examples

        Args:
            truth_scores: list which contains rating of the user taken from its ground truth. Has a
                1 to 1 relationship with the 'pred_scores' Series
            pred_scores: list which contains rating predicted by the recommender system. Has a
                1 to 1 relationship with the 'truth_scores' Series
        """
        raise NotImplementedError


class MSE(ErrorMetric):
    r"""
    The MSE (Mean Squared Error) metric is calculated as such for the **single user**:

    $$
    MSE_u = \sum_{i \in T_u} \frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u|}
    $$

    Where:

    - $T_u$ is the *test set* of the user $u$
    - $r_{u, i}$ is the actual score give by user $u$ to item $i$
    - $\hat{r}_{u, i}$ is the predicted score give by user $u$ to item $i$

    And it is calculated as such for the **entire system**:

    $$
    MSE_{sys} = \sum_{u \in T} \frac{MSE_u}{|T|}
    $$
    Where:

    - $T$ is the *test set*
    - $MSE_u$ is the MSE calculated for user $u$

    There may be cases in which some items of the *test set* of the user could not be predicted (eg. A CBRS was chosen
    and items were not present locally)

    In those cases the $MSE_u$ formula becomes

    $$
    MSE_u = \sum_{i \in T_u} \frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u| - unk}
    $$

    Where:

    - $unk$ (*unknown*) is the number of items of the *user test set* that could not be predicted

    If no items of the user test set has been predicted ($|T_u| - unk = 0$), then:

    $$
    MSE_u = NaN
    $$
    """

    def __str__(self):
        return "MSE"

    def __repr__(self):
        return "MSE()"

    def _calc_metric(self, truth_scores: list, pred_scores: list):
        return mean_squared_error(truth_scores, pred_scores)


class RMSE(ErrorMetric):
    r"""
    The RMSE (Root Mean Squared Error) metric is calculated as such for the **single user**:

    $$
    RMSE_u = \sqrt{\sum_{i \in T_u} \frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u|}}
    $$

    Where:

    - $T_u$ is the *test set* of the user $u$
    - $r_{u, i}$ is the actual score give by user $u$ to item $i$
    - $\hat{r}_{u, i}$ is the predicted score give by user $u$ to item $i$

    And it is calculated as such for the **entire system**:

    $$
    RMSE_{sys} = \sum_{u \in T} \frac{RMSE_u}{|T|}
    $$

    Where:

    - $T$ is the *test set*
    - $RMSE_u$ is the RMSE calculated for user $u$

    There may be cases in which some items of the *test set* of the user could not be predicted (eg. A CBRS was chosen
    and items were not present locally, a methodology different than *TestRatings* was chosen).

    In those cases the $RMSE_u$ formula becomes

    $$
    RMSE_u = \sqrt{\sum_{i \in T_u} \frac{(r_{u,i} - \hat{r}_{u,i})^2}{|T_u| - unk}}
    $$

    Where:

    - $unk$ (*unknown*) is the number of items of the *user test set* that could not be predicted

    If no items of the user test set has been predicted ($|T_u| - unk = 0$), then:

    $$
    RMSE_u = NaN
    $$
    """

    def __str__(self):
        return "RMSE"

    def __repr__(self):
        return f"RMSE()"

    def _calc_metric(self, truth_scores: list, pred_scores: list):
        return mean_squared_error(truth_scores, pred_scores, squared=False)


class MAE(ErrorMetric):
    r"""
    The MAE (Mean Absolute Error) metric is calculated as such for the **single user**:

    $$
    MAE_u = \sum_{i \in T_u} \frac{|r_{u,i} - \hat{r}_{u,i}|}{|T_u|}
    $$

    Where:

    - $T_u$ is the *test set* of the user $u$
    - $r_{u, i}$ is the actual score give by user $u$ to item $i$
    - $\hat{r}_{u, i}$ is the predicted score give by user $u$ to item $i$

    And it is calculated as such for the **entire system**:

    $$
    MAE_{sys} = \sum_{u \in T} \frac{MAE_u}{|T|}
    $$

    Where:

    - $T$ is the *test set*
    - $MAE_u$ is the MAE calculated for user $u$

    There may be cases in which some items of the *test set* of the user could not be predicted (eg. A CBRS was chosen
    and items were not present locally, a methodology different than *TestRatings* was chosen).

    In those cases the $MAE_u$ formula becomes

    $$
    MAE_u = \sum_{i \in T_u} \frac{|r_{u,i} - \hat{r}_{u,i}|}{|T_u| - unk}
    $$

    Where:

    - $unk$ (*unknown*) is the number of items of the *user test set* that could not be predicted

    If no items of the user test set has been predicted ($|T_u| - unk = 0$), then:

    $$
    MAE_u = NaN
    $$
    """

    def __str__(self):
        return "MAE"

    def __repr__(self):
        return f"MAE()"

    def _calc_metric(self, truth_scores: list, pred_scores: list):
        return mean_absolute_error(truth_scores, pred_scores)
