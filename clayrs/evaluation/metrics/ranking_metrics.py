from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import pandas as pd
import numpy as np
import numpy_indexed as npi

if TYPE_CHECKING:
    from clayrs.recsys.partitioning import Split

from clayrs.evaluation.metrics.metrics import Metric, handler_different_users


class RankingMetric(Metric):
    """
    Abstract class that generalize ranking metrics.
    A ranking metric evaluates the quality of the recommendation list
    """
    pass


class NDCG(RankingMetric):
    r"""
    The NDCG (Normalized Discounted Cumulative Gain) metric is calculated for the **single user** by first computing
    the DCG score using the following formula:

    $$
    DCG_{u}(scores_{u}) = \sum_{r\in scores_{u}}{\frac{f(r)}{log_x(2 + i)}}
    $$

    Where:

    - $scores_{u}$ are the ground truth scores for predicted items, ordered according to the order of said items in the
        ranking for the user $u$
    - $f$ is a gain function (linear or exponential, in particular)
    - $x$ is the base of the logarithm
    - $i$ is the index of the truth score $r$ in the list of scores $scores_{u}$

    If $f$ is "linear", then the truth score $r$ is returned as is. Otherwise, in the "exponential" case, the following
    formula is applied to $r$:

    $$
    f(r) = 2^{r} - 1
    $$

    The NDCG for a single user is then calculated using the following formula:

    $$
    NDCG_u(scores_{u}) = \frac{DCG_{u}(scores_{u})}{IDCG_{u}(scores_{u})}
    $$

    Where:

    - $IDCG_{u}$ is the DCG of the ideal ranking for the truth scores

    So the basic idea is to compare the actual ranking with the ideal one

    Finally, the NDCG of the **entire system** is calculated instead as such:

    $$
    NDCG_{sys} = \frac{\sum_{u} NDCG_u}{|U|}
    $$

    Where:

    - $NDCG_u$ is the NDCG calculated for user :math:`u`
    - $U$ is the set of all users

    The system average excludes NaN values.

    Arguments:
        gains: type of gain function to use when calculating the DCG score, the possible options are "linear" or
            "exponential"
        discount_log: logarithm function to use when calculating the DCG score, by default numpy logarithm in base 2
            is used
    """
    def __init__(self, gains: str = "linear", discount_log: Callable = np.log2):
        self.gains = gains
        self.discount_log = discount_log

        if self.gains == "exponential":
            self.gains_fn = lambda r: 2 ** r - 1
        elif self.gains == "linear":
            self.gains_fn = lambda r: r
        else:
            raise ValueError("Invalid gains option.")

    def __str__(self):
        return "NDCG"

    def __repr__(self):
        return "NDCG()"

    def _dcg_score(self, r: np.ndarray):
        """Discounted cumulative gain (DCG)
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        Returns
        -------
        DCG : float
        """
        dcg = np.nan
        if len(r) != 0:

            gains = self.gains_fn(r)
            discounts = self.discount_log(np.arange(2, len(r) + 2))

            dcg = np.sum(gains / discounts)

        return dcg

    def _calc_ndcg(self, r: np.ndarray):
        """Normalized discounted cumulative gain (NDCG)
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        Returns
        -------
        NDCG : float
        """
        actual = self._dcg_score(r)
        ideal = self._dcg_score(np.sort(r)[::-1])
        return actual / ideal

    @handler_different_users
    def perform(self, split: Split):

        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}

        # users in truth and pred of the split to evaluate must be the same!
        user_idx_truth = truth.unique_user_idx_column
        user_idx_pred = pred.user_map.convert_seq_str2int(truth.unique_user_id_column)

        for uidx_pred, uidx_truth in zip(user_idx_pred, user_idx_truth):
            user_prediction_idxs = pred.get_user_interactions(uidx_pred, as_indices=True)
            user_truth_idxs = truth.get_user_interactions(uidx_truth, as_indices=True)

            user_prediction_items = pred.item_id_column[user_prediction_idxs]
            user_truth_items = truth.item_id_column[user_truth_idxs]

            idx_pred_in_truth = npi.indices(user_truth_items, user_prediction_items, missing=-1)
            common_idx_pred_in_truth = idx_pred_in_truth[idx_pred_in_truth != -1]
            # scores in truth of the items for which there is a prediction ordered according to predictions
            common_truth_scores = truth.score_column[user_truth_idxs][common_idx_pred_in_truth]

            user_ndcg = self._calc_ndcg(common_truth_scores)

            split_result['user_id'].append(uidx_truth)
            split_result[str(self)].append(user_ndcg)

        split_result['user_id'] = list(truth.user_map.convert_seq_int2str(split_result['user_id']))

        split_result['user_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)


class NDCGAtK(NDCG):
    r"""
    The NDCG@K (Normalized Discounted Cumulative Gain at K) metric is calculated for the **single user** by using the
    [framework implementation of the NDCG][clayrs.evaluation.NDCG] but considering $scores_{u}$ cut at the first $k$
    predictions

    Args:
        k (int): the cutoff parameter
        gains: type of gain function to use when calculating the DCG score, the possible options are "linear" or
            "exponential"
        discount_log: logarithm function to use when calculating the DCG score, by default numpy logarithm in base 2
            is used
    """

    def __init__(self, k: int, gains: str = "linear", discount_log: Callable = np.log2):
        super().__init__(gains, discount_log)

        self._k = k

    def __str__(self):
        return "NDCG@{}".format(self._k)

    def __repr__(self):
        return f'NDCGAtK(k={self._k})'

    def _calc_ndcg(self, r: np.ndarray):
        """Normalized discounted cumulative gain (NDCG) at rank k
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        Returns
        -------
        NDCG @k : float
        """
        ideal_r = np.sort(r)[::-1][:self._k]
        actual_r = r[:self._k]

        actual = self._dcg_score(actual_r)
        ideal = self._dcg_score(ideal_r)
        return actual / ideal


class MRR(RankingMetric):
    r"""
    The MRR (Mean Reciprocal Rank) metric is a system wide metric, so only its result it will be returned and not those
    of every user.
    MRR is calculated as such:

    $$
    MRR_{sys} = \frac{1}{|Q|}\cdot\sum_{i=1}^{|Q|}\frac{1}{rank(i)}
    $$

    Where:

    - $Q$ is the set of recommendation lists
    - $rank(i)$ is the position of the first relevant item in the i-th recommendation list

    The MRR metric needs to discern relevant items from the not relevant ones: in order to do that, one could pass a
    custom `relevant_threshold` parameter that will be applied to every user, so that if a rating of an item
    is >= relevant_threshold, then it's relevant, otherwise it's not.
    If no `relevant_threshold` parameter is passed then, for every user, its mean rating score will be used

    Args:
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
    """

    def __init__(self, relevant_threshold: float = None):
        self.__relevant_threshold = relevant_threshold

    @property
    def relevant_threshold(self):
        return self.__relevant_threshold

    def __str__(self):
        return "MRR"

    def __repr__(self):
        return f'MRR(relevant_threshold={self.relevant_threshold})'

    def calc_reciprocal_rank(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        """
        Method which calculates the RR (Reciprocal Rank) for a single user

        Args:
            user_predictions_items: list of ranked item ids for the user computed by the Recommender
            user_truth_relevant_items: list of relevant item ids for the user in its truth set
        """

        common_idxs = npi.indices(user_truth_relevant_items, user_predictions_items, missing=-1)
        non_missing_idxs = np.where(common_idxs != -1)[0]

        reciprocal_rank = 0
        if len(non_missing_idxs) != 0:
            reciprocal_rank = 1 / (non_missing_idxs[0] + 1)  # [0][0] because where returns a tuple

        return reciprocal_rank

    @handler_different_users
    def perform(self, split: Split) -> pd.DataFrame:
        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}

        # users in truth and pred of the split to evaluate must be the same!
        user_idx_truth = truth.unique_user_idx_column
        user_idx_pred = pred.user_map.convert_seq_str2int(truth.unique_user_id_column)

        rr_list = []
        for uidx_pred, uidx_truth in zip(user_idx_pred, user_idx_truth):
            user_predictions_idxs = pred.get_user_interactions(uidx_pred, as_indices=True)
            user_truth_idxs = truth.get_user_interactions(uidx_truth, as_indices=True)

            user_truth_scores = truth.score_column[user_truth_idxs]
            user_truth_items = truth.item_id_column[user_truth_idxs]

            relevant_threshold = self.relevant_threshold
            if relevant_threshold is None:
                relevant_threshold = np.nanmean(user_truth_scores)

            user_predictions_items = pred.item_id_column[user_predictions_idxs]
            user_truth_relevant_items = user_truth_items[np.where(user_truth_scores >= relevant_threshold)]

            if len(user_truth_relevant_items) != 0:
                user_reciprocal_rank = self.calc_reciprocal_rank(user_predictions_items, user_truth_relevant_items)
            else:
                user_reciprocal_rank = np.nan

            rr_list.append(user_reciprocal_rank)

        # trick to check for nan values, if all values are nan then an exception is thrown
        if all(rr != rr for rr in rr_list):
            raise ValueError("No user has a rating above the given threshold! Try lower it")

        mrr = np.nanmean(rr_list)

        split_result['user_id'].append('sys')
        split_result[str(self)].append(mrr)

        return pd.DataFrame(split_result)


class MRRAtK(MRR):
    r"""
    The MRR@K (Mean Reciprocal Rank at K) metric is a system wide metric, so only its result will be returned and
    not those of every user.
    MRR@K is calculated as such

    $$
    MRR@K_{sys} = \frac{1}{|Q|}\cdot\sum_{i=1}^{K}\frac{1}{rank(i)}
    $$

    Where:

    - $K$ is the cutoff parameter
    - $Q$ is the set of recommendation lists
    - $rank(i)$ is the position of the first relevant item in the i-th recommendation list


    Args:
        k (int): the cutoff parameter. It must be >= 1, otherwise a ValueError exception is raised
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used

    Raises:
        ValueError: if an invalid cutoff parameter is passed (0 or negative)
    """

    def __init__(self, k: int, relevant_threshold: float = None):
        if k < 1:
            raise ValueError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k
        super().__init__(relevant_threshold)

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "MRR@{}".format(self.k)

    def __repr__(self):
        return f'MRRAtK(relevant_threshold={self.relevant_threshold})'

    def calc_reciprocal_rank(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        """
        Method which calculates the RR (Reciprocal Rank) for a single user

        Args:
            user_predictions_items: list of ranked item ids for the user computed by the Recommender
            user_truth_relevant_items: list of relevant item ids for the user in its truth set
        """
        user_predictions_cut = user_predictions_items[:self.k]

        return super().calc_reciprocal_rank(user_predictions_cut, user_truth_relevant_items)


class MAP(RankingMetric):
    r"""

    The $MAP$ metric (*Mean average Precision*) is a ranking metric computed by first calculating the $AP$
    (*Average Precision*) for each user and then taking its mean.

    The $AP$ is calculated as such for the single user:

    $$
    AP_u = \frac{1}{m_u}\sum_{i=1}^{N_u}P(i)\cdot rel(i)
    $$

    Where:

    - $m_u$ is the number of relevant items for the user $u$
    - $N_u$ is the number of recommended items for the user $u$
    - $P(i)$ is the precision computed at cutoff $i$
    - $rel(i)$ is an indicator variable that says whether the i-th item is relevant ($rel(i)=1$) or not ($rel(i)=0$)

    After computing the $AP$ for each user, we can compute the $MAP$ for the whole system:

    $$
    MAP_{sys} = \frac{1}{|U|}\sum_{u}AP_u
    $$

    This metric will return the $AP$ computed for each user in the dataframe containing users results, and the $MAP$
    computed for the whole system in the dataframe containing system results

    Args:
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
    """
    def __init__(self, relevant_threshold: float = None):
        self.relevant_threshold = relevant_threshold

    def _compute_ap(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):

        # all items both in prediction and relevant truth are retrieved, if an item only appears in prediction it is
        # marked with a -1, we then retrieve only the indexes of items that appear in both
        common_idxs = npi.indices(user_truth_relevant_items, user_predictions_items, missing=-1)
        non_missing_idxs = np.where(common_idxs != -1)[0]

        # we initialize an array for true positives. True positive is incremented by 1 each time a relevant item
        # is found, therefore this array will be as long as the array containing the indices of items both in
        # prediction and relevant truth (and values will be as such [1, 2, 3, ...])
        tp_array = np.arange(start=1, stop=len(non_missing_idxs) + 1)

        # finally, precision is computed by dividing each true positive value to each corresponding position
        precision_array = tp_array / (non_missing_idxs + 1)
        cumulative_precision = np.sum(precision_array)

        user_ap = (1 / len(user_truth_relevant_items)) * cumulative_precision

        return user_ap

    @handler_different_users
    def perform(self, split: Split):
        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], 'AP': []}

        # users in truth and pred of the split to evaluate must be the same!
        user_idx_truth = truth.unique_user_idx_column
        user_idx_pred = pred.user_map.convert_seq_str2int(truth.unique_user_id_column)

        for uidx_pred, uidx_truth in zip(user_idx_pred, user_idx_truth):
            user_predictions_idxs = pred.get_user_interactions(uidx_pred, as_indices=True)
            user_truth_idxs = truth.get_user_interactions(uidx_truth, as_indices=True)

            user_truth_scores = truth.score_column[user_truth_idxs]
            user_truth_items = truth.item_id_column[user_truth_idxs]

            relevant_threshold = self.relevant_threshold
            if relevant_threshold is None:
                relevant_threshold = np.nanmean(user_truth_scores)

            user_predictions_items = pred.item_id_column[user_predictions_idxs]
            user_truth_relevant_items = user_truth_items[np.where(user_truth_scores >= relevant_threshold)]

            if len(user_truth_relevant_items) != 0:
                user_ap = self._compute_ap(user_predictions_items, user_truth_relevant_items)
            else:
                user_ap = np.nan

            split_result['user_id'].append(uidx_truth)
            split_result['AP'].append(user_ap)  # for users we are computing Average Precision

        split_result['user_id'] = list(truth.user_map.convert_seq_int2str(split_result['user_id']))
        df_users = pd.DataFrame(split_result)

        # for the system we are computing Mean Average Precision
        sys_map = np.nanmean(df_users['AP'])
        df_sys = pd.DataFrame({'user_id': ['sys'], str(self): [sys_map]})

        df = pd.concat([df_users, df_sys])

        return df

    def __str__(self):
        return "MAP"

    def __repr__(self):
        return f"MAP(relevant_threshold={self.relevant_threshold})"


class MAPAtK(MAP):
    r"""

    The $MAP@K$ metric (*Mean average Precision At K*) is a ranking metric computed by first calculating the $AP@K$
    (*Average Precision At K*) for each user and then taking its mean.

    The $AP@K$ is calculated as such for the single user:

    $$
    AP@K_u = \frac{1}{m_u}\sum_{i=1}^{K}P(i)\cdot rel(i)
    $$

    Where:

    - $m_u$ is the number of relevant items for the user $u$
    - $K$ is the cutoff value
    - $P(i)$ is the precision computed at cutoff $i$
    - $rel(i)$ is an indicator variable that says whether the i-th item is relevant ($rel(i)=1$) or not ($rel(i)=0$)

    After computing the $AP@K$ for each user, we can compute the $MAP@K$ for the whole system:

    $$
    MAP@K_{sys} = \frac{1}{|U|}\sum_{u}AP@K_u
    $$

    This metric will return the $AP@K$ computed for each user in the dataframe containing users results, and the $MAP@K$
    computed for the whole system in the dataframe containing system results

    Args:
        k (int): the cutoff parameter. It must be >= 1, otherwise a ValueError exception is raised
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
    """
    def __init__(self, k: int, relevant_threshold: float = None):
        super().__init__(relevant_threshold)
        self.k = k

    def _compute_ap(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        user_predictions_items = user_predictions_items[:self.k]

        return super()._compute_ap(user_predictions_items, user_truth_relevant_items)

    def __str__(self):
        return "MAPAtK"

    def __repr__(self):
        return f"MAPAtK(k={self.k}, relevant_threshold={self.relevant_threshold})"


class Correlation(RankingMetric):
    r"""
    The Correlation metric calculates the correlation between the ranking of a user and its ideal ranking.
    The currently correlation methods implemented are:

    - `pearson`
    - `kendall`
    - `spearman`

    Every correlation method is implemented by the pandas library, so read its [documentation][pd_link] for more

    [pd_link]: https://pandas.pydata.org/docs/reference/api/pandas.Series.corr.html

    The correlation metric is calculated as such for the **single user**:

    $$
    Corr_u = Corr(ranking_u, ideal\_ranking_u)
    $$

    Where:

    - $ranking_u$ is ranking of the user
    - $ideal\_ranking_u$ is the ideal ranking for the user

    The ideal ranking is calculated based on the rating inside the *ground truth* of the user

    The Correlation metric calculated for the **entire system** is simply the average of every $Corr$:

    $$
    Corr_{sys} = \frac{\sum_{u} Corr_u}{|U|}
    $$

    Where:

    - $Corr_u$ is the correlation of the user $u$
    - $U$ is the set of all users

    The system average excludes NaN values.

    It's also possible to specify a cutoff parameter thanks to the 'top_n' parameter: if specified, only the first
    $n$ results of the recommendation list will be used in order to calculate the correlation

    Args:
        method (str): The correlation method to use. It must be 'pearson', 'kendall' or 'spearman', otherwise a
            ValueError exception is raised. By default is 'pearson'
        top_n (int): Cutoff parameter, if specified only the first n items of the recommendation list will be used
            in order to calculate the correlation

    Raises:
        ValueError: if an invalid method parameter is passed
    """

    def __init__(self, method: str = 'pearson', top_n: int = None):
        valid = {'pearson', 'kendall', 'spearman'}
        self.__method = method.lower()

        if self.__method not in valid:
            raise ValueError("Method {} is not supported! Methods available:\n"
                             "{}".format(method, valid))

        self.__top_n = top_n

    def __str__(self):
        name = self.__method
        if self.__top_n is not None:
            name += " - Top {}".format(self.__top_n)

        return name

    def __repr__(self):
        return f'Correlation(method={self.__method}, top_n={self.__top_n})'

    @handler_different_users
    def perform(self, split: Split) -> pd.DataFrame:
        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}

        # users in truth and pred of the split to evaluate must be the same!
        user_idx_truth = truth.unique_user_idx_column
        user_idx_pred = pred.user_map.convert_seq_str2int(truth.unique_user_id_column)

        for uidx_pred, uidx_truth in zip(user_idx_pred, user_idx_truth):
            user_predictions_idxs = pred.get_user_interactions(uidx_pred, as_indices=True, head=self.__top_n)
            user_truth_idxs = truth.get_user_interactions(uidx_truth, as_indices=True)

            user_prediction_items = pred.item_id_column[user_predictions_idxs]
            user_prediction_scores = pred.score_column[user_predictions_idxs]

            user_truth_items = truth.item_id_column[user_truth_idxs]

            idx_truth_in_pred = npi.indices(user_prediction_items, user_truth_items, missing=-1)
            idx_truth_not_in_pred = np.where(idx_truth_in_pred == -1)

            user_pred_common_idxs = np.delete(idx_truth_in_pred, idx_truth_not_in_pred)
            user_truth_common_idxs = np.delete(user_truth_idxs, idx_truth_not_in_pred)

            common_truth_scores = truth.score_column[user_truth_common_idxs]
            common_prediction_scores = user_prediction_scores[user_pred_common_idxs]

            if len(common_truth_scores) < 2:
                coef = np.nan
            else:
                truth_scores = pd.Series(common_truth_scores)
                rank_scores = pd.Series(common_prediction_scores)
                coef = rank_scores.corr(truth_scores, method=self.__method)

            split_result['user_id'].append(uidx_truth)
            split_result[str(self)].append(coef)

        split_result['user_id'] = list(truth.user_map.convert_seq_int2str(split_result['user_id']))

        split_result['user_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)
