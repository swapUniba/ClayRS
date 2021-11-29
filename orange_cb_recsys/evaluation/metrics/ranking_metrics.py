import statistics
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from orange_cb_recsys.evaluation.exceptions import KError
from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric
from orange_cb_recsys.utils.const import logger


class RankingMetric(RankingNeededMetric):
    """
    Abstract class that generalize ranking metrics.
    A ranking metric evaluates the quality of the recommendation list
    """
    def _get_ideal_actual_rank(self, valid: pd.DataFrame):
        """
        Private method which calculates two lists, actual_rank list and ideal_rank list.

        actual_rank - given the ranking of the user, for every item 'i' in the ranking it extracts the rating
        that the user has effectively given to 'i' and adds it to the actual_rank list.
        If the item is not present in the truth, a 0 is added to the list.

        ideal_rank - it's the actual_rank list ordered from the highest rating to the lowest one. It represents the
        perfect ranking for the user

        Args:
            valid (pd.DataFrame): DataFrame which contains ranking for a user and its test set
        """
        actual_rank = []
        for item_id, score in zip(valid['to_id'], valid['score_truth']):
            if not pd.isna(score):
                actual_rank.append(float(score))
            else:
                actual_rank.append(0)

        ideal_rank = sorted(actual_rank, reverse=True)
        return ideal_rank, actual_rank


class NDCG(RankingMetric):
    """
    The NDCG (Normalized Discounted Cumulative Gain) metric is calculated for the **single user** by using the sklearn
    implementation, so be sure to check its documentation for more.

    The NDCG of the **entire system** is calculated instead as such:

    .. math:: NDCG_sys = \\frac{\sum_{u} NDCG_u}{|U|}
    |
    Where:

    - :math:`NDCG_u` is the NDCG calculated for user :math:`u`
    - :math:`U` is the set of all users

    The system average excludes NaN values.
    """

    def __str__(self):
        return "NDCG"

    def _calc_ndcg(self, ideal_rank: np.array, actual_rank: np.array):
        """
        Private method which calculates the NDCG for a single user using sklearn implementation
        """
        return ndcg_score(ideal_rank, actual_rank)

    def perform(self, split: Split):

        pred = split.pred
        truth = split.truth

        split_result = {'from_id': [], str(self): []}

        for user in set(truth.from_id):
            user_predictions = pred.loc[split.pred['from_id'] == user]
            user_truth = truth.loc[split.truth['from_id'] == user]

            user_predictions = user_predictions[['to_id', 'score']]
            user_truth = user_truth[['to_id', 'score']]

            valid = user_predictions.merge(user_truth, on='to_id', how='left',
                                           suffixes=('_pred', '_truth'))

            ideal, actual = self._get_ideal_actual_rank(valid)

            if len(ideal) == 1:
                user_ndcg = 1
            else:
                ideal_rank = np.array([ideal])
                actual_rank = np.array([actual])
                user_ndcg = self._calc_ndcg(ideal_rank, actual_rank)

            split_result['from_id'].append(user)
            split_result[str(self)].append(user_ndcg)

        split_result['from_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)


class NDCGAtK(NDCG):
    """
    The NDCG@K (Normalized Discounted Cumulative Gain at K) metric is calculated for the **single user** by using the
    sklearn implementation, so be sure to check its documentation for more.

    The NDCG@K of the **entire system** is calculated instead as such:

    .. math:: NDCG@K_sys = \\frac{\sum_{u} NDCG@K_u}{|U|}
    |
    Where:

    - :math:`NDCG@K_u` is the NDCG@K calculated for user :math:`u`
    - :math:`U` is the set of all users

    The system average excludes NaN values.

    Args:
        k (int): the cutoff parameter
    """

    def __init__(self, k: int):
        self.__k = k

    def __str__(self):
        return "NDCG@{}".format(self.__k)

    def _calc_ndcg(self, ideal_rank: np.array, actual_rank: np.array):
        return ndcg_score(ideal_rank, actual_rank, k=self.__k)


class MRR(RankingMetric):
    """
    The MRR (Mean Reciprocal Rank) metric is a system wide metric, so only its result it will be returned and not those
    of every user.
    MRR is calculated as such

    .. math:: MRR_sys = \\frac{1}{|Q|}\cdot\sum_{i=1}^{|Q|}\\frac{1}{rank(i)}

    |
    Where:

    - :math:`Q` is the set of recommendation lists
    - :math:`rank(i)` is the position of the first relevant item in the i-th recommendation list

    The MRR metric needs to discern relevant items from the not relevant ones: in order to do that, one could pass a
    custom relevant_threshold parameter that will be applied to every user, so that if a rating of an item
    is >= relevant_threshold, then it's relevant, otherwise it's not.
    If no relevant_threshold parameter is passed then, for every user, its mean rating score will be used

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

    def calc_reciprocal_rank(self, valid: pd.DataFrame):
        """
        Method which calculates the RR (Reciprocal Rank) for a single user
        Args:
            valid (pd.DataFrame): a DataFrame containing the recommendation list and the truth of a single user
        """
        if self.relevant_threshold is None:
            relevant_threshold = valid['score_truth'].mean()
        else:
            relevant_threshold = self.relevant_threshold

        actually_predicted = valid.query('score_pred.notna()', engine='python')
        reciprocal_rank = 0
        i = 1
        for item_id, score in zip(actually_predicted['to_id'], actually_predicted['score_truth']):
            if score >= relevant_threshold:
                reciprocal_rank = 1 / i  # index starts ad 0
                break  # We only need the first relevant item position in the rank

            i += 1

        return reciprocal_rank

    def perform(self, split: Split) -> pd.DataFrame:
        pred = split.pred
        truth = split.truth

        split_result = {'from_id': [], str(self): []}

        rr_list = []
        for user in set(truth['from_id']):
            user_predictions = pred.loc[split.pred['from_id'] == user]
            user_truth = truth.loc[split.truth['from_id'] == user]

            user_predictions = user_predictions[['to_id', 'score']]
            user_truth = user_truth[['to_id', 'score']]

            valid = user_predictions.merge(user_truth, on='to_id', how='outer',
                                           suffixes=('_pred', '_truth'))

            user_reciprocal_rank = self.calc_reciprocal_rank(valid)

            rr_list.append(user_reciprocal_rank)

        mrr = statistics.mean(rr_list)

        split_result['from_id'].append('sys')
        split_result[str(self)].append(mrr)

        return pd.DataFrame(split_result)


class MRRAtK(MRR):
    r"""
    The MRR@K (Mean Reciprocal Rank at K) metric is a system wide metric, so only its result will be returned and
    not those of every user.
    MRR@K is calculated as such

    .. math:: MRR@K_sys = \frac{1}{|Q|}\cdot\sum_{i=1}^{K}\frac{1}{rank(i)}

    |
    Where:

    - :math:`K` is the cutoff parameter
    - :math:`Q` is the set of recommendation lists
    - :math:`rank(i)` is the position of the first relevant item in the i-th recommendation list


    Args:
        k (int): the cutoff parameter. It must be >= 1, otherwise a KError exception is raised
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used

    Raises:
        KError: if an invalid cutoff parameter is passed (0 or negative)
    """

    def __init__(self, k: int, relevant_threshold: float = None):
        if k < 1:
            raise KError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k
        super().__init__(relevant_threshold)

    def __str__(self):
        return "MRR@{}".format(self.__k)

    def calc_reciprocal_rank(self, valid: pd.DataFrame):
        """
        Method which calculates the RR@K (Reciprocal Rank at K) for a single user
        Args:
            valid (pd.DataFrame): DataFrame which contains ranking for a user and its test set
        """
        if self.relevant_threshold is None:
            relevant_threshold = valid['score_truth'].mean()
        else:
            relevant_threshold = self.relevant_threshold

        actually_predicted = valid.query('score_pred.notna()', engine='python').head(self.__k)
        reciprocal_rank = 0
        i = 1
        for item_id, score in zip(actually_predicted['to_id'], actually_predicted['score_truth']):
            if score >= relevant_threshold:
                reciprocal_rank = 1 / i  # index starts ad 0
                break  # We only need the first relevant item position in the rank

            i += 1

        return reciprocal_rank


class Correlation(RankingMetric):
    """
    The Correlation metric calculates the correlation between the ranking of a user and its ideal ranking.
    The currently correlation methods implemented are:

    - pearson
    - kendall
    - spearman

    Every correlation method is implemented by the scipy library, so read its documentation for more

    The correlation metric is calculated as such for the **single user**:

    .. math:: Corr_u = Corr(ranking_u, ideal\_ranking_u)

    |
    Where:

    - :math:`ranking_u` is ranking of the user
    - :math:`ideal\_ranking_u` is the ideal ranking for the user

    The ideal ranking is calculated based on the rating inside the *ground truth* of the user

    The Correlation metric calculated for the **entire system** is simply the average of every :math:`Corr`:

    .. math:: Corr_sys = \\frac{\sum_{u} Corr_u}{|U|}

    |
    Where:

    - :math:`Corr_u` is the correlation of the user :math:`u`
    - :math:`U` is the set of all users

    The system average excludes NaN values.

    It's also possible to specify a cutoff parameter thanks to the 'top_n' parameter: if specified, only the first
    :math:`n` results of the recommendation list will be used in order to calculate the correlation

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

    def perform(self, split: Split) -> pd.DataFrame:
        pred = split.pred
        truth = split.truth

        split_result = {'from_id': [], str(self): []}
        for user in set(truth.from_id):
            user_predictions = pred.loc[split.pred['from_id'] == user]
            user_truth = truth.loc[split.truth['from_id'] == user]

            user_predictions = user_predictions[['to_id', 'score']]
            user_truth = user_truth[['to_id', 'score']]

            valid = user_predictions.merge(user_truth, on='to_id', how='left',
                                           suffixes=('_pred', '_truth'))

            ideal, actual = self._get_ideal_actual_rank(valid)

            if len(actual) < 2:
                coef = np.nan
            else:
                ideal_ranking = pd.Series(ideal)

                actual_ranking = pd.Series(actual)
                coef = actual_ranking.corr(ideal_ranking, method=self.__method)

            split_result['from_id'].append(user)
            split_result[str(self)].append(coef)

        split_result['from_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)

    def _get_ideal_actual_rank(self, valid: pd.DataFrame):
        items_ideal = list(valid.sort_values(by=['score_truth'], ascending=False).dropna()['to_id'])

        actual_rank = [list(valid['to_id']).index(item) for item in items_ideal]

        ideal_rank = [items_ideal.index(item) for item in items_ideal]

        if self.__top_n is not None:
            actual_rank = actual_rank[:self.__top_n]
            ideal_rank = ideal_rank

        return ideal_rank, actual_rank
