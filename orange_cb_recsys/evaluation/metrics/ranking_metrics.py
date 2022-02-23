import statistics
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import Interaction, Ratings
from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.evaluation.metrics.metrics import Metric


class RankingMetric(Metric):
    """
    Abstract class that generalize ranking metrics.
    A ranking metric evaluates the quality of the recommendation list
    """
    pass


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

    def _get_ideal_actual_rank(self, user_predictions: List[Interaction], user_truth: List[Interaction]):
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
        # important that predicted items is a list, we must maintain the order
        predicted_items = [interaction.item_id for interaction in user_predictions]
        item_score_truth = {interaction.item_id: interaction.score for interaction in user_truth}

        actual_rank = [item_score_truth.get(item_id)
                       if item_score_truth.get(item_id) is not None
                       else 0
                       for item_id in predicted_items]

        ideal_rank = sorted(actual_rank, reverse=True)
        return ideal_rank, actual_rank

    def perform(self, split: Split):

        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}

        for user in set(truth.user_id_column):
            user_predictions = pred.get_user_interactions(user)
            user_truth = truth.get_user_interactions(user)

            ideal, actual = self._get_ideal_actual_rank(user_predictions, user_truth)

            if len(ideal) == 1:
                user_ndcg = 1
            else:
                ideal_rank = np.array([ideal])
                actual_rank = np.array([actual])
                user_ndcg = self._calc_ndcg(ideal_rank, actual_rank)

            split_result['user_id'].append(user)
            split_result[str(self)].append(user_ndcg)

        split_result['user_id'].append('sys')
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

    def calc_reciprocal_rank(self, user_predictions: pd.DataFrame, user_truth: pd.DataFrame, relevant_threshold: float):
        """
        Method which calculates the RR (Reciprocal Rank) for a single user
        Args:
            valid (pd.DataFrame): a DataFrame containing the recommendation list and the truth of a single user
        """
        item_score_truth = {interaction.item_id: interaction.score for interaction in user_truth}

        reciprocal_rank = 0
        i = 1
        for interaction_pred in user_predictions:
            truth_score = item_score_truth.get(interaction_pred.item_id)
            if (truth_score is not None) and (truth_score >= relevant_threshold):
                reciprocal_rank = 1 / i
                break  # We only need the first relevant item position in the rank

            i += 1

        return reciprocal_rank

    def perform(self, split: Split) -> pd.DataFrame:
        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}

        rr_list = []
        for user in set(truth.user_id_column):
            user_predictions = pred.get_user_interactions(user)
            user_truth = truth.get_user_interactions(user)

            relevant_threshold = self.relevant_threshold
            if relevant_threshold is None:
                relevant_threshold = statistics.mean([interaction.score for interaction in user_truth])

            user_reciprocal_rank = self.calc_reciprocal_rank(user_predictions, user_truth, relevant_threshold)

            rr_list.append(user_reciprocal_rank)

        mrr = statistics.mean(rr_list)

        split_result['user_id'].append('sys')
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

    def calc_reciprocal_rank(self, user_predictions: pd.DataFrame, user_truth: pd.DataFrame, relevant_threshold: float):
        """
        Method which calculates the RR (Reciprocal Rank) for a single user
        Args:
            valid (pd.DataFrame): a DataFrame containing the recommendation list and the truth of a single user
        """

        item_score_truth = {interaction.item_id: interaction.score for interaction in user_truth}

        user_predictions_cut = user_predictions[:self.k]
        reciprocal_rank = 0
        i = 1
        for interaction_pred in user_predictions_cut:
            truth_score = item_score_truth.get(interaction_pred.item_id)
            if (truth_score is not None) and (truth_score >= relevant_threshold):
                reciprocal_rank = 1 / i
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

        split_result = {'user_id': [], str(self): []}
        for user in set(truth.user_id_column):
            user_predictions = pred.get_user_interactions(user)
            user_truth = truth.get_user_interactions(user)

            ideal, actual = self._get_ideal_actual_rank(user_predictions, user_truth)

            if len(actual) < 2:
                coef = np.nan
            else:
                ideal_ranking = pd.Series(ideal)
                actual_ranking = pd.Series(actual)
                coef = actual_ranking.corr(ideal_ranking, method=self.__method)

            split_result['user_id'].append(user)
            split_result[str(self)].append(coef)

        split_result['user_id'].append('sys')
        split_result[str(self)].append(np.nanmean(split_result[str(self)]))

        return pd.DataFrame(split_result)

    def _get_ideal_actual_rank(self, user_predictions: List[Interaction], user_truth: List[Interaction]):

        if self.__top_n is not None:
            user_predictions = user_predictions[:self.__top_n]

        # sorting truth on score values
        user_truth_ordered = sorted(user_truth, key=lambda interaction: interaction.score, reverse=True)
        ideal_rank = [interaction.item_id for interaction in user_truth_ordered]

        predicted_items = [interaction.item_id for interaction in user_predictions]

        actual_rank = [predicted_items.index(item)
                       for item in ideal_rank
                       if item in set(predicted_items)]

        # the ideal rank is basically 0, 1, 2, 3 etc.
        return [i for i in range(len(ideal_rank))], actual_rank
