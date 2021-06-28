import statistics
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from orange_cb_recsys.evaluation.exceptions import KError
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric
from orange_cb_recsys.utils.const import logger


class RankingMetric(RankingNeededMetric):
    """
    Abstract class that generalize ranking metrics.
    It measures the quality of the given predicted ranking

    Args:
        relevance_split: specify how to map each truth score
        to a discrete relevance judgement
    """
    def _get_ideal_actual_rank(self, valid: pd.DataFrame):
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
    Discounted cumulative gain
    .. image:: metrics_img/dcg.png
    \n\n
    This is then normalized as follows:
    .. image:: metrics_img/ndcg.png
    \n\n
    """

    def __str__(self):
        return "NDCG"

    def _calc_ndcg(self, ideal_rank: np.array, actual_rank: np.array):
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

    def __init__(self, k: int):
        self.__k = k

    def __str__(self):
        return "NDCG@{}".format(self.__k)

    def _calc_ndcg(self, ideal_rank: np.array, actual_rank: np.array):
        return ndcg_score(ideal_rank, actual_rank, k=self.__k)


class MRR(RankingMetric):
    """
    MRR

    .. image:: metrics_img/mrr.png
    \n\n
    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """
    def __init__(self, relevant_threshold: float = None):
        self.__relevant_threshold = relevant_threshold

    @property
    def relevant_threshold(self):
        return self.__relevant_threshold

    def __str__(self):
        return "MRR"

    def calc_reciprocal_rank(self, valid: pd.DataFrame):
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
        """
        Compute the Mean Reciprocal Rank metric

        https://gist.github.com/bwhite/3726239


        Where:
            • Q is the set of recommendation lists
            • rank(i) is the position of the first relevant item in the i-th recommendation list

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating;
                  it represents the ranking of all the items in the test set,
                  first n will be considered relevant,
                  with n equal to the number of relevant items in the test set

        Returns:
            (float): the mrr value
        """
        logger.info("Computing MRR")

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

    def __init__(self, k: int, relevant_threshold: float = None):
        if k < 1:
            raise KError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k
        super().__init__(relevant_threshold)

    def __str__(self):
        return "MRR@{}".format(self.__k)

    def calc_reciprocal_rank(self, valid: pd.DataFrame):
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

    def __init__(self, method: str = 'pearson', top_n: int = None):
        """
        Args:
            method: {'pearson, 'kendall', 'spearman'} or callable
        """
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
        """
        Compute the correlation between the two ranks

        Args:
            truth (pd.DataFrame): dataframe whose columns are: to_id, rating
            predictions (pd.DataFrame): dataframe whose columns are: to_id, rating;
                it represents the ranking of all the items in the test set

        Returns:
            (float): value of the specified correlation metric
        """
        logger.info("Computing correlation")

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

            if len(actual) == 0:
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
