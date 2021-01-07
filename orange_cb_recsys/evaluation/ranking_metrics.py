import statistics
from abc import abstractmethod
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr

from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.utils.const import logger


class RankingMetric(Metric):
    """
    Abstract class that generalize ranking metrics.
    It measures the quality of the given predicted ranking

    Args:
        relevance_split: specify how to map each truth score
        to a discrete relevance judgement
    """
    def __init__(self, relevance_split: Dict[int, Tuple[float, float]]):
        self.__relevance_split = relevance_split

    @abstractmethod
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Calculates the metric value

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating;
                  it represents the ranking of all the items in the test set


        """
        raise NotImplementedError


class NDCG(RankingMetric):
    """
    Discounted cumulative gain
    .. image:: metrics_img/dcg.png
    \n\n
    This is then normalized as follows:
    .. image:: metrics_img/ndcg.png
    \n\n
    """
    def __init__(self, relevance_split: Dict[int, Tuple[float, float]] = None):
        super().__init__(relevance_split)
        self.__relevance_split = relevance_split

    def __str__(self):
        return "NDCG"

    @staticmethod
    def perform_DCG(gain_values: pd.Series) -> List[float]:
        """
        Compute the Discounted Cumulative Gain array of a gain vector
        Args:
            gain_values (pd.Series): Series of gains

        Returns:
            dcg (List<float>): array of dcg
        """
        if gain_values.size == 0:
            return []
        dcg = []
        for i, gain in enumerate(gain_values):
            if i == 0:
                dcg.append(gain)
            else:
                dcg.append((gain / np.log2(i + 1)) + dcg[i - 1])
        return dcg

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Compute the Normalized DCG measure using Truth rank as ideal DCG
        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating;
                  it represents the ranking of all the items in the test set

        Returns:
            ndcg (List[float]): array of ndcg
        """
        logger.info("Computing NDCG")

        def discrete(score_: float):
            if self.__relevance_split is not None and len(self.__relevance_split.keys()) != 0:

                shift_class = 0
                while 0 + shift_class not in self.__relevance_split.keys():
                    shift_class += 1
                shift_class += 1  # no negative
                for class_ in self.__relevance_split.keys():
                    min_, max_ = self.__relevance_split[class_]
                    if min_ <= score_ <= max_:  # assumption
                        return class_ + shift_class

                # if score_ not in split ranges
                if score_ > 0.0:
                    return max(self.__relevance_split.keys())
                return min(self.__relevance_split.keys())

            return score_ + 1  # no negative, shift to range(0,2) from range (-1, 1)

        gain = []
        for idx, row in predictions.iterrows():
            label = row['to_id']
            score = discrete(truth.rating[truth['to_id'] == label].values[0])
            gain.append(score)
        gain = np.array(gain)

        igain = gain.copy()
        igain[::-1].sort()
        idcg = NDCG.perform_DCG(pd.Series(igain))
        dcg = NDCG.perform_DCG(pd.Series(gain))
        ndcg = [dcg[x] / (idcg[x]) for x in range(len(idcg))]
        if len(ndcg) == 0:
            return 0.0
        return statistics.mean(ndcg)


class Correlation(RankingMetric):
    def __init__(self, method: str):
        """
        Args:
            method: {'pearson, 'kendall', 'spearman'} or callable
        """
        super().__init__(None)
        self.__method = method

    def __str__(self):
        return self.__method

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
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

        truth_labels = pd.Series(truth['to_id'].values)
        prediction_labels = pd.Series(predictions['to_id'].values)

        t_series = pd.Series()
        p_series = pd.Series()
        for t_index, t_value in truth_labels.iteritems():
            for p_index, p_value in prediction_labels.iteritems():
                if t_value == p_value:
                    t_series = t_series.append(pd.Series(int(t_index)))
                    p_series = p_series.append(pd.Series(int(p_index)))
        if t_series.size > 1:
            coef, p = 0, 0
            if self.__method == 'pearson':
                coef, p = pearsonr(t_series, p_series)
            if self.__method == 'kendall':
                coef, p = kendalltau(t_series, p_series)
            if self.__method == 'spearman':
                coef, p = spearmanr(t_series, p_series)

            return coef
        return 0.0
