from __future__ import annotations
from abc import abstractmethod
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from clayrs.evaluation.eval_pipeline_modules.metric_evaluator import Split

from clayrs.evaluation.metrics.metrics import Metric, handler_different_users


class ClassificationMetric(Metric):
    """
    Abstract class that generalize classification metrics.
    A classification metric uses confusion matrix terminology (true positive, false positive, etc.) to classify each
    item predicted

    Args:
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, relevant_threshold: float = None, sys_average: str = 'macro',
                 precision: [Callable] = np.float64):
        valid_avg = {'macro', 'micro'}
        self.__avg = sys_average.lower()

        if self.__avg not in valid_avg:
            raise ValueError("Average {} is not supported! Average methods available for {} are:\n"
                             "{}".format(sys_average, str(self), valid_avg))

        self.__relevant_threshold = relevant_threshold
        self.__precision = precision

    @property
    def relevant_threshold(self):
        return self.__relevant_threshold

    @property
    def sys_avg(self):
        return self.__avg

    @property
    def precision(self):
        return self.__precision

    @handler_different_users
    def perform(self, split: Split) -> pd.DataFrame:
        # This method will calculate for every split true positive, false positive, true negative, false negative items
        # so that every metric must simply implement the method _calc_metric(...).
        # Thanks to polymorphism, everything will work without changing this main method

        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}
        sys_confusion_matrix = np.array([[0, 0],
                                         [0, 0]], dtype=np.int32)

        # users in truth and pred of the split to evaluate must be the same!
        user_idx_truth = truth.unique_user_idx_column
        user_idx_pred = pred.user_map.convert_seq_str2int(truth.unique_user_id_column)

        for uidx_pred, uidx_truth in zip(user_idx_pred, user_idx_truth):
            user_predictions_indices = pred.get_user_interactions(uidx_pred, as_indices=True)
            user_truth_indices = truth.get_user_interactions(uidx_truth, as_indices=True)

            user_truth_scores = truth.score_column[user_truth_indices]
            user_truth_items = truth.item_id_column[user_truth_indices]

            relevant_threshold = self.relevant_threshold
            if relevant_threshold is None:
                relevant_threshold = np.nanmean(user_truth_scores)

            user_predictions_items = pred.item_id_column[user_predictions_indices]
            user_truth_relevant_items = user_truth_items[np.where(user_truth_scores >= relevant_threshold)]

            # If basically the user has not provided a rating to any items greater than the threshold,
            # then we don't consider it since it's not fault of the system
            if len(user_truth_relevant_items) != 0:
                metric_user, user_confusion_matrix = self._perform_single_user(user_predictions_items,
                                                                               user_truth_relevant_items)

                sys_confusion_matrix += user_confusion_matrix
            else:
                metric_user = np.nan

            # temporarily append user_idx, later convert to user id
            split_result['user_id'].append(uidx_truth)
            split_result[str(self)].append(metric_user)

        split_result['user_id'] = list(truth.user_map.convert_seq_int2str(split_result['user_id']))

        # trick to check for nan values, if all values are nan then an exception is thrown
        if all(user_result != user_result for user_result in split_result[str(self)]):
            raise ValueError("No user has a rating above the given threshold! Try lower it")

        sys_metric = -1
        if self.sys_avg == 'micro':
            sys_metric = self._calc_metric(sys_confusion_matrix)
        elif self.sys_avg == 'macro':
            sys_metric = np.nanmean(split_result[str(self)])

        split_result['user_id'].append('sys')
        split_result[str(self)].append(sys_metric)

        return pd.DataFrame(split_result)

    @abstractmethod
    def _calc_metric(self, confusion_matrix: np.ndarray):
        """
        Private method that must be implemented by every metric which must specify how to use the confusion matrix
        terminology in order to compute the metric
        """
        raise NotImplementedError

    @abstractmethod
    def _perform_single_user(self, user_prediction_items: np.ndarray, user_truth_items: np.ndarray):
        raise NotImplementedError


class Precision(ClassificationMetric):
    r"""
    The Precision metric is calculated as such for the **single user**:

    $$
    Precision_u = \frac{tp_u}{tp_u + fp_u}
    $$

    Where:

    - $tp_u$ is the number of items which are in the recommendation list of the user and have a
      rating >= relevant_threshold in its 'ground truth'
    - $fp_u$ is the number of items which are in the recommendation list of the user and have a
      rating < relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    $$ 
    Precision_{sys} - micro = \frac{\sum_{u \in U} tp_u}{\sum_{u \in U} tp_u + \sum_{u \in U} fp_u}
    $$
    
    $$
    Precision_{sys} - macro = \frac{\sum_{u \in U} Precision_u}{|U|}
    $$

    Args:
        relevant_threshold: parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average: specify how the system average must be computed. Default is 'macro'
    """
    def __init__(self, relevant_threshold: float = None, sys_average: str = 'macro',
                 precision: [Callable] = np.float64):
        super(Precision, self).__init__(relevant_threshold, sys_average, precision)

    def __str__(self):
        return "Precision - {}".format(self.sys_avg)

    def __repr__(self):
        return f"Precision(relevant_threshold={self.relevant_threshold}, sys_average={self.sys_avg}, " \
               f"precision={self.precision})"

    def _calc_metric(self, confusion_matrix: np.ndarray):
        tp = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        return self.precision((tp + fp) and tp / (tp + fp) or 0)  # safediv between tp and (tp + fp)

    def _perform_single_user(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        tp = len(np.intersect1d(user_predictions_items, user_truth_relevant_items))
        fp = len(user_predictions_items) - tp

        # we do not compute the full confusion matrix for the user
        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [0, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class PrecisionAtK(Precision):
    r"""
    The Precision@K metric is calculated as such for the **single user**:

    $$
    Precision@K_u = \frac{tp@K_u}{tp@K_u + fp@K_u}
    $$

    Where:

    - $tp@K_u$ is the number of items which are in the recommendation list  of the user
      **cutoff to the first K items** and have a rating >= relevant_threshold in its 'ground truth'
    - $tp@K_u$ is the number of items which are in the recommendation list  of the user
      **cutoff to the first K items** and have a rating < relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    $$
    Precision@K_{sys} - micro = \frac{\sum_{u \in U} tp@K_u}{\sum_{u \in U} tp@K_u + \sum_{u \in U} fp@K_u}
    $$

    $$
    Precision@K_{sys} - macro = \frac{\sum_{u \in U} Precision@K_u}{|U|}
    $$

    Args:
        k (int): cutoff parameter. Only the first k items of the recommendation list will be considered
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, k: int, relevant_threshold: float = None, sys_average: str = 'macro',
                 precision: [Callable] = np.float64):
        super().__init__(relevant_threshold, sys_average, precision)
        if k < 1:
            raise ValueError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "Precision@{} - {}".format(self.k, self.sys_avg)

    def __repr__(self):
        return f"PrecisionAtK(k={self.k}, relevant_threshold={self.relevant_threshold}, sys_average={self.sys_avg}, " \
               f"precision={self.precision})"

    def _perform_single_user(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        user_prediction_cut = user_predictions_items[:self.k]

        tp = len(np.intersect1d(user_prediction_cut, user_truth_relevant_items))
        fp = len(user_prediction_cut) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [0, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class RPrecision(Precision):
    r"""
    The R-Precision metric is calculated as such for the **single user**:

    $$
    R-Precision_u = \frac{tp@R_u}{tp@R_u + fp@R_u}
    $$

    Where:

    - $R$ it's the number of relevant items for the user *u*
    - $tp@R_u$ is the number of items which are in the recommendation list  of the user
      **cutoff to the first R items** and have a rating >= relevant_threshold in its 'ground truth'
    - $tp@R_u$ is the number of items which are in the recommendation list  of the user
      **cutoff to the first R items** and have a rating < relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    $$
    Precision@R_{sys} - micro = \frac{\sum_{u \in U} tp@R_u}{\sum_{u \in U} tp@R_u + \sum_{u \in U} fp@R_u}
    $$

    $$
    Precision@R_{sys} - macro = \frac{\sum_{u \in U} R-Precision_u}{|U|}
    $$

    Args:
        relevant_threshold: parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average: specify how the system average must be computed. Default is 'macro'
    """
    def __init__(self, relevant_threshold: float = None, sys_average: str = 'macro',
                 precision: [Callable] = np.float64):
        super().__init__(relevant_threshold, sys_average, precision)

    def __str__(self):
        return "R-Precision - {}".format(self.sys_avg)

    def __repr__(self):
        return f"RPrecision(relevant_threshold={self.relevant_threshold}, sys_average={self.sys_avg}, " \
               f"precision={self.precision})"

    def _perform_single_user(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        r = len(user_truth_relevant_items)
        user_prediction_cut = user_predictions_items[:r]

        tp = len(np.intersect1d(user_prediction_cut, user_truth_relevant_items))
        fp = len(user_prediction_cut) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [0, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class Recall(ClassificationMetric):
    r"""
    The Recall metric is calculated as such for the **single user**:

    $$
    Recall_u = \frac{tp_u}{tp_u + fn_u}
    $$

    Where:

    - $tp_u$ is the number of items which are in the recommendation list of the user and have a
      rating >= relevant_threshold in its 'ground truth'
    - $fn_u$ is the number of items which are NOT in the recommendation list of the user and have a
      rating >= relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    $$
    Recall_{sys} - micro = \frac{\sum_{u \in U} tp_u}{\sum_{u \in U} tp_u + \sum_{u \in U} fn_u}
    $$

    $$
    Recall_{sys} - macro = \frac{\sum_{u \in U} Recall_u}{|U|}
    $$

    Args:
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, relevant_threshold: float = None, sys_average: str = 'macro',
                 precision: [Callable] = np.float64):
        super().__init__(relevant_threshold, sys_average, precision)

    def __str__(self):
        return "Recall - {}".format(self.sys_avg)

    def __repr__(self):
        return f"Recall(relevant_threshold={self.relevant_threshold}, sys_average={self.sys_avg}, " \
               f"precision={self.precision})"

    def _calc_metric(self, confusion_matrix: np.ndarray):
        tp = confusion_matrix[0, 0]
        fn = confusion_matrix[1, 0]
        return self.precision((tp + fn) and tp / (tp + fn) or 0)  # safediv between tp and (tp + fn)

    def _perform_single_user(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        tp = len(np.intersect1d(user_predictions_items, user_truth_relevant_items))
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, 0],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class RecallAtK(Recall):
    r"""
    The Recall@K metric is calculated as such for the **single user**:

    $$
    Recall@K_u = \frac{tp@K_u}{tp@K_u + fn@K_u}
    $$

    Where:

    - $tp@K_u$ is the number of items which are in the recommendation list  of the user
      **cutoff to the first K items** and have a rating >= relevant_threshold in its 'ground truth'
    - $tp@K_u$ is the number of items which are NOT in the recommendation list  of the user
      **cutoff to the first K items** and have a rating >= relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    $$
    Recall@K_{sys} - micro = \frac{\sum_{u \in U} tp@K_u}{\sum_{u \in U} tp@K_u + \sum_{u \in U} fn@K_u}
    $$

    $$
    Recall@K_{sys} - macro = \frac{\sum_{u \in U} Recall@K_u}{|U|}
    $$

    Args:
        k (int): cutoff parameter. Only the first k items of the recommendation list will be considered
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, k: int, relevant_threshold: float = None, sys_average: str = 'macro',
                 precision: [Callable] = np.float64):
        super().__init__(relevant_threshold, sys_average, precision)
        if k < 1:
            raise ValueError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "Recall@{} - {}".format(self.k, self.sys_avg)

    def __repr__(self):
        return f"RecallAtK(k={self.k}, relevant_threshold={self.relevant_threshold}, sys_average={self.sys_avg}, " \
               f"precision={self.precision})"

    def _perform_single_user(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        user_prediction_cut = user_predictions_items[:self.k]

        tp = len(np.intersect1d(user_prediction_cut, user_truth_relevant_items))
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, 0],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class FMeasure(ClassificationMetric):
    r"""
    The FMeasure metric combines Precision and Recall into a single metric. It is calculated as such for the
    **single user**:

    $$
    FMeasure_u = (1 + \beta^2) \cdot \frac{P_u \cdot R_u}{(\beta^2 \cdot P_u) + R_u}
    $$

    Where:

    - $P_u$ is the Precision calculated for the user *u*
    - $R_u$ is the Recall calculated for the user *u*
    - $\beta$ is a real factor which could weight differently Recall or Precision based on its value:

        - $\beta = 1$: Equally weight Precision and Recall
        - $\beta > 1$: Weight Recall more
        - $\beta < 1$: Weight Precision more

    A famous FMeasure is the F1 Metric, where $\beta = 1$, which basically is the harmonic mean of recall and
    precision:

    $$
    F1_u = \frac{2 \cdot P_u \cdot R_u}{P_u + R_u}
    $$

    The FMeasure metric is calculated as such for the **entire system**, depending if 'macro' average or 'micro'
    average has been chosen:

    $$
    FMeasure_{sys} - micro = (1 + \beta^2) \cdot \frac{P_u \cdot R_u}{(\beta^2 \cdot P_u) + R_u}
    $$

    $$
    FMeasure_{sys} - macro = \frac{\sum_{u \in U} FMeasure_u}{|U|}
    $$

    Args:
        beta (float): real factor which could weight differently Recall or Precision based on its value. Default is 1
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, beta: float = 1, relevant_threshold: float = None, sys_average: str = 'macro',
                 precision: [Callable] = np.float64):
        super().__init__(relevant_threshold, sys_average, precision)
        self.__beta = beta

    @property
    def beta(self):
        return self.__beta

    def __str__(self):
        return "F{} - {}".format(self.beta, self.sys_avg)

    def __repr__(self):
        return f"FMeasure(beta={self.beta}, relevant_threshold={self.relevant_threshold}, " \
               f"sys_average={self.sys_avg}, precision={self.precision})"

    def _calc_metric(self, confusion_matrix: np.ndarray):
        prec = Precision()._calc_metric(confusion_matrix)
        reca = Recall()._calc_metric(confusion_matrix)

        beta_2 = self.beta ** 2

        num = prec * reca
        den = (beta_2 * prec) + reca

        fbeta = self.precision((1 + beta_2) * (den and num / den or 0))  # safediv between num and den

        return fbeta

    def _perform_single_user(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        tp = len(np.intersect1d(user_predictions_items, user_truth_relevant_items))
        fp = len(user_predictions_items) - tp
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class FMeasureAtK(FMeasure):
    r"""
    The FMeasure@K metric combines Precision@K and Recall@K into a single metric. It is calculated as such for the
    **single user**:

    $$
    FMeasure@K_u = (1 + \beta^2) \cdot \frac{P@K_u \cdot R@K_u}{(\beta^2 \cdot P@K_u) + R@K_u}
    $$

    Where:

    - $P@K_u$ is the Precision at K calculated for the user *u*
    - $R@K_u$ is the Recall at K calculated for the user *u*
    - $\beta$ is a real factor which could weight differently Recall or Precision based on its value:

        - $\beta = 1$: Equally weight Precision and Recall
        - $\beta > 1$: Weight Recall more
        - $\beta < 1$: Weight Precision more

    A famous FMeasure@K is the F1@K Metric, where :math:`\beta = 1`, which basically is the harmonic mean of recall and
    precision:

    $$
    F1@K_u = \frac{2 \cdot P@K_u \cdot R@K_u}{P@K_u + R@K_u}
    $$

    The FMeasure@K metric is calculated as such for the **entire system**, depending if 'macro' average or 'micro'
    average has been chosen:

    $$
    FMeasure@K_{sys} - micro = (1 + \beta^2) \cdot \frac{P@K_u \cdot R@K_u}{(\beta^2 \cdot P@K_u) + R@K_u}
    $$

    $$
    FMeasure@K_{sys} - macro = \frac{\sum_{u \in U} FMeasure@K_u}{|U|}
    $$

    Args:
        k (int): cutoff parameter. Will be used for the computation of Precision@K and Recall@K
        beta (float): real factor which could weight differently Recall or Precision based on its value. Default is 1
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, k: int, beta: int = 1, relevant_threshold: float = None, sys_average: str = 'macro'):
        super().__init__(beta, relevant_threshold, sys_average)
        if k < 1:
            raise ValueError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "F{}@{} - {}".format(self.beta, self.k, self.sys_avg)

    def __repr__(self):
        return f"FMeasureAtK(k={self.k}, beta={self.beta}, relevant_threshold={self.relevant_threshold}, " \
               f"sys_average={self.sys_avg}, precision={self.precision})"

    def _perform_single_user(self, user_predictions_items: np.ndarray, user_truth_relevant_items: np.ndarray):
        user_prediction_cut = user_predictions_items[:self.k]

        tp = len(np.intersect1d(user_prediction_cut, user_truth_relevant_items))
        fp = len(user_prediction_cut) - tp
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user
