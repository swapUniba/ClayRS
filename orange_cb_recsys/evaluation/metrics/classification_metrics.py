import statistics
from abc import abstractmethod
from typing import Union, List, Set

import numpy as np

from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Prediction, Rank, Ratings, Interaction
from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.evaluation.metrics.metrics import Metric

import pandas as pd


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

    def __init__(self, relevant_threshold: float = None, sys_average: str = 'macro'):
        valid_avg = {'macro', 'micro'}
        self.__avg = sys_average.lower()

        if self.__avg not in valid_avg:
            raise ValueError("Average {} is not supported! Average methods available for {} are:\n"
                             "{}".format(sys_average, str(self), valid_avg))

        self.__relevant_threshold = relevant_threshold

    @property
    def relevant_threshold(self):
        return self.__relevant_threshold

    @property
    def sys_avg(self):
        return self.__avg

    def perform(self, split: Split) -> pd.DataFrame:
        # This method will calculate for every split true positive, false positive, true negative, false negative items
        # so that every metric must simply implement the method _calc_metric(...).
        # Thanks to polymorphism, everything will work without changing this main method

        pred = split.pred
        truth = split.truth

        split_result = {'user_id': [], str(self): []}
        sys_confusion_matrix = np.array([[0, 0],
                                         [0, 0]], dtype=np.int32)

        for user in set(truth.user_id_column):
            user_predictions = pred.get_user_interactions(user)
            user_truth = truth.get_user_interactions(user)

            relevant_threshold = self.relevant_threshold
            if relevant_threshold is None:
                relevant_threshold = statistics.mean([truth_interaction.score
                                                      for truth_interaction in user_truth])

            user_truth_relevant_items = set([truth_interaction.item_id for truth_interaction in user_truth
                                             if truth_interaction.score >= relevant_threshold])

            metric_user, user_confusion_matrix = self._perform_single_user(user_predictions, user_truth_relevant_items)

            sys_confusion_matrix += user_confusion_matrix

            split_result['user_id'].append(user)
            split_result[str(self)].append(metric_user)

        sys_metric = -1
        if self.sys_avg == 'micro':
            sys_metric = self._calc_metric(sys_confusion_matrix)
        elif self.sys_avg == 'macro':
            sys_metric = statistics.mean(split_result[str(self)])

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
    def _perform_single_user(self, user_prediction_items: List[Interaction], user_truth_items: Set[str]):
        raise NotImplementedError


class Precision(ClassificationMetric):
    r"""
    The Precision metric is calculated as such for the **single user**:

    .. math:: Precision_u = \frac{tp_u}{tp_u + fp_u}

    |
    Where:

    - :math:`tp_u` is the number of items which are in the recommendation list of the user and have a
      rating >= relevant_threshold in its 'ground truth'
    - :math:`fp_u` is the number of items which are in the recommendation list of the user and have a
      rating < relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    .. math::
        Precision_sys - micro = \frac{\sum_{u \in U} tp_u}{\sum_{u \in U} tp_u + \sum_{u \in U} fp_u}

        Precision_sys - macro = \frac{\sum_{u \in U} Precision_u}{|U|}

    Args:
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __str__(self):
        return "Precision - {}".format(self.sys_avg)

    def _calc_metric(self, confusion_matrix: np.ndarray):
        tp = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        return np.float32((tp + fp) and tp / (tp + fp) or 0)  # safediv between tp and (tp + fp)

    def _perform_single_user(self, user_prediction: List[Interaction], user_truth_relevant_items: Set[str]):
        tp = len([prediction_interaction for prediction_interaction in user_prediction
                  if prediction_interaction.item_id in user_truth_relevant_items])
        fp = len(user_prediction) - tp

        # we do not compute the full confusion matrix for the user
        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [0, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class PrecisionAtK(Precision):
    r"""
    The Precision@K metric is calculated as such for the **single user**:

    .. math:: Precision@K_u = \frac{tp@K_u}{tp@K_u + fp@K_u}

    |
    Where:

    - :math:`tp@K_u` is the number of items which are in the recommendation list  of the user
      **cutoff to the first K items** and have a rating >= relevant_threshold in its 'ground truth'
    - :math:`tp@K_u` is the number of items which are in the recommendation list  of the user
      **cutoff to the first K items** and have a rating < relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    .. math::
        Precision@K_sys - micro = \frac{\sum_{u \in U} tp@K_u}{\sum_{u \in U} tp@K_u + \sum_{u \in U} fp@K_u}

        Precision@K_sys - macro = \frac{\sum_{u \in U} Precision@K_u}{|U|}

    Args:
        k (int): cutoff parameter. Only the first k items of the recommendation list will be considered
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, k: int, relevant_threshold: float = None, sys_average: str = 'macro'):
        super().__init__(relevant_threshold, sys_average)
        if k < 1:
            raise ValueError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "Precision@{} - {}".format(self.k, self.sys_avg)

    def _perform_single_user(self, user_prediction: List[Interaction], user_truth_relevant_items: Set[str]):
        user_prediction_cut = user_prediction[:self.k]

        tp = len([prediction_interaction for prediction_interaction in user_prediction_cut
                  if prediction_interaction.item_id in user_truth_relevant_items])
        fp = len(user_prediction_cut) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [0, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class RPrecision(Precision):
    r"""
    The R-Precision metric is calculated as such for the **single user**:

    .. math:: R-Precision_u = \frac{tp@R_u}{tp@R_u + fp@R_u}

    |
    Where:

    - :math:`R` it's the number of relevant items for the user *u*
    - :math:`tp@R_u` is the number of items which are in the recommendation list  of the user
      **cutoff to the first R items** and have a rating >= relevant_threshold in its 'ground truth'
    - :math:`tp@R_u` is the number of items which are in the recommendation list  of the user
      **cutoff to the first R items** and have a rating < relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    .. math::
        Precision@K_sys - micro = \frac{\sum_{u \in U} tp@R_u}{\sum_{u \in U} tp@R_u + \sum_{u \in U} fp@R_u}

        Precision@K_sys - macro = \frac{\sum_{u \in U} R-Precision_u}{|U|}

    Args:
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __str__(self):
        return "R-Precision - {}".format(self.sys_avg)

    def _perform_single_user(self, user_prediction: List[Interaction], user_truth_relevant_items: Set[str]):

        r = len(user_truth_relevant_items)
        user_prediction_cut = user_prediction[:r]

        tp = len([prediction_interaction for prediction_interaction in user_prediction_cut
                  if prediction_interaction.item_id in user_truth_relevant_items])
        fp = len(user_prediction_cut) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [0, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class Recall(ClassificationMetric):
    r"""
    The Recall metric is calculated as such for the **single user**:

    .. math:: Recall_u = \frac{tp_u}{tp_u + fn_u}

    |
    Where:

    - :math:`tp_u` is the number of items which are in the recommendation list of the user and have a
      rating >= relevant_threshold in its 'ground truth'
    - :math:`fn_u` is the number of items which are NOT in the recommendation list of the user and have a
      rating >= relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    .. math::
        Recall_sys - micro = \frac{\sum_{u \in U} tp_u}{\sum_{u \in U} tp_u + \sum_{u \in U} fn_u}

        Recall_sys - macro = \frac{\sum_{u \in U} Recall_u}{|U|}

    Args:
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __str__(self):
        return "Recall - {}".format(self.sys_avg)

    def _calc_metric(self, confusion_matrix: np.ndarray):
        tp = confusion_matrix[0, 0]
        fn = confusion_matrix[1, 0]
        return np.float32((tp + fn) and tp / (tp + fn) or 0)  # safediv between tp and (tp + fn)

    def _perform_single_user(self, user_prediction: List[Interaction], user_truth_relevant_items: Set[str]):

        tp = len([prediction_interaction for prediction_interaction in user_prediction
                  if prediction_interaction.item_id in user_truth_relevant_items])
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, 0],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class RecallAtK(Recall):
    r"""
    The Recall@K metric is calculated as such for the **single user**:

    .. math:: Recall@K_u = \frac{tp@K_u}{tp@K_u + fn@K_u}

    |
    Where:

    - :math:`tp@K_u` is the number of items which are in the recommendation list  of the user
      **cutoff to the first K items** and have a rating >= relevant_threshold in its 'ground truth'
    - :math:`tp@K_u` is the number of items which are NOT in the recommendation list  of the user
      **cutoff to the first K items** and have a rating >= relevant_threshold in its 'ground truth'

    And it is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    .. math::
        Recall@K_sys - micro = \frac{\sum_{u \in U} tp@K_u}{\sum_{u \in U} tp@K_u + \sum_{u \in U} fn@K_u}

        Recall@K_sys - macro = \frac{\sum_{u \in U} Recall@K_u}{|U|}

    Args:
        k (int): cutoff parameter. Only the first k items of the recommendation list will be considered
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, k: int, relevant_threshold: float = None, sys_average: str = 'macro'):
        super().__init__(relevant_threshold, sys_average)
        if k < 1:
            raise ValueError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "Recall@{} - {}".format(self.k, self.sys_avg)

    def _perform_single_user(self, user_prediction: List[Interaction], user_truth_relevant_items: Set[str]):

        user_prediction_cut = user_prediction[:self.k]

        tp = len([prediction_interaction for prediction_interaction in user_prediction_cut
                  if prediction_interaction.item_id in user_truth_relevant_items])
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, 0],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class FMeasure(ClassificationMetric):
    r"""
    The FMeasure metric combines Precision and Recall into a single metric. It is calculated as such for the
    **single user**:

    .. math:: FMeasure_u = (1 + \beta^2) \cdot \frac{P_u \cdot R_u}{(\beta^2 \cdot P_u) + R_u}

    |
    Where:

    - :math:`P_u` is the Precision calculated for the user *u*
    - :math:`R_u` is the Recall calculated for the user *u*
    - :math:`\beta` is a real factor which could weight differently Recall or Precision based on its value:

        - :math:`\beta = 1`: Equally weight Precision and Recall
        - :math:`\beta > 1`: Weight Recall more
        - :math:`\beta < 1`: Weight Precision more

    A famous FMeasure is the F1 Metric, where :math:`\beta = 1`, which basically is the harmonic mean of recall and
    precision:

    .. math:: F1_u = \frac{2 \cdot P_u \cdot R_u}{P_u + R_u}
    |

    The FMeasure metric is calculated as such for the **entire system**, depending if 'macro' average or 'micro' average has been
    chosen:

    .. math::
        FMeasure_sys - micro = (1 + \beta^2) \cdot \frac{P_u \cdot R_u}{(\beta^2 \cdot P_u) + R_u}

        FMeasure_sys - macro = \frac{\sum_{u \in U} FMeasure_u}{|U|}

    Args:
        beta (float): real factor which could weight differently Recall or Precision based on its value. Default is 1
        relevant_threshold (float): parameter needed to discern relevant items and non-relevant items for every
            user. If not specified, the mean rating score of every user will be used
        sys_average (str): specify how the system average must be computed. Default is 'macro'
    """

    def __init__(self, beta: float = 1, relevant_threshold: float = None, sys_average: str = 'macro'):
        super().__init__(relevant_threshold, sys_average)
        self.__beta = beta

    @property
    def beta(self):
        return self.__beta

    def __str__(self):
        return "F{} - {}".format(self.beta, self.sys_avg)

    def _calc_metric(self, confusion_matrix: np.ndarray):
        prec = Precision()._calc_metric(confusion_matrix)
        reca = Recall()._calc_metric(confusion_matrix)

        beta_2 = self.beta ** 2

        num = prec * reca
        den = (beta_2 * prec) + reca

        fbeta = (1 + beta_2) * (den and num / den or 0)  # safediv between num and den

        return fbeta

    def _perform_single_user(self, user_prediction: List[Interaction], user_truth_relevant_items: Set[str]):

        tp = len([prediction_interaction for prediction_interaction in user_prediction
                  if prediction_interaction.item_id in user_truth_relevant_items])
        fp = len(user_prediction) - tp
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user


class FMeasureAtK(FMeasure):
    r"""
    The FMeasure@K metric combines Precision@K and Recall@K into a single metric. It is calculated as such for the
    **single user**:

    .. math:: FMeasure_u = (1 + \beta^2) \cdot \frac{P@K_u \cdot R@K_u}{(\beta^2 \cdot P@K_u) + R@K_u}

    |
    Where:

    - :math:`P@K_u` is the Precision at K calculated for the user *u*
    - :math:`R@K_u` is the Recall at K calculated for the user *u*
    - :math:`\beta` is a real factor which could weight differently Recall or Precision based on its value:

        - :math:`\beta = 1`: Equally weight Precision and Recall
        - :math:`\beta > 1`: Weight Recall more
        - :math:`\beta < 1`: Weight Precision more

    A famous FMeasure@K is the F1@K Metric, where :math:`\beta = 1`, which basically is the harmonic mean of recall and
    precision:

    .. math:: F1@K_u = \frac{2 \cdot P@K_u \cdot R@K_u}{P@K_u + R@K_u}
    |

    The FMeasure@K metric is calculated as such for the **entire system**, depending if 'macro' average or 'micro'
    average has been chosen:

    .. math::
        FMeasure@K_sys - micro = (1 + \beta^2) \cdot \frac{P@K_u \cdot R@K_u}{(\beta^2 \cdot P@K_u) + R@K_u}

        FMeasure@K_sys - macro = \frac{\sum_{u \in U} FMeasure@K_u}{|U|}

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

    def _perform_single_user(self, user_prediction: List[Interaction], user_truth_relevant_items: Set[str]):

        user_prediction_cut = user_prediction[:self.k]

        tp = len([prediction_interaction for prediction_interaction in user_prediction_cut
                  if prediction_interaction.item_id in user_truth_relevant_items])
        fp = len(user_prediction_cut) - tp
        fn = len(user_truth_relevant_items) - tp

        useful_confusion_matrix_user = np.array([[tp, fp],
                                                 [fn, 0]], dtype=np.int32)

        return self._calc_metric(useful_confusion_matrix_user), useful_confusion_matrix_user
