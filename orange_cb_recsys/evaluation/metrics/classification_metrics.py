import statistics
from abc import abstractmethod

from orange_cb_recsys.evaluation.exceptions import StringNotSupported, KError
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric

import pandas as pd


class ClassificationMetric(RankingNeededMetric):
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
            raise StringNotSupported("Average {} is not supported! Average methods available for {} are:\n"
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

        split_result = {'from_id': [], str(self): []}
        tp_sys = 0
        fp_sys = 0
        tn_sys = 0
        fn_sys = 0
        for user in set(truth['from_id']):
            user_predictions = pred.loc[split.pred['from_id'] == user]
            user_truth = truth.loc[split.truth['from_id'] == user]

            user_predictions = user_predictions[['to_id', 'score']]
            user_truth = user_truth[['to_id', 'score']]

            user_merged = user_predictions.merge(user_truth, on='to_id', how='outer',
                                                 suffixes=('_pred', '_truth'))

            tp_user, fp_user, tn_user, fn_user = self._calc_confusion_matrix_terminology(user_merged)

            metric_user = self._calc_metric(tp_user, fp_user, tn_user, fn_user)

            tp_sys += tp_user
            fp_sys += fp_user
            tn_sys += tn_user
            fn_sys += fn_user

            split_result['from_id'].append(user)
            split_result[str(self)].append(metric_user)

        sys_metric = -1
        if self.sys_avg == 'micro':
            sys_metric = self._calc_metric(tp_sys, fp_sys, tn_sys, fn_sys)
        elif self.sys_avg == 'macro':
            sys_metric = statistics.mean(split_result[str(self)])

        split_result['from_id'].append('sys')
        split_result[str(self)].append(sys_metric)

        return pd.DataFrame(split_result)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        """
        Private method which will calculate true positive, false positive, true negative, false negative
        items for a single user.

        If the 'cutoff' parameter is specified, then recommendation list will be reduced to the top-n items where
        :math:`n = cutoff`

        Args:
            user_merged (pd.DataFrame): a DataFrame containing recommendation list of the user and its ground truth
            cutoff (int): if specified the user recommendation list will be reduced to the top-n items

        Returns:
            true positive, false positive, true negative, false negative for the user
        """
        if self.relevant_threshold is None:
            relevant_threshold = user_merged['score_truth'].mean()
        else:
            relevant_threshold = self.relevant_threshold

        if cutoff:
            # We consider as 'not_predicted' also those excluded from cutoff other than those
            # not effectively retrieved (score_pred is nan)
            actually_predicted = user_merged.query('score_pred.notna()', engine='python')[:cutoff]
            not_predicted = user_merged.query('score_pred.notna()', engine='python')[cutoff:]
            if not user_merged.query('score_pred.isna()', engine='python').empty:
                not_predicted = pd.concat([not_predicted, user_merged.query('score_pred.isna()', engine='python')])
        else:
            actually_predicted = user_merged.query('score_pred.notna()', engine='python')
            not_predicted = user_merged.query('score_pred.isna()', engine='python')

        tp = len(actually_predicted.query('score_truth >= @relevant_threshold'))
        fp = len(actually_predicted.query('(score_truth < @relevant_threshold) or (score_truth.isna())', engine='python'))
        tn = len(not_predicted.query('score_truth < @relevant_threshold'))
        fn = len(not_predicted.query('score_truth >= @relevant_threshold'))

        return tp, fp, tn, fn

    @abstractmethod
    def _calc_metric(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int):
        """
        Private method that must be implemented by every metric which must specify how to use the confusion matrix
        terminology in order to compute the metric
        """
        raise NotImplementedError

    @staticmethod
    def _perform_division(numerator: float, denominator: float):
        """
        Simple static method which performs division given the numerator and the denominator

        If the denominator is 0, then the method will return 0
        Args:
            numerator (float): upper part of the fraction
            denominator (float): lower part of the fraction

        Returns:
            numerator/division if division != 0, 0 otherwise
        """
        res = 0.0
        if denominator != 0:
            res = numerator / denominator

        return res


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

    def _calc_metric(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int):
        return self._perform_division(true_positive, (true_positive + false_positive))


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
            raise KError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "Precision@{} - {}".format(self.k, self.sys_avg)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=self.k)


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

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        if self.relevant_threshold is None:
            relevant_threshold = user_merged['score_truth'].mean()
        else:
            relevant_threshold = self.relevant_threshold

        truth_relevant = len(user_merged.query('score_truth >= @relevant_threshold'))
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=truth_relevant)


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

    def _calc_metric(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int):
        return self._perform_division(true_positive, (true_positive + false_negative))


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
            raise KError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "Recall@{} - {}".format(self.k, self.sys_avg)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=self.__k)


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

    def _calc_metric(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int):
        prec = Precision()._calc_metric(true_positive, false_positive, true_negative, false_negative)
        reca = Recall()._calc_metric(true_positive, false_positive, true_negative, false_negative)

        beta_2 = self.beta ** 2

        num = prec * reca
        den = (beta_2 * prec) + reca

        fbeta = (1 + beta_2) * self._perform_division(num, den)

        return fbeta


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
            raise KError('k={} not valid! k must be >= 1!'.format(k))
        self.__k = k

    @property
    def k(self):
        return self.__k

    def __str__(self):
        return "F{}@{} - {}".format(self.beta, self.k, self.sys_avg)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=self.k)
