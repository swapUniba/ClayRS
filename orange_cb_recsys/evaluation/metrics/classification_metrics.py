import statistics
from abc import abstractmethod

from orange_cb_recsys.evaluation.exceptions import StringNotSupported, KError
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric

import pandas as pd


class ClassificationMetric(RankingNeededMetric):
    """
    Abstract class that generalize classification metrics.
    A classification metric measure if
    known relevant items are predicted as relevant

    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
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
        """
        Computes the precision of each user's list of recommendations, and averages precision over all users.
        ----------
        actual : a list of lists
            Actual items to be predicted
            example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
        predicted : a list of lists
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        Returns:
        -------
            precision: int
        """

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

        sys_prec = -1
        if self.sys_avg == 'micro':
            sys_prec = self._calc_metric(tp_sys, fp_sys, tn_sys, fn_sys)
        elif self.sys_avg == 'macro':
            sys_prec = statistics.mean(split_result[str(self)])

        split_result['from_id'].append('sys')
        split_result[str(self)].append(sys_prec)

        return pd.DataFrame(split_result)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
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
        raise NotImplementedError

    @staticmethod
    def _perform_division(numerator: float, denominator: float):
        res = 0.0
        if denominator != 0:
            res = numerator / denominator

        return res


class Precision(ClassificationMetric):
    """
    Precision

    .. image:: metrics_img/precision.png
    \n\n
    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """

    def __str__(self):
        return "Precision - {}".format(self.sys_avg)

    def _calc_metric(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int):
        return self._perform_division(true_positive, (true_positive + false_positive))


class PrecisionAtK(Precision):
    """
    Precision@K

    .. image:: metrics_img/precision.png
    \n\n
    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
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
        return "Precision@{}".format(self.k)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=self.k)


class RPrecision(Precision):
    """
    R-Precision

    .. image:: metrics_img/precision.png
    \n\n
    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """

    def __str__(self):
        return "R-Precision"

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        if self.relevant_threshold is None:
            relevant_threshold = user_merged['score_truth'].mean()
        else:
            relevant_threshold = self.relevant_threshold

        truth_relevant = len(user_merged.query('score_truth >= @relevant_threshold'))
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=truth_relevant)


class Recall(ClassificationMetric):
    """
    Recall

    .. image:: metrics_img/recall.png
    \n\n
    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """

    def __str__(self):
        return "Recall"

    def _calc_metric(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int):
        return self._perform_division(true_positive, (true_positive + false_negative))


class RecallAtK(Recall):
    """
    Recall@K

    .. image:: metrics_img/precision.png
    \n\n
    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
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
        return "Recall@{}".format(self.k)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=self.__k)


class FMeasure(ClassificationMetric):
    """
    FnMeasure

    .. image:: metrics_img/fn.png
    \n\n
    Args:
        n (int): multiplier
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """

    def __init__(self, beta: int = 1, relevant_threshold: float = None, sys_average: str = 'macro'):
        super().__init__(relevant_threshold, sys_average)
        self.__beta = beta

    @property
    def beta(self):
        return self.__beta

    def __str__(self):
        return "F{}".format(self.beta)

    def _calc_metric(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int):
        prec = Precision()._calc_metric(true_positive, false_positive, true_negative, false_negative)
        reca = Recall()._calc_metric(true_positive, false_positive, true_negative, false_negative)

        beta_2 = self.beta ** 2

        num = prec * reca
        den = (beta_2 * prec) + reca

        fbeta = (1 + beta_2) * self._perform_division(num, den)

        return fbeta


class FMeasureAtK(FMeasure):
    """
    FnMeasure

    .. image:: metrics_img/fn.png
    \n\n
    Args:
        n (int): multiplier
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
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
        return "F{}@{}".format(self.beta, self.k)

    def _calc_confusion_matrix_terminology(self, user_merged: pd.DataFrame, cutoff: int = None):
        return super()._calc_confusion_matrix_terminology(user_merged, cutoff=self.k)
