from typing import List, Union

from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Prediction, Rank, Ratings
from orange_cb_recsys.evaluation.eval_pipeline_modules.metric_evaluator import MetricEvaluator
import pandas as pd
from orange_cb_recsys.evaluation.metrics.metrics import Metric
from orange_cb_recsys.recsys.partitioning import Partitioning
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.const import logger


class EvalModel:
    """
    Class for evaluating a recommender system.

    It needs to be specified which partitioning technique must be used and which methodology to use (by default
    TestRatings methodology is used, check the Methodology module documentation for more), as well as the recsys to
    evaluate and on which metrics it must be evaluated.

    This class automates the evaluation for a recommender system, but every part of the evaluation pipeline can be used
    manually. Check the documentation of eval pipeline modules for more

    Args:
        recsys (RecSys): Recommender System to evaluate
        partitioning (Partitioning): Partitioning technique that will be used to split the original dataframe containing
            interactions between users and items in 'train set' and 'test set'
        metric_list (list[Metric]): List of metrics that eval model will compute for the recsys specified
        methodology (Methodology): Methodology to use for evaluating the recsys, TestRatings methodology is used by
            default
        verbose_predictions (bool): If True, the logger is enabled for the Recommender module, printing possible
            warnings. Else, the logger will be disabled for the Recommender module. This parameter is False by default
    """
    def __init__(self,
                 pred_list: Union[List[Prediction], List[Rank]],
                 truth_list: List[Ratings],
                 metric_list: List[Metric]):

        if len(pred_list) != len(truth_list):
            raise ValueError("List containing predictions and list containing ground truth must have the same length!")

        self._pred_list = pred_list
        self._truth_list = truth_list
        self._metric_list = metric_list

    @property
    def pred_list(self):
        return self._pred_list

    @property
    def truth_list(self):
        return self._truth_list

    @property
    def metric_list(self):
        """
        List of metrics that eval model will compute for the recsys
        """
        return self._metric_list

    def append_metric(self, metric: Metric):
        """
        Append a metric to the metric list that will be used to evaluate the recommender system

        Args:
            metric (Metric): Metric to append to the metric list
        """
        self._metric_list.append(metric)

    def fit(self, user_id_list: list = None):
        """
        This method performs the evaluation for all the users of the recommender system or for the user list specified
        in the 'user_id_list' parameter.

        The evaluation is performed by firstly creating a training set and a test set based on the partitioning
        technique specified.
        Then the EvalModel calculates for every user which items must be used to generate recommendations (or to make
        score prediction) based on the methodology chosen, and eventually generate recommendations lists for every users
        and evaluate them based on the metric list specified.

        Note that if a metric needs to calculate score prediction (e.g. MAE, RMSE) and the recsys evaluated doesn't use
        a score prediction algorithm, then the metric will be popped from the metric list

        The method returns two pandas DataFrame: one containing system results for every metric in the metric list, one
        containing users results for every metric eligible

        Returns:
            Two pandas DataFrame, the first will contain the system result for every metric specified inside the metric
            list, the second one will contain every user results for every metric eligible
        """
        logger.info('Performing evaluation on metrics chosen')

        pred_list = self._pred_list
        truth_list = self._truth_list

        if user_id_list is not None:
            user_id_list_set = set([str(user_id) for user_id in user_id_list])

            pred_list = [pred.filter_ratings(user_id_list_set) for pred in self._pred_list]
            truth_list = [truth.filter_ratings(user_id_list_set) for truth in self._truth_list]

        result = MetricEvaluator(pred_list, truth_list).eval_metrics(self.metric_list)

        return result
