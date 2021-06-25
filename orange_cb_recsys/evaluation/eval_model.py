from typing import List

from orange_cb_recsys.evaluation.eval_pipeline_modules.metric_evaluator import MetricCalculator
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import PartitionModule
from orange_cb_recsys.evaluation.eval_pipeline_modules.prediction_calculator import PredictionCalculator
from orange_cb_recsys.evaluation.eval_pipeline_modules.methodology import Methodology, TestRatingsMethodology
from orange_cb_recsys.evaluation.metrics.metrics import Metric
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import Partitioning
from orange_cb_recsys.recsys.recsys import RecSys


class EvalModel:
    """
    Class for automating the process of recommending and
    evaluate produced recommendations
    Args:
        config (RecSysConfig): Configuration of the
        recommender system that will be internally created
        partitioning (Partitioning): Partitioning technique
        metric_list (list<Metric>): List of metrics that eval model will compute
    """
    def __init__(self, recsys: RecSys,
                 partitioning: Partitioning,
                 metric_list: List[Metric],
                 methodology: Methodology = TestRatingsMethodology()):

        self.__recsys = recsys
        self.__partitioning = partitioning
        self.__metrics = metric_list
        self.__methodology = methodology

    @property
    def partitioning(self):
        return self.__partitioning

    @property
    def recsys(self):
        return self.__recsys

    @property
    def metrics(self):
        return self.__metrics

    @property
    def methodology(self):
        return self.__methodology

    def append_metric(self, metric: Metric):
        self.__metrics.append(metric)

    def fit(self, user_id_list: list = None):
        """
        This method performs the evaluation by initializing
        internally a recommender system that produces
        recommendations for all the users in the directory
        specified in the configuration phase.
        The evaluation is performed by creating a training set,
        and a test set with its corresponding
        truth base. The ranking algorithm will use the test set as candidate items list.

        Returns:
            ranking_metric_results: has a 'from' column, representing the user_ids for
                which the metrics was computed, and then one different column for every metric
                performed. The returned DataFrames contain one row per user, and the corresponding
                metric values are given by the mean of the values obtained for that user.
        """

        if user_id_list is None:
            user_id_list = set(self.recsys.rating_frame.from_id)

        splitted_ratings = PartitionModule(self.partitioning).split_all(self.recsys.rating_frame, user_id_list)

        test_items_list = self.methodology.get_item_to_predict(splitted_ratings)

        metric_valid = PredictionCalculator(splitted_ratings, self.recsys).calc_predictions(test_items_list, self.metrics)

        # We pass the parameter at None so that the MetricCalculator will use the predictions
        # calculated with the PredictionCalculator module. Those predictions are in the class attribute
        # of every metric
        result = MetricCalculator(None).eval_metrics(metric_valid)

        return result
