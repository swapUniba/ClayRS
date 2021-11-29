from typing import List

from orange_cb_recsys.evaluation.eval_pipeline_modules.metric_evaluator import MetricCalculator
from orange_cb_recsys.recsys.partitioning import PartitionModule
from orange_cb_recsys.evaluation.eval_pipeline_modules.prediction_calculator import PredictionCalculator
from orange_cb_recsys.evaluation.eval_pipeline_modules.methodology import Methodology, TestRatingsMethodology
from orange_cb_recsys.evaluation.metrics.metrics import Metric
from orange_cb_recsys.recsys.partitioning import Partitioning
from orange_cb_recsys.recsys.recsys import RecSys


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
    def __init__(self, recsys: RecSys,
                 partitioning: Partitioning,
                 metric_list: List[Metric],
                 methodology: Methodology = TestRatingsMethodology(),
                 verbose_predictions: bool = False):

        self.__recsys = recsys
        self.__partitioning = partitioning
        self.__metrics = metric_list
        self.__methodology = methodology
        self.__verbose_predictions = verbose_predictions

    @property
    def partitioning(self):
        """
        Partitioning technique that will be used to split the original dataframe containing interactions between
        users and items in 'train set' and 'test set'
        """
        return self.__partitioning

    @property
    def recsys(self):
        """
        Recommender System to evaluate
        """
        return self.__recsys

    @property
    def metrics(self):
        """
        List of metrics that eval model will compute for the recsys
        """
        return self.__metrics

    @property
    def methodology(self):
        """
        Methodology to use for evaluating the recsys, TestRatings methodology is used by default
        """
        return self.__methodology

    @property
    def verbose_predictions(self):
        """
        Bool parameter which enables or disables the logger for the recommender module while generating recommendations
        for every user that will be evaluated
        """
        return self.__verbose_predictions

    def append_metric(self, metric: Metric):
        """
        Append a metric to the metric list that will be used to evaluate the recommender system

        Args:
            metric (Metric): Metric to append to the metric list
        """
        self.__metrics.append(metric)

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

        if user_id_list is None:
            user_id_list = self.recsys.users

        splitted_ratings = PartitionModule(self.partitioning).split_all(self.recsys.rating_frame, user_id_list)

        test_items_list = self.methodology.get_item_to_predict(splitted_ratings)

        metric_valid = PredictionCalculator(splitted_ratings, self.recsys).calc_predictions(
            test_items_list, self.metrics, self.verbose_predictions)

        # We pass the parameter at None so that the MetricCalculator will use the predictions
        # calculated with the PredictionCalculator module. Those predictions are in the class attribute
        # of every metric
        result = MetricCalculator(None).eval_metrics(metric_valid)

        return result
