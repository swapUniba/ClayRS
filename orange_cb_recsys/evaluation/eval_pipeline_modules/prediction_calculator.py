from typing import List

from orange_cb_recsys.evaluation.exceptions import AlreadyFittedRecSys
from orange_cb_recsys.evaluation.metrics.metrics import Metric, RankingNeededMetric

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.const import logger

import pandas as pd


class PredictionCalculator:

    def __init__(self, split_list: List[Split], recsys: RecSys):
        self.__split_list = split_list
        self.__recsys = recsys

    def calc_predictions(self, test_items_list: List[pd.DataFrame], metric_list: List[Metric]):

        metric_valid = metric_list.copy()

        # We must clean otherwise on multiple runs if we want to calc other predictions
        # the predictions are already there and they won't overwrite. So before every run of
        # calc_predictions first of all we reset the predictions calculated previously
        for metric in metric_list:
            metric._clean_pred_truth_list()

        for metric in metric_list:

            try:
                metric.eval_fit_recsys(self.__recsys, self.__split_list, test_items_list)

            except AlreadyFittedRecSys:
                continue
            except NotPredictionAlg:
                logger.warning("The RecSys chosen can't predict the score, only ranking!\n"
                               "The {} metric won't be calculated".format(metric))
                metric_valid.remove(metric)
                continue

        return metric_valid
