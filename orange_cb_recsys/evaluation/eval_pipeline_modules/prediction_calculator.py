import logging
from typing import List

from orange_cb_recsys.evaluation.exceptions import AlreadyFittedRecSys
from orange_cb_recsys.evaluation.metrics.metrics import Metric

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.const import eval_logger, recsys_logger, utils_logger

import pandas as pd


class PredictionCalculator:

    def __init__(self, split_list: List[Split], recsys: RecSys):
        self.__split_list = split_list
        self.__recsys = recsys

    def calc_predictions(self, test_items_list: List[pd.DataFrame], metric_list: List[Metric], verbose: bool = False):

        metric_valid = metric_list.copy()

        # We must clean otherwise on multiple runs if we want to calc other predictions
        # the predictions are already there and they won't overwrite. So before every run of
        # calc_predictions first of all we reset the predictions calculated previously
        for metric in metric_list:
            metric._clean_pred_truth_list()

        eval_logger.info("Calculating predictions needed to evaluate the RecSys")

        if verbose:
            precedent_level_recsys_logger = recsys_logger.getEffectiveLevel()
            recsys_logger.setLevel(logging.WARNING)
        else:
            precedent_level_recsys_logger = recsys_logger.getEffectiveLevel()
            recsys_logger.setLevel(logging.CRITICAL)

        precedent_level_utils_logger = utils_logger.getEffectiveLevel()
        utils_logger.setLevel(logging.CRITICAL)

        for metric in metric_list:

            try:
                metric.eval_fit_recsys(self.__recsys, self.__split_list, test_items_list)

            except AlreadyFittedRecSys:
                continue
            except NotPredictionAlg:
                eval_logger.warning("The RecSys chosen can't predict the score, only ranking!\n"
                                    "The {} metric won't be calculated".format(metric))
                metric_valid.remove(metric)
                continue

        recsys_logger.setLevel(precedent_level_recsys_logger)
        utils_logger.setLevel(precedent_level_utils_logger)
        return metric_valid
