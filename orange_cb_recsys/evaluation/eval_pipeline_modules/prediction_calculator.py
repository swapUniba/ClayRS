import logging
from typing import List

from orange_cb_recsys.evaluation.exceptions import AlreadyFittedRecSys
from orange_cb_recsys.evaluation.metrics.metrics import Metric

from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.const import eval_logger, recsys_logger, utils_logger

import pandas as pd


class PredictionCalculator:
    """
    Module of the Evaluation pipeline which has the task of generating recommendations, given a list of Split and a
    recommender system. Every Split contains a 'train set' and a 'test set'.
    \n\n
    The 'recsys' passed as parameter will be trained on the 'train set' and will predict a set of items for every user.

    Args:
        split_list (List[Split]): List of Split, where every split contains a 'train set' and a 'test set'
        recsys (RecSys): Recommender system that will generate recommendations
    """

    def __init__(self, split_list: List[Split], recsys: RecSys):
        self.__split_list = split_list
        self.__recsys = recsys

    def calc_predictions(self, test_items_list: List[pd.DataFrame], metric_list: List[Metric], verbose: bool = False):
        """
        Method which effectively generate recommendations for every user.

        It will calculate predictions for every item inside the 'test_items_list' parameter, usually given by the
        methodology chosen (Check the methodology module for more).
        It also needs the metric list on which the recsys will be later evaluated since every metric needs a different
        type of predictions (The MAE metric needs the RecSys to predict ratings, the NDCG Metric needs the RecSys to
        calculate a rank, etc.). Since not every algorithm is capable of score predicting, in case it is asked to be
        evaluated on metrics that requires it, those metrics will be popped by the metric list

        Args:
            test_items_list (List[pd.DataFrame]): List of DataFrame, where every DataFrame contains, for every user,
                items that must be predicted
            metric_list (List[Metric]): List of Metric that will be later used to evaluate the recommender system.
                It's needed since every metric needs a different type of prediction (Ranking or score prediction).
                In case the recsys is not able to score predict and a metric requiring score prediction is present into
                the list, it will be popped.
            verbose (bool): If True, the logger is enabled for the Recommender module, printing possible
                warnings. Else, the logger will be disabled for the Recommender module.
                This parameter is False by default.

        Returns:
            The list of metric originally passed as parameter minus metrics that were passed but cannot be calculated
            for the recsys
        """

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
