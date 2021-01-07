import pandas as pd
import numpy as np

from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.utils.const import logger


class PredictionMetric(Metric):
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Method that execute the prediction metric computation

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating
        """
        raise NotImplementedError


class RMSE(PredictionMetric):
    """
    RMSE
    .. image:: metrics_img/rmse.png
    \n\n
    Where T is the test set and r' is the actual score give by user u to item i
    """
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Compute the RMSE metric
        
        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating

        Returns:
            (float): The Root Mean Squared Error
        """
        logger.info("Computing RMSE")

        predictions = pd.Series(predictions['rating'].values, name="rating", dtype=float)
        truth = pd.Series(truth['rating'].values, name="rating", dtype=float)

        if len(predictions) != len(truth):
            if len(predictions) > len(truth):
                predictions = predictions[0:len(truth)]
            else:
                truth = truth[0:len(predictions)]
        diff = predictions - truth
        sq = np.square(diff)
        return np.sqrt(np.mean(sq))


class MAE(PredictionMetric):
    """
    MAE
    .. image:: metrics_img/mae.png
    """
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Compute the MAE metric
        
        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating

        Returns:
            (float): The Mean Average Error
        """
        logger.info("Computing MAE")

        predictions = pd.Series(predictions['rating'].values, name="rating", dtype=float)
        truth = pd.Series(truth['rating'].values, name="rating", dtype=float)

        if len(predictions) != len(truth):
            if len(predictions) > len(truth):
                predictions = predictions[0:len(truth)]
            else:
                truth = truth[0:len(predictions)]
        abs_diff = (predictions - truth).apply(abs)
        return np.mean(abs_diff)
