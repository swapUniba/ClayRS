import os
from abc import abstractmethod
from typing import List

import pandas as pd

from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.evaluation.partitioning import Partitioning
from orange_cb_recsys.recsys.algorithm import ScorePredictionAlgorithm
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import remove_not_existent_items


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
    def __init__(self, config: RecSysConfig,
                 partitioning: Partitioning,
                 metric_list: List[Metric] = None):
        if metric_list is None:
            metric_list = {}
        self.__metric_list = metric_list
        self.__config: RecSysConfig = config
        self.__partitioning = partitioning

    @property
    def partitioning(self):
        return self.__partitioning

    @property
    def config(self):
        return self.__config

    def append_metric(self, metric: Metric):
        self.__metric_list.append(metric)

    @property
    def metrics(self):
        for metric in self.__metric_list:
            yield metric

    @abstractmethod
    def fit(self):
        raise NotImplementedError


class RankingAlgEvalModel(EvalModel):
    """
    Class for automating the process of recommending and
    evaluate produced recommendations.
    This subclass automate the computation of metrics
    whose input are the result of a RecSys
    configured with a ranking algorithm.
    The metrics are iteratively computed for each user

    Args:
        config (RecSysConfig): Configuration of the
        recommender system that will be internally created
        partitioning (Partitioning): Partitioning technique
        metric_list (list<Metric>): List of metrics that eval model will compute
    """
    def __init__(self, config, partitioning, metric_list: List[Metric] = None):
        super().__init__(config, partitioning, metric_list)

    def fit(self):
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
        # initialize recommender to call for prediction computing
        recsys = RecSys(self.config)

        # get all users in specified directory
        logger.info("Loading user instances")
        user_id_list = \
            [os.path.splitext(filename)[0]
             for filename in os.listdir(self.config.users_directory)]

        # define results structure
        ranking_alg_metrics_results = pd.DataFrame()

        # calculate metrics on ranking algorithm results
        if self.config.ranking_algorithm is None:
            raise ValueError("You must set ranking algorithm to compute ranking metrics")
        for user_id in user_id_list:
            logger.info("Computing ranking metrics for user %s", user_id)
            user_ratings = self.config.rating_frame[
                self.config.rating_frame['from_id'] == user_id]

            try:
                self.partitioning.dataframe = user_ratings
            except ValueError:
                continue

            for partition_index in self.partitioning:
                result_dict = {}
                train = user_ratings.iloc[partition_index[0]]
                test = user_ratings.iloc[partition_index[1]]

                truth = test.loc[:, 'to_id':'score']
                truth.columns = ["to_id", "rating"]
                recs_number = len(truth['rating'].values)
                predictions = recsys.fit_eval_ranking(
                    user_id, train, truth['to_id'].tolist(), recs_number)
                for metric in self.metrics:
                    result_dict['from'] = user_id
                    result_dict[str(metric)] = metric.perform(predictions, truth)

                ranking_alg_metrics_results = \
                    ranking_alg_metrics_results.append(result_dict, ignore_index=True)

        ranking_alg_metrics_results = \
            ranking_alg_metrics_results.groupby('from').mean().reset_index()

        return ranking_alg_metrics_results


class PredictionAlgEvalModel(EvalModel):
    """
    Class for automating the process of recommending and evaluate produced recommendations.
    This subclass automate the computation of metrics
    whose input are the result of a RecSys
    configured with a rating prediction algorithm.
    The metrics are iteratively computed for each user

    Args:
        config (RecSysConfig): Configuration of the
        recommender system that will be internally created
        partitioning (Partitioning): Partitioning technique
        metric_list (list<Metric>): List of metrics that eval model will compute
    """
    def __init__(self, config, partitioning, metric_list: List[Metric] = None):
        super().__init__(config, partitioning, metric_list)

    def fit(self):
        """
        This method performs the rating prediction evaluation by initializing internally
            a recommender system that produces recommendations for all the
            users in the directory specified in the configuration phase.
            The evaluation is performed by creating a training set,
            and a test set with its corresponding
            truth base. The rating prediction will be computed on every item in the test eet.

        Returns:
            prediction_metric_results: has a 'from' column, representing the user_ids for
                which the metrics was computed, and then one different column for every metric
                performed. The returned DataFrames contain one row per user, and the corresponding
                metric values are given by the mean of the values obtained for that user.
        """
        # initialize recommender to call for prediction computing
        recsys = RecSys(self.config)

        # get all users in specified directory
        logger.info("Loading user instances")
        user_id_list = [
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.config.users_directory)]

        # define results structure
        prediction_metric_results = pd.DataFrame()

        # calculate metrics on prediction algorithm results
        if self.config.score_prediction_algorithm is None:
            raise ValueError("You must set score prediction algorithm to compute this eval model")

        for user_id in user_id_list:
            logger.info("User %s", user_id)
            logger.info("Loading user ratings")

            user_ratings = self.config.rating_frame[
                self.config.rating_frame['from_id'] == user_id]
            user_ratings = user_ratings.sort_values(['to_id'], ascending=True)

            try:
                self.partitioning.dataframe = user_ratings
            except ValueError:
                continue

            for partition_index in self.partitioning:
                result_dict = {}
                logger.info("Computing prediction metrics")
                train = user_ratings.iloc[partition_index[0]]
                test = user_ratings.iloc[partition_index[1]]
                test = remove_not_existent_items(test, self.config.items_directory)

                predictions = recsys.fit_eval_predict(user_id, train, test)
                for metric in self.metrics:
                    result_dict[str(metric)] = metric.perform(predictions, test)

                prediction_metric_results.append(result_dict, ignore_index=True)

        prediction_metric_results = prediction_metric_results.groupby('from').mean().reset_index()

        return prediction_metric_results


class ReportEvalModel(EvalModel):
    """
    Class for automating the process of recommending
    and evaluate produced recommendations.
    This subclass automate the computation of metrics
    whose input is the result of a RecSys
    configured with a ranking algorithm.
    The recommendation are computed for each user and
    the metrics are computed only after the whole
    recommendation process, on the entire frame

    Args:
        config (RecSysConfig): Configuration of the
        recommender system that will be internally created
        metric_list (list<Metric>): List of metrics that eval model will compute
    """
    def __init__(self, config, recs_number: int, metric_list: List[Metric] = None):
        super().__init__(config, None, metric_list)
        self.__recs_number = recs_number

    def fit(self):
        """
        This method performs the rating prediction evaluation by initializing internally
            a recommender system that produces recommendations for all the
            users in the directory specified in the configuration phase.


        Returns:
            result_list: each element of this list is a metric
                result that can be of different types,
                according to the metric, for example a DataFrame or a float
        """
        # initialize recommender to call for prediction computing
        recsys = RecSys(self.config)

        # get all users in specified directory
        logger.info("Loading user instances")
        user_id_list = [
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.config.users_directory)]

        # define results structure
        no_truth_metrics_results = []

        # calculate metrics that not require ground truth
        # for example fairness metrics, serendipity, novelty

        if isinstance(self.config.score_prediction_algorithm, ScorePredictionAlgorithm):
            raise ValueError("You must set ranking algorithm to compute this metrics")

        columns = ["from_id", "to_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)
        for user_id in user_id_list:
            logger.info("User %s", user_id)
            fit_result = recsys.fit_ranking(user_id, self.__recs_number)

            fit_result_with_user = pd.DataFrame(columns=columns)
            fit_result.columns = ["to_id", "rating"]
            for i, row in fit_result.iterrows():
                fit_result_with_user = pd.concat([fit_result_with_user, pd.DataFrame.from_records(
                    [(user_id, row["to_id"], row["rating"])], columns=columns)], ignore_index=True)

            score_frame = pd.concat([fit_result_with_user, score_frame], ignore_index=True)

        logger.info("Computing no truth metrics")
        for metric in self.metrics:
            no_truth_metrics_results.append(
                metric.perform(score_frame, self.config.rating_frame))

        return no_truth_metrics_results
