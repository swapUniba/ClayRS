from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split

from orange_cb_recsys.evaluation.exceptions import AlreadyFittedRecSys
from orange_cb_recsys.recsys.recsys import RecSys


class Metric(ABC):
    """
    Abstract class that generalize metric concept;
    """

    @classmethod
    @abstractmethod
    def eval_fit_recsys(cls, recsys: RecSys, split_list: List[Split], test_items_list: List[pd.DataFrame]):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_pred_truth_list(cls) -> List[Split]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _clean_pred_truth_list(cls) -> List[Split]:
        raise NotImplementedError

    @abstractmethod
    def perform(self, split: Split):
        """
        Method that execute the metric computation

        Args:
              truth (pd.DataFrame): dataframe with known ratings,
                  it is used as ground truth in metric computation
                  predictions (pd.DataFrame): dataframe with predicted items and
                  associated scores
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class RankingNeededMetric(Metric):
    rank_truth_list = []

    @classmethod
    def eval_fit_recsys(cls, recsys: RecSys, split_list: List[Split], test_items_list: List[pd.DataFrame]):
        if len(cls.rank_truth_list) != 0:
            raise AlreadyFittedRecSys

        for split, test_items_frame in zip(split_list, test_items_list):

            train = split.train
            test = split.test

            rank_truth = Split()
            rank_truth.truth = test

            frame_to_concat = []
            user_list_to_fit = set(train.from_id)

            for user in user_list_to_fit:

                user_ratings_train = train.loc[train['from_id'] == user]

                test_items = list(test_items_frame.query('from_id == @user').to_id)

                result = recsys._eval_fit_rank(user_ratings_train, test_items)

                # if len(result) != 0:
                frame_to_concat.append(result)
                # else:
                #     nan_result = pd.DataFrame({'from_id': [], 'to_id': [], 'score': []})
                #
                #     nan_result.from_id = user_ratings_test.from_id
                #     nan_result.to_id = user_ratings_test.to_id
                #     nan_result.score = [np.nan for i in range(len(user_ratings_test))]
                #
                #     frame_to_concat.append(nan_result)

            rank_truth.pred = pd.concat(frame_to_concat)

            cls.rank_truth_list.append(rank_truth)

    @classmethod
    def _get_pred_truth_list(cls):
        return cls.rank_truth_list

    @classmethod
    def _clean_pred_truth_list(cls):
        RankingNeededMetric.rank_truth_list = []


class ScoresNeededMetric(Metric):
    score_truth_list = []

    @classmethod
    def eval_fit_recsys(cls, recsys: RecSys, split_list: List[Split], test_items_list: List[pd.DataFrame]):
        if len(cls.score_truth_list) != 0:
            raise AlreadyFittedRecSys

        for split, test_items_frame in zip(split_list, test_items_list):

            train = split.train
            test = split.test

            score_truth = Split()
            score_truth.truth = test

            frame_to_concat = []
            user_list_to_fit = set(train.from_id)

            for user in user_list_to_fit:

                user_ratings_train = train.loc[train['from_id'] == user]

                test_items = test_items_frame.query('from_id == @user').to_id

                result = recsys._eval_fit_predict(user_ratings_train, test_items)

                # if len(result) != 0:
                frame_to_concat.append(result)
                # else:
                #     nan_result = pd.DataFrame({'from_id': [], 'to_id': [], 'score': []})
                #
                #     nan_result.from_id = user_ratings_test.from_id
                #     nan_result.to_id = user_ratings_test.to_id
                #     nan_result.score = [np.nan for i in range(len(user_ratings_test))]
                #
                #     frame_to_concat.append(nan_result)

            score_truth.pred = pd.concat(frame_to_concat)

            cls.score_truth_list.append(score_truth)

    @classmethod
    def _get_pred_truth_list(cls):
        return cls.score_truth_list

    @classmethod
    def _clean_pred_truth_list(cls):
        ScoresNeededMetric.score_truth_list = []
