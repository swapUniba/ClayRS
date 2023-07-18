from __future__ import annotations
from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from clayrs.recsys.partitioning import Split


class Metric(ABC):
    """
    Abstract class that generalize metric concept

    Every metric may need different kind of "prediction": some (eg. NDCG, MRR, etc.) may need recommendation lists in
    which the recsys ranks every unseen item, some (eg. MAE, RMSE, etc.) may need a score prediction where the recsys
    must predict the rating that a user would give to an unseen item.
    So a Metric category (subclass of this class) must implement the "eval_fit_recsys(...)" specifying its needs,
    while every single metric (subclasses of the metric category class) must implement the "perform(...)" method
    specifying how to execute the metric computation
    """

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def perform(self, split: Split):
        raise NotImplementedError


def handler_different_users(func):
    """
    Handler that covers the case in which there are different users between the predictions and the truth of a split: in
    that case a ValueError exception is raised
    """
    @wraps(func)
    def inner_function(self, split, *args, **kwargs):

        if not np.array_equal(np.sort(split.pred.unique_user_id_column.flat),
                              np.sort(split.truth.unique_user_id_column.flat)):
            raise ValueError("Predictions and truths must contain the same users!")

        return func(self, split, *args, **kwargs)

    return inner_function
