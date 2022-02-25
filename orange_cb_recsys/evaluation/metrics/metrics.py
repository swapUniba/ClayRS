from abc import ABC, abstractmethod

from orange_cb_recsys.recsys import Split


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
