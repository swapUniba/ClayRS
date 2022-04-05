import abc
from abc import ABC
from copy import deepcopy
from itertools import chain


class Algorithm(ABC):
    """
    Abstract class for an Algorithm.

    Every algorithm must be able to predict or to rank, or maybe both.
    In case some algorithms can only do one of the two (eg. PageRank), simply implement both
    methods and raise the NotPredictionAlg or NotRankingAlg exception accordingly.
    """
    __slots__ = ()

    @abc.abstractmethod
    def predict(self, **kwargs):
        """
        Method to call when score prediction must be done.

        If the Algorithm can't do score prediction, implement this method and raise
        the NotPredictionAlg exception
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, **kwargs):
        """
        Method to call when ranking must be done.

        If the Algorithm can't rank, implement this method and raise the NotRankingAlg exception
        """
        raise NotImplementedError

    def __deepcopy__(self, memo):
        # Create a new instance
        cls = self.__class__
        result = cls.__new__(cls)

        # Don't copy self reference
        memo[id(self)] = result

        # Don't copy the cache - if it exists
        if hasattr(self, "_cache"):
            memo[id(self._cache)] = self._cache.__new__(dict)

        # Get all __slots__ of the derived class
        slots = chain.from_iterable(getattr(s, '__slots__', []) for s in self.__class__.__mro__)

        # Deep copy all other attributes
        for var in slots:
            setattr(result, var, deepcopy(getattr(self, var), memo))

        # Return updated instance
        return result