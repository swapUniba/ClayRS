from abc import ABC, abstractmethod


class GraphMetrics(ABC):
    """
    Interface for graph metrics
    """

    @abstractmethod
    def degree_centrality(self):
        raise NotImplementedError

    @abstractmethod
    def closeness_centrality(self):
        raise NotImplementedError

    @abstractmethod
    def dispersion(self):
        raise NotImplementedError
