import inspect
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Any
import numpy as np

from clayrs.utils.automatic_methods import autorepr

from clayrs.content_analyzer.content_representation.content import FieldRepresentation, EmbeddingField


class VisualPostProcessor(ABC):
    """
    Abstract class that generalizes data post-processing
    """

    @abstractmethod
    def process(self, field_repr_list: List[FieldRepresentation]) -> List[FieldRepresentation]:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class DimensionalityReduction(VisualPostProcessor):

    @abstractmethod
    def process(self, field_repr_list: List[EmbeddingField]) -> List[EmbeddingField]:
        raise NotImplementedError


class SkLearnKMeans(VisualPostProcessor):

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto"):
        self.k_means = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                              random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[FieldRepresentation]) -> List[FieldRepresentation]:
        return self.k_means.fit_transform(np.stack(field_repr_list, axis=0))

    def __str__(self):
        return "SkLearnKMeans"

    def __repr__(self):
        return self._repr_string


class SkLearnPCA(DimensionalityReduction):

    def __init__(self, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        super().__init__()
        self.pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol,
                       iterated_power=iterated_power, random_state=random_state)
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[EmbeddingField]) -> List[EmbeddingField]:
        return self.pca.fit_transform(np.stack([e.value for e in field_repr_list], axis=0))

    def __str__(self):
        return 'SkLearnPca'

    def __repr__(self):
        return self._repr_string
