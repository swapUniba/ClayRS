import inspect
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.random_projection import GaussianRandomProjection
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
        k_means_arrays = self.k_means.fit_transform(np.stack(field_repr_list, axis=0))
        return [EmbeddingField(k_means_array) for k_means_array in k_means_arrays]

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
        pca_arrays = self.pca.fit_transform(np.stack(field_repr_list, axis=0))
        return [EmbeddingField(pca_array) for pca_array in pca_arrays]

    def __str__(self):
        return 'SkLearnPca'

    def __repr__(self):
        return self._repr_string


class SkLearnRandomProjections(DimensionalityReduction):

    def __init__(self, n_components='auto', eps=0.1):
        super().__init__()
        self.random_proj = GaussianRandomProjection(n_components=n_components, eps=eps)
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[EmbeddingField]) -> List[EmbeddingField]:
        random_proj_arrays = self.random_proj.fit_transform(np.stack(field_repr_list, axis=0))
        return [EmbeddingField(random_proj_array) for random_proj_array in random_proj_arrays]

    def __str__(self):
        return 'SkLearnRandomProjections'

    def __repr__(self):
        return self._repr_string


class SkLearnFeatureAgglomeration(DimensionalityReduction):

    def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto',
                 linkage='ward', pooling_func=np.mean, distance_threshold=None, compute_distances=False):
        super().__init__()
        self.feature_agg = FeatureAgglomeration(n_clusters=n_clusters, affinity=affinity, memory=memory,
                                                connectivity=connectivity, compute_full_tree=compute_full_tree,
                                                linkage=linkage, pooling_func=pooling_func,
                                                distance_threshold=distance_threshold,
                                                compute_distances=compute_distances)
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[EmbeddingField]) -> List[EmbeddingField]:
        feature_agg_arrays = self.feature_agg.fit_transform(np.stack(field_repr_list, axis=0))
        return [EmbeddingField(feature_agg_array) for feature_agg_array in feature_agg_arrays]

    def __str__(self):
        return 'SkLearnFeatureAgglomeration'

    def __repr__(self):
        return self._repr_string

