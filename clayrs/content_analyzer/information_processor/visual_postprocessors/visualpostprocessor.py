import inspect
from abc import ABC, abstractmethod

import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.cluster.vq import vq
from scipy.sparse import csc_matrix, vstack
from typing import List, Any, Union
import numpy as np

from clayrs.utils.automatic_methods import autorepr

from clayrs.content_analyzer.content_representation.content import FieldRepresentation, EmbeddingField, FeaturesBagField


class PostProcessor(ABC):
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


class EmbeddingInputPostProcessor(PostProcessor):

    @abstractmethod
    def process(self, field_repr_list: List[EmbeddingField]) -> List[FieldRepresentation]:
        raise NotImplementedError


class VisualBagOfFeatures(EmbeddingInputPostProcessor):

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto"):
        self.clustering_algorithm = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                                           random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[EmbeddingField]) -> List[FeaturesBagField]:
        first_inst = field_repr_list[0].value
        if len(first_inst.shape) == 2:
            descriptors = [feature for field_repr in field_repr_list for feature in field_repr.value]
            self.clustering_algorithm.fit(np.vstack(descriptors))
            codewords = self.clustering_algorithm.cluster_centers_
            new_field_repr_list = []
            indptr = [0]
            indices = []
            data = []
            for field_repr in field_repr_list:
                code, _ = vq(field_repr.value, codewords)
                for codeword in code:
                    index = codeword
                    indices.append(index)
                    data.append(1)
                indptr.append(len(indices))
            sparse_repr = scipy.sparse.csr_matrix((data, indices, indptr))
            sparse_repr = self.apply_weights(sparse_repr)
            for single_repr in sparse_repr:
                new_field_repr_list.append(
                    FeaturesBagField(single_repr.tocsc(),
                                     [(index, str(codewords[index])) for index in np.unique(single_repr.nonzero()[1])]))
            return new_field_repr_list
        else:
            raise Exception('Unsupported dimensionality')

    @abstractmethod
    def apply_weights(self, sparse_matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        raise NotImplementedError


class CountVisualBagOfFeatures(VisualBagOfFeatures):

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto"):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_weights(self, sparse_repr: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        return sparse_repr

    def __str__(self):
        return "CountVisualBagOfFeatures"

    def __repr__(self):
        return self._repr_string


class TfIdfVisualBagOfFeatures(VisualBagOfFeatures):

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto"):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_weights(self, sparse_repr: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        return TfidfTransformer().fit_transform(sparse_repr)

    def __str__(self):
        return "TfIdfVisualBagOfFeatures"

    def __repr__(self):
        return self._repr_string


class ScipyVQ(EmbeddingInputPostProcessor):

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto"):
        self.k_means = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                              random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[EmbeddingField]) -> List[EmbeddingField]:
        first_inst = field_repr_list[0].value
        if len(first_inst.shape) == 1:
            features = [feature.value for feature in field_repr_list]
        elif len(first_inst.shape) == 2:
            features = [feature for field_repr in field_repr_list for feature in field_repr.value]
        else:
            raise Exception('Unsupported granularity')
        self.k_means.fit(np.vstack(features))
        codewords = self.k_means.cluster_centers_
        new_field_repr_list = []
        if len(first_inst.shape) == 1:
            for field_repr in field_repr_list:
                code, _ = vq(np.matrix(field_repr.value), codewords)
                new_field_repr_list.append(EmbeddingField(codewords[code]))
        else:
            for field_repr in field_repr_list:
                code, _ = vq(field_repr.value, codewords)
                new_field_repr_list.append(EmbeddingField(codewords[code]))
        return new_field_repr_list

    def __str__(self):
        return "ScipyVQ"

    def __repr__(self):
        return self._repr_string


class DimensionalityReduction(EmbeddingInputPostProcessor):

    @staticmethod
    def vstack(field_values_repr_list: Union[List[csc_matrix], List[np.ndarray]]) -> np.ndarray:
        if isinstance(field_values_repr_list[0], csc_matrix):
            return vstack(field_values_repr_list).A
        else:
            return np.vstack(field_values_repr_list)

    def process(self, field_repr_list: Union[List[EmbeddingField], List[FeaturesBagField]]) -> List[EmbeddingField]:
        first_inst = field_repr_list[0].value
        # single array containing the whole embedding (document embedding, for example)
        # or 1 dimensional sparse array
        if len(first_inst.shape) == 1 or (len(first_inst.shape) == 2 and first_inst.shape[0] == 1):
            processed_arrays = self.apply_processing(self.vstack([field.value for field in field_repr_list]))
            return [EmbeddingField(array) for array in processed_arrays]
        # array of word embeddings (330 word embeddings with 100 as dimensionality, for example)
        # this code applies the post-processing to each single word embedding
        elif len(first_inst.shape) == 2:
            values = [field.value for field in field_repr_list]
            processed_arrays = self.apply_processing(self.vstack(values))
            next_instance_index = 0
            new_field_repr_list = []
            for embedding in values:
                num_of_elements = len(embedding)
                new_embedding_repr = processed_arrays[next_instance_index:num_of_elements+next_instance_index]
                next_instance_index = num_of_elements+next_instance_index
                new_field_repr_list.append(EmbeddingField(new_embedding_repr))
            return new_field_repr_list
        # other cases (?)
        else:
            raise Exception('Unsupported dimensionality')

    @abstractmethod
    def apply_processing(self, field_repr_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SkLearnPCA(DimensionalityReduction):

    def __init__(self, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        super().__init__()
        self.pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol,
                       iterated_power=iterated_power, random_state=random_state)
        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_processing(self, field_repr_array: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(field_repr_array)

    def __str__(self):
        return 'SkLearnPCA'

    def __repr__(self):
        return self._repr_string


class SkLearnRandomProjections(DimensionalityReduction):

    def __init__(self, n_components='auto', eps=0.1):
        super().__init__()
        self.random_proj = GaussianRandomProjection(n_components=n_components, eps=eps)
        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_processing(self, field_repr_array: np.ndarray) -> np.ndarray:
        return self.random_proj.fit_transform(field_repr_array)

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

    def apply_processing(self, field_repr_array: np.ndarray) -> np.ndarray:
        return self.feature_agg.fit_transform(field_repr_array)

    def __str__(self):
        return 'SkLearnFeatureAgglomeration'

    def __repr__(self):
        return self._repr_string

