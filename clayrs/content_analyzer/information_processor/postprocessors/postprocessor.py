import inspect
from abc import ABC, abstractmethod

import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import vq
from scipy.sparse import csc_matrix, vstack
from typing import List, Any, Union, Optional
import numpy as np
import pandas as pd

from clayrs.utils.automatic_methods import autorepr

from clayrs.content_analyzer.content_representation.content import FieldRepresentation, EmbeddingField, FeaturesBagField


class PostProcessor(ABC):
    """
    Abstract class that generalizes data post-processing

    The class is then extended by specifying the input representation types.
    So, for example, if the post-processing technique takes embeddings as input it will be extended as
    EmbeddingPostProcessor
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
    """
    Abstract class to represent post-processors which take an embedding as input
    """

    @abstractmethod
    def process(self, field_repr_list: List[EmbeddingField]) -> List[FieldRepresentation]:
        raise NotImplementedError


class EmbeddingFeaturesInputPostProcessor(PostProcessor):
    """
    Abstract class to represent post-processors which take an embedding or a features bag as input
    """

    @abstractmethod
    def process(self, field_repr_list: Union[List[FeaturesBagField], List[EmbeddingField]]) -> List[FieldRepresentation]:
        raise NotImplementedError


class VisualBagOfWords(EmbeddingInputPostProcessor):
    """
    Technique which mirrors the Bag Of Words approach in NLP but extends it to images.
    The idea is to represent the target image as a collection of features.

    To do so, the algorithm performs the following steps sequentially:

        - Codebook construction: builds the vocabulary from which features will be retrieved for the images, this is
            done by applying k-means clustering on the input feature vectors;

        - Vector quantization: given the features for an image, find the ones that are closest to them in the
            vocabulary produced at the previous step and apply a weighting schema to produce the final representation
            (that is, for example, a particular visual word appears two times in an image, if using a "count" weighting
            schema the final representation will have a value of 2 associated to this visual word)

    If you want to know more about the approach or have more details, the following tutorial is suggested:
    https://customers.pyimagesearch.com/the-bag-of-visual-words-model/

    Arguments for [SkLearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

    Arguments for [SkLearn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

    NOTE: for this technique it is mandatory for the parameter "with_std" to be set to True
    """

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto",
                 with_mean: bool = True, with_std: bool = True):
        self.clustering_algorithm = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                                           random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.with_mean = with_mean
        self.with_std = with_std
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[EmbeddingField]) -> List[FeaturesBagField]:
        first_inst = field_repr_list[0].value

        # extract features from the representations and apply clustering to them
        if len(first_inst.shape) == 2:
            descriptors = np.vstack([feature for field_repr in field_repr_list for feature in field_repr.value])
        else:
            raise ValueError(f'Unsupported dimensionality for technique {self}, '
                             f'only two dimensional arrays are supported')

        scaler = None

        if self.with_mean or self.with_std:
            scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
            descriptors = scaler.fit_transform(descriptors)

        self.clustering_algorithm.fit(descriptors)
        codewords = self.clustering_algorithm.cluster_centers_

        new_field_repr_list = []
        indptr = [0]
        indices = []
        data = []

        # apply vector quantization to each representation w.r.t. the codewords
        # dictionary created at the previous step
        for field_repr in field_repr_list:

            if scaler is not None:
                code, _ = vq(scaler.transform(field_repr.value), codewords)
            else:
                code, _ = vq(field_repr.value, codewords)

            for codeword in code:
                index = codeword
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

        # instantiate the visual bag of features as sparse matrix and apply a weighting schema to it
        sparse_repr = scipy.sparse.csr_matrix((data, indices, indptr))
        sparse_repr = self.apply_weights(sparse_repr)
        for single_repr in sparse_repr:
            new_field_repr_list.append(
                FeaturesBagField(single_repr.tocsc(),
                                 [(index, str(codewords[index])) for index in pd.unique(single_repr.nonzero()[1])]))
        return new_field_repr_list

    @abstractmethod
    def apply_weights(self, sparse_matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Apply a weighting schema to the representations obtained from the vector quantization step

        Args:
            sparse_matrix: scipy sparse csr matrix containing the count of occurrences of each visual word
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class CountVisualBagOfWords(VisualBagOfWords):
    """
    Class which implements the count weighting schema, which means that the final representation will contain counts
    of each visual word appearing from the codebook

    Example:

        codebook = [[0.6, 1.7, 0.3],
                    [0.2, 0.7, 1.8]]

        repr = [[0.6, 1.7, 0.3],
                [0.6, 1.7, 0.3]]

        output of weighting schema = [2, 0]

    NOTE: a preliminary step is necessary, that is feature extraction. You should do that using one of the provided
    visual techniques and setting this as postprocessor for the output of that technique as follows:

    ```python
    import clayrs.content_analyzer as ca
    ca.FieldConfig(ca.SkImageCannyEdgeDetector(), postprocessing=[ca.CountVisualBagOfWords()])
    ```

    ADDITIONAL NOTE: the technique requires 2D arrays of features for each image, such as edges in the case of the
    Canny Edge detector. In case any other dimensionality is provided, a ValueError will be raised.

    Arguments for [SkLearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

    Arguments for [SkLearn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

    NOTE: for this technique it is mandatory for the parameter "with_std" to be set to True
    """

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto",
                 with_mean: bool = True, with_std: bool = True):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         random_state=random_state, copy_x=copy_x, algorithm=algorithm, with_mean=with_mean,
                         with_std=with_std)
        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_weights(self, sparse_matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Apply a count wight schema to the representations obtained from the vector quantization step

        Args:
            sparse_matrix: scipy sparse csr matrix containing the count of occurrences of each visual word
        """
        return sparse_matrix

    def __str__(self):
        return "Count Visual Bag of Words"

    def __repr__(self):
        return self._repr_string


class TfIdfVisualBagOfWords(VisualBagOfWords):
    """
    Class which implements the tf-idf weighting schema, which means that the final representation will contain tf-idf
    scores of each visual word appearing from the codebook

    Example:

        codebook = [[0.6, 1.7, 0.3],
                    [0.2, 0.7, 1.8]]

        repr1 = [[0.6, 1.7, 0.3],
                 [0.6, 1.7, 0.3]]

        repr2 = [[0.6, 1.7, 0.3],
                 [0.2, 0.7, 1.8]]

        output of weighting schema = [[2, 0], [1, 1.69]]

    NOTE: a preliminary step is necessary, that is feature extraction. You should do that using one of the provided
    visual techniques and setting this as postprocessor for the output of that technique as follows:

    ```python
    import clayrs.content_analyzer as ca
    ca.FieldConfig(ca.SkImageCannyEdgeDetector(), postprocessing=[ca.TfIdfVisualBagOfWords()])
    ```

    ADDITIONAL NOTE: the technique requires 2D arrays of features for each image, such as edges in the case of the
    Canny Edge detector. In case any other dimensionality is provided, a ValueError will be raised.

    Arguments for [SkLearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

    Arguments for [SkLearn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

    Arguments for [SkLearn TfIdf Transformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

    NOTE: for this technique it is mandatory for the parameter "with_std" to be set to True
    """

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto",
                 with_mean: bool = True, with_std: bool = True,
                 norm: Optional[str] = "l2", use_idf: bool = True,
                 smooth_idf: bool = True, sublinear_tf: bool = False):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         random_state=random_state, copy_x=copy_x, algorithm=algorithm, with_mean=with_mean,
                         with_std=with_std)

        self.tf_idf_params = {"norm": norm,
                              "use_idf": use_idf,
                              "smooth_idf": smooth_idf,
                              "sublinear_tf": sublinear_tf}

        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_weights(self, sparse_matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Apply a tf-idf weighting schema to the representations obtained from the vector quantization step

        Args:
            sparse_matrix: scipy sparse csr matrix containing the count of occurrences of each visual word
        """
        return TfidfTransformer(**self.tf_idf_params).fit_transform(sparse_matrix.todok())

    def __str__(self):
        return "Tf-Idf Visual Bag of Features"

    def __repr__(self):
        return self._repr_string


class ScipyVQ(EmbeddingInputPostProcessor):
    """
    Vector quantization using Scipy implementation and SkLearn KMeans.
    The idea behind this technique is to "approximate" feature vectors, using only a finite set of prototype vectors
    from a codebook.
    The codebook is computed using the SkLearn KMeans implementation. After that, for each feature in the
    representation, the closest one from the codebook is found using the Vector Quantization implementation from
    scipy and the retrieved vector is replaced to the original one in the final representation.

    ```python
    import clayrs.content_analyzer as ca
    ca.FieldConfig(ca.SkImageCannyEdgeDetector(), postprocessing=[ca.ScipyVQ()])
    ```

    Arguments for [SkLearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

    Arguments for [SkLearn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

    NOTE: for this technique it is mandatory for the parameter "with_std" to be set to True
    """

    def __init__(self, n_clusters: Any = 8, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto",
                 with_mean: bool = True, with_std: bool = True):
        self.k_means = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                              random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.with_mean = with_mean
        self.with_std = with_std
        self._repr_string = autorepr(self, inspect.currentframe())

    def process(self, field_repr_list: List[EmbeddingField]) -> List[EmbeddingField]:
        first_inst = field_repr_list[0].value

        # stack the representations from all fields

        if len(first_inst.shape) == 2:
            descriptors = np.vstack([feature for field_repr in field_repr_list for feature in field_repr.value])
        elif len(first_inst.shape) == 1:
            descriptors = np.vstack([field_repr.value for field_repr in field_repr_list])
        else:
            raise ValueError(f'Unsupported dimensionality for technique {self}, '
                             f'only one and two dimensional arrays are supported')

        scaler = None

        if self.with_mean or self.with_std:
            scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
            descriptors = scaler.fit_transform(descriptors)

        # learn clusters using kmeans
        self.k_means.fit(descriptors)
        codewords = self.k_means.cluster_centers_
        new_field_repr_list = []

        # replace the old representations with the new ones (which will be the most similar codeword from the
        # codebook)
        if len(first_inst.shape) == 1:
            for field_repr in field_repr_list:

                if scaler is not None:
                    code, _ = vq(scaler.transform(np.expand_dims(field_repr.value, axis=0)), codewords)
                else:
                    code, _ = vq(np.expand_dims(field_repr.value, axis=0), codewords)

                new_field_repr_list.append(EmbeddingField(codewords[code]))
        else:
            for field_repr in field_repr_list:

                if scaler is not None:
                    code, _ = vq(scaler.transform(field_repr.value), codewords)
                else:
                    code, _ = vq(field_repr.value, codewords)

                new_field_repr_list.append(EmbeddingField(codewords[code]))
        return new_field_repr_list

    def __str__(self):
        return "Scipy Vector Quantization"

    def __repr__(self):
        return self._repr_string


class DimensionalityReduction(EmbeddingFeaturesInputPostProcessor):
    """
    Abstract class that encapsulates the logic for dimensionality reduction techniques.
    It contains the methods to manage two different kinds of embedding representations:

        - All embedding representations are 1 dimensional: in this case the embeddings from all items are vertically
            stacked into a matrix and, after reducing the dimensions of its rows, each row is returned separately

        - All embedding representations are 2 dimensional: in this case, each field representation is vertically
            stacked into a matrix and, after reducing the dimensions of its rows, the 2 dimensional arrays are
            re-created from the processed matrix

    NOTE: embeddings of dimensionality greater than 2 are not supported

    Extending this class and implementing the "apply_processing" method allows to implement a new dimensionality
    reduction technique
    """

    @staticmethod
    def vstack(field_values_repr_list: Union[List[csc_matrix], List[np.ndarray]]) -> np.ndarray:
        """method for vertically stacking"""
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
            raise ValueError(f'Unsupported dimensionality for technique {self}, '
                             f'only one dimensional and two dimensional arrays are supported')

    @abstractmethod
    def apply_processing(self, field_repr_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SkLearnPCA(DimensionalityReduction):
    """
    Dimensionality reduction using the PCA implementation from SkLearn

    Usage example:

    ```python
    import clayrs.content_analyzer as ca
    ca.FieldConfig(ca.SkImageCannyEdgeDetector(), postprocessing=[ca.SkLearnPCA()])
    ```

    Arguments for [SkLearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    """

    def __init__(self, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        super().__init__()
        self.pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol,
                       iterated_power=iterated_power, random_state=random_state)
        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_processing(self, field_repr_array: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(field_repr_array)

    def __str__(self):
        return 'SkLearn PCA'

    def __repr__(self):
        return self._repr_string


class SkLearnGaussianRandomProjections(DimensionalityReduction):
    """
    Dimensionality reduction using the Gaussian Random Projections implementation from SkLearn

    Usage example:

    ```python
    import clayrs.content_analyzer as ca
    ca.FieldConfig(ca.SkImageCannyEdgeDetector(), postprocessing=[ca.SkLearnGaussianRandomProjections()])
    ```

    Arguments for [SkLearn Gaussian Random Projection](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html)
    """

    def __init__(self, n_components='auto', eps=0.1, random_state=None):
        super().__init__()
        self.random_proj = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)
        self._repr_string = autorepr(self, inspect.currentframe())

    def apply_processing(self, field_repr_array: np.ndarray) -> np.ndarray:
        return self.random_proj.fit_transform(field_repr_array)

    def __str__(self):
        return 'SkLearn Random Projections'

    def __repr__(self):
        return self._repr_string


class SkLearnFeatureAgglomeration(DimensionalityReduction):
    """
    Dimensionality reduction using the Feature Agglomeration implementation from SkLearn

    Usage example:

    ```python
    import clayrs.content_analyzer as ca
    ca.FieldConfig(ca.SkImageCannyEdgeDetector(), postprocessing=[ca.SkLearnFeatureAgglomeration()])
    ```

    Arguments for [SkLearn Feature Agglomeration](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html)
    """

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
        return 'SkLearn Feature Agglomeration'

    def __repr__(self):
        return self._repr_string

