from __future__ import annotations
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Union, Mapping, Iterable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.content_analyzer.information_processor.information_processor_abstract import InformationProcessor
    from clayrs.content_analyzer.raw_information_source import RawInformationSource

from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique \
    import TfIdfTechnique
from clayrs.content_analyzer.memory_interfaces.text_interface import KeywordIndex
from clayrs.content_analyzer.utils.check_tokenization import check_tokenized, check_not_tokenized
from clayrs.utils.const import logger


class SkLearnTfIdf(TfIdfTechnique):
    """
    Class that produces a sparse vector for each content representing the tf-idf scores of its terms using SkLearn.

    Please refer to [its documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    for more information about how it's computed

    Args:
        max_df:
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float in range [0.0, 1.0], the parameter represents a proportion of
            documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        min_df:
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float in range of [0.0, 1.0], the parameter represents a proportion
            of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        max_features:
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.

            This parameter is ignored if vocabulary is not None.

        vocabulary:
            Either a Mapping (e.g., a dict) where keys are terms and values are
            indices in the feature matrix, or an iterable over terms. If not
            given, a vocabulary is determined from the input documents.

        binary:
            If True, all non-zero term counts are set to 1. This does not mean
            outputs will have only 0/1 values, only that the tf term in tf-idf
            is binary. (Set idf and normalization to False to get 0/1 outputs).

        dtype:
            Precision of the tf-idf scores

        norm:
            Each output row will have unit norm, either:

            - 'l2': Sum of squares of vector elements is 1. The cosine
              similarity between two vectors is their dot product when l2 norm has
              been applied.
            - 'l1': Sum of absolute values of vector elements is 1.
              See :func:`preprocessing.normalize`.

        use_idf:
            Enable inverse-document-frequency reweighting. If False, idf(t) = 1.

        smooth_idf:
            Smooth idf weights by adding one to document frequencies, as if an
            extra document was seen containing every term in the collection
            exactly once. Prevents zero divisions.

        sublinear_tf:
            Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    """
    def __init__(self, max_df: Union[float, int] = 1.0, min_df: Union[float, int] = 1, max_features: int = None,
                 vocabulary: Union[Mapping, Iterable] = None, binary: bool = False, dtype: Callable = np.float64,
                 norm: str = 'l2', use_idf: bool = True, smooth_idf: bool = True, sublinear_tf: bool = False):

        super().__init__()
        self._sk_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features,
                                              vocabulary=vocabulary, binary=binary, dtype=dtype,
                                              norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                                              sublinear_tf=sublinear_tf)

    def dataset_refactor(self, information_source: RawInformationSource, field_name: str,
                         preprocessor_list: List[InformationProcessor]) -> int:
        # Creates a corpus structure, a list of string where each string is a document.
        # Then calls TfIdfVectorizer on this collection, obtaining term-document tf-idf matrix,
        # the corpus is then deleted

        corpus = []
        logger.info(f"Computing tf-idf with {str(self)}")
        for raw_content in information_source:
            processed_field_data = self.process_data(raw_content[field_name], preprocessor_list)

            processed_field_data = check_not_tokenized(processed_field_data)
            corpus.append(processed_field_data)

        self._tfidf_matrix = self._sk_vectorizer.fit_transform(corpus)
        self._feature_names = self._sk_vectorizer.get_feature_names_out()

        return self._tfidf_matrix.shape[0]

    def __str__(self):
        return "SkLearnTfIdf"

    def __repr__(self):
        return f"SkLearnTfIdf(max_df={self._sk_vectorizer.max_df}, min_df={self._sk_vectorizer.min_df}, " \
               f"max_features={self._sk_vectorizer.max_features}, vocabulary={self._sk_vectorizer.vocabulary}, " \
               f"binary={self._sk_vectorizer.binary}, dtype={self._sk_vectorizer.dtype}, " \
               f"norm={self._sk_vectorizer.norm}, use_idf={self._sk_vectorizer.use_idf}, " \
               f"smooth_idf={self._sk_vectorizer.smooth_idf}, sublinear_tf={self._sk_vectorizer.sublinear_tf})"


class WhooshTfIdf(TfIdfTechnique):
    r"""
    Class that produces a sparse vector for each content representing the tf-idf scores of its terms using Whoosh

    The tf-idf computation formula is:

    $$
    tf \mbox{-} idf = (1 + log10(tf)) * log10(idf)
    $$

    """

    def __init__(self):
        super().__init__()

    def dataset_refactor(self, information_source: RawInformationSource, field_name: str,
                         preprocessor_list: List[InformationProcessor]):
        # Saves the processed data in a index that will be used for frequency calculation

        logger.info(f"Computing tf-idf with {str(self)}")
        index = KeywordIndex(f'./tf_idf_{field_name}')
        index.init_writing(True)
        dataset_len = 0
        for raw_content in information_source:
            index.new_content()
            processed_field_data = self.process_data(raw_content[field_name], preprocessor_list)

            processed_field_data = check_tokenized(processed_field_data)
            index.new_field(field_name, processed_field_data)
            index.serialize_content()
            dataset_len += 1

        index.stop_writing()

        tfidf_dicts = [index.get_tf_idf(field_name, i) for i in range(dataset_len)]
        index.delete()

        vectorizer = DictVectorizer(sparse=True)
        self._tfidf_matrix = vectorizer.fit_transform(tfidf_dicts)
        self._feature_names = vectorizer.get_feature_names_out()

        return dataset_len

    def __str__(self):
        return "WhooshTfIdf"

    def __repr__(self):
        return "WhooshTfIdf()"
