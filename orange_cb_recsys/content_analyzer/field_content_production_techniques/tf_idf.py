from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

from orange_cb_recsys.content_analyzer.content_representation.content import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.\
    field_content_production_technique import TfIdfTechnique
from orange_cb_recsys.content_analyzer.information_processor import InformationProcessor
from orange_cb_recsys.content_analyzer.memory_interfaces.text_interface import KeywordIndex
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.check_tokenization import check_tokenized, check_not_tokenized
from orange_cb_recsys.utils.id_merger import id_merger


class SkLearnTfIdf(TfIdfTechnique):
    """
    Tf-idf computed using the sklearn library
    """
    def __init__(self):
        super().__init__()
        self.__corpus = []
        self.__tfidf_matrix = None
        self.__feature_names = None
        self.__matching = {}

    def produce_single_repr(self, content_id: str) -> FeaturesBagField:
        """
        Retrieves the tf-idf values, for terms in document that match with content_id,
        from the pre-computed word - document matrix.
        """
        doc = self.__matching[content_id]
        feature_index = self.__tfidf_matrix[doc, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [self.__tfidf_matrix[doc, x] for x in feature_index])

        features = {}
        for word, score in [(self.__feature_names[i], score) for (i, score) in tfidf_scores]:
            features[word] = score

        return FeaturesBagField(features)

    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: list,
                         field_name: str, preprocessor_list: List[InformationProcessor]):
        """
        Creates a corpus structure, a list of string where each string is a document.
        Then calls TfIdfVectorizer on this collection, obtaining term-document tf-idf matrix, the corpus is then deleted
        """
        self.__corpus = []
        self.__matching = {}

        for raw_content in information_source:
            processed_field_data = self.process_data(raw_content[field_name], preprocessor_list)

            processed_field_data = check_not_tokenized(processed_field_data)
            content_id = id_merger(raw_content, id_field_names)
            self.__matching[content_id] = len(self.__corpus)
            self.__corpus.append(processed_field_data)

        tf_vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.__tfidf_matrix = tf_vectorizer.fit_transform(self.__corpus)

        del self.__corpus

        self.__feature_names = tf_vectorizer.get_feature_names()

    def delete_refactored(self):
        del self.__tfidf_matrix
        del self.__feature_names
        del self.__matching

    def __str__(self):
        return "SkLearnTfIdf"

    def __repr__(self):
        return "< SkLearnTfIdf >"


class WhooshTfIdf(TfIdfTechnique):
    """
    Class that produces a Bag of words with tf-idf metric using Whoosh
    """

    def __init__(self):
        super().__init__()
        self.__index = KeywordIndex('./frequency-index')
        self.__field_name = None

    def produce_single_repr(self, content_id: str) -> FeaturesBagField:
        """
        Retrieves the tf-idf value directly from the index
        """
        return FeaturesBagField(self.__index.get_tf_idf(self.__field_name, content_id))

    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: list,
                         field_name: str, preprocessor_list: List[InformationProcessor]):
        """
        Saves the processed data in a index that will be used for frequency calculation
        """
        self.__field_name = field_name
        self.__index = KeywordIndex('./' + field_name)
        self.__index.init_writing()
        for raw_content in information_source:
            self.__index.new_content()
            content_id = id_merger(raw_content, id_field_names)
            self.__index.new_field("content_id", content_id)
            processed_field_data = self.process_data(raw_content[field_name], preprocessor_list)

            processed_field_data = check_tokenized(processed_field_data)
            self.__index.new_field(field_name, processed_field_data)
            self.__index.serialize_content()

        self.__index.stop_writing()

    def delete_refactored(self):
        self.__index.delete()

    def __str__(self):
        return "WhooshTfIdf"

    def __repr__(self):
        return "< WhooshTfIdf >"
