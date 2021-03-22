from sklearn.feature_extraction.text import TfidfVectorizer

from orange_cb_recsys.content_analyzer.content_representation.content_field import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.\
    field_content_production_technique import TfIdfTechnique
from orange_cb_recsys.content_analyzer.memory_interfaces.text_interface import IndexInterface
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

    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: list):
        """
        Creates a corpus structure, a list of string where each string is a document.
        Then call TfIdfVectorizer this collection, obtaining term-document
        tf-idf matrix, the corpus is then deleted

        Args:
            information_source (RawInformationSource): Source for the raw data
            id_field_names: names of the fields that compounds the id
        """

        field_name = self.field_need_refactor
        preprocessor_list = self.processor_list

        for raw_content in information_source:
            processed_field_data = raw_content[field_name]
            for preprocessor in preprocessor_list:
                processed_field_data = preprocessor.process(processed_field_data)

            processed_field_data = check_not_tokenized(processed_field_data)
            content_id = id_merger(raw_content, id_field_names)
            self.__matching[content_id] = len(self.__corpus)
            self.__corpus.append(processed_field_data)

        tf_vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.__tfidf_matrix = tf_vectorizer.fit_transform(self.__corpus)

        del self.__corpus

        self.__feature_names = tf_vectorizer.get_feature_names()

    def produce_content(self, field_representation_name: str, content_id: str, field_name: str):
        """
        Retrieve the tf-idf values, for terms in document that match with content_id,
        from the pre-computed word - document matrix.

        Args:
            field_representation_name (str): Name of the field representation
            content_id (str): Id of the content that contains the terms for which extract the tf-idf
            field_name (str): Name of the field to consider

        Returns:
            (FeaturesBag): <term, tf-idf>
        """

        doc = self.__matching[content_id]
        feature_index = self.__tfidf_matrix[doc, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [self.__tfidf_matrix[doc, x] for x in feature_index])

        features = {}
        for word, score in [(self.__feature_names[i], score) for (i, score) in tfidf_scores]:
            features[word] = score

        return FeaturesBagField(field_representation_name, features)

    def delete_refactored(self):
        pass


class WhooshTfIdf(TfIdfTechnique):
    """
    Class that produces a Bag of words with tf-idf metric using Whoosh
    """

    def __init__(self):
        super().__init__()
        self.__index = IndexInterface('./frequency-index')

    def __str__(self):
        return "WhooshTfIdf"

    def __repr__(self):
        return "< WhooshTfIdf: " + "index = " + str(self.__index) + ">"

    def produce_content(self, field_representation_name: str, content_id: str,
                        field_name: str) -> FeaturesBagField:

        return FeaturesBagField(
            field_representation_name, self.__index.get_tf_idf(field_name, content_id))

    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: list):
        """
        Saves the processed data in a index that will be used for frequency calculation

        Args:
            information_source (RawInformationSource): data source from
                which extract the field data
                to create the index for tf-idf computing
            id_field_names (list<str>): names of the fields that compound the id
        """

        field_name = self.field_need_refactor
        preprocessor_list = self.processor_list
        pipeline_id = self.pipeline_need_refactor

        self.__index = IndexInterface('./' + field_name + pipeline_id)
        self.__index.init_writing()
        for raw_content in information_source:
            self.__index.new_content()
            content_id = id_merger(raw_content, id_field_names)
            self.__index.new_field("content_id", content_id)
            processed_field_data = raw_content[field_name]
            for preprocessor in preprocessor_list:
                processed_field_data = preprocessor.process(processed_field_data)

            processed_field_data = check_tokenized(processed_field_data)
            self.__index.new_field(field_name, processed_field_data)
            self.__index.serialize_content()

        self.__index.stop_writing()

    def delete_refactored(self):
        """
        Delete the index used for term vectors and relative frequencies
        """
        self.__index.delete_index()
