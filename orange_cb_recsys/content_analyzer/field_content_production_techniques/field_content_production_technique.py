from abc import ABC, abstractmethod
from typing import List

import nltk
import numpy as np

from nltk.tokenize import sent_tokenize
from orange_cb_recsys.content_analyzer.content_representation. \
    content_field import FieldRepresentation, FeaturesBagField, EmbeddingField
from orange_cb_recsys.content_analyzer.information_processor. \
    information_processor import InformationProcessor
from orange_cb_recsys.content_analyzer.memory_interfaces.text_interface import IndexInterface
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.check_tokenization import check_tokenized, check_not_tokenized


class FieldContentProductionTechnique(ABC):
    """
    Abstract class that generalizes the techniques to use for producing the semantic description
    of a content's field's representation
    """

    def __init__(self):
        self.__lang = "EN"

    @property
    def lang(self):
        return self.__lang

    @lang.setter
    def lang(self, lang: str):
        self.__lang = lang


class SearchIndexing(FieldContentProductionTechnique):
    def produce_content(self, field_name: str, pipeline_id, field_data, indexer: IndexInterface):
        """
        Save field data as a document field using the given indexer,
        the resulting can be used for an index query recommender

        Args:
            indexer: Index in which new field will be created
            field_data: Data that will be stored in the index
            pipeline_id: Second part of the field name in the indexer index,
                complete field_name is field_name + pipeline_id
            field_name (str): First part of the field name in the indexer index,
                complete field_name is field_name + pipeline_id

        """
        field_data = check_not_tokenized(field_data)
        indexer.new_searching_field(field_name + pipeline_id, field_data)

    def __str__(self):
        return "Indexing for search-engine recommender"


class CollectionBasedTechnique(FieldContentProductionTechnique):
    """
    This class generalizes the techniques that work on the entire content collection,
    such as the tf-idf technique
    """

    def __init__(self):
        super().__init__()
        self.__field_need_refactor: str = None
        self.__pipeline_need_refactor: str = None
        self.__processor_list: List[InformationProcessor] = None

    @property
    def field_need_refactor(self) -> str:
        return self.__field_need_refactor

    @property
    def pipeline_need_refactor(self) -> str:
        return self.__pipeline_need_refactor

    @property
    def processor_list(self) -> List[InformationProcessor]:
        return self.__processor_list

    @field_need_refactor.setter
    def field_need_refactor(self, field_name: str):
        self.__field_need_refactor = field_name

    @pipeline_need_refactor.setter
    def pipeline_need_refactor(self, pipeline_id: str):
        self.__pipeline_need_refactor = pipeline_id

    @processor_list.setter
    def processor_list(self, processor_list: List[InformationProcessor]):
        self.__processor_list = processor_list

    @abstractmethod
    def produce_content(self, field_representation_name: str, content_id: str,
                        field_name: str) -> FieldRepresentation:
        raise NotImplementedError

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, id_field_names):
        """
        This method restructures the raw data in a way functional to the final representation.
        This is done only for those field representations that require this phase to be done

        Args:
            information_source (RawInformationSource):
            id_field_names: fields where to find data that compound content's id
        """
        raise NotImplementedError

    @abstractmethod
    def delete_refactored(self):
        raise NotImplementedError

    def __str__(self):
        return "CollectionBasedTechnique"

    def __repr__(self):
        return "CollectionBasedTechnique "


class SingleContentTechnique(FieldContentProductionTechnique):
    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data) -> FieldRepresentation:
        """
        Given data of certain field it returns a complex representation's instance of the field.

        Args:
            field_representation_name: name of the field representation object that will be created
            field_data: input for the complex representation production

        Returns:
            FieldRepresentation: an instance of FieldRepresentation,
                 the particular type of representation depends from the technique
        """
        raise NotImplementedError


class TfIdfTechnique(CollectionBasedTechnique):
    """
    Class that produce a Bag of words with tf-idf metric
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_representation_name: str, content_id: str,
                        field_name: str) -> FeaturesBagField:
        raise NotImplementedError

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: str):
        raise NotImplementedError

    @abstractmethod
    def delete_refactored(self):
        raise NotImplementedError

    def __str__(self):
        return "TfIdfTechnique"

    def __repr__(self):
        return "TfIdfTechnique "


class EntityLinking(SingleContentTechnique):
    """
    Abstract class that generalizes implementations that use entity linking
    for producing the semantic description
    """

    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data) -> FeaturesBagField:
        raise NotImplementedError


class CombiningTechnique(ABC):
    """
    Class that generalizes the modality in which loaded embeddings will be
    combined to produce a semantic representation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def combine(self, embedding_matrix: np.ndarray):
        """
        Combine, in a way specified in the implementations,
        the row of the input matrix

        Args:
            embedding_matrix: matrix whose rows will be combined

        Returns:

        """
        raise NotImplementedError


class EmbeddingSource(ABC):
    """
    General class whose purpose is to store the loaded pre-trained embeddings model and
    extract specified words from it

    Args:
        self.__model: embeddings model loaded from source
    """

    def __init__(self):
        self.__model = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def load(self, text: List[str]) -> np.ndarray:
        """
        Function that extracts from the embeddings model
        the vectors of the words contained in text

        Args:
            text (list<str>): contains words of which vectors will be extracted

        Returns:
            embedding_matrix (np.ndarray): bi-dimensional numpy vector,
                each row is a term vector
        """
        embedding_matrix = np.ndarray(shape=(len(text), self.get_vector_size()))

        text = check_tokenized(text)
        for i, word in enumerate(text):
            word = word.lower()
            try:
                embedding_matrix[i, :] = self.__model[word]
            except KeyError:
                embedding_matrix[i, :] = np.zeros(self.get_vector_size())

        return embedding_matrix

    def get_vector_size(self) -> int:
        return self.__model.vector_size

    def __str__(self):
        return "EmbeddingSource"

    def __repr__(self):
        return "EmbeddingSource " + str(self.__model)


class EmbeddingTechnique(SingleContentTechnique):
    """
    Class that can be used to combine different embeddings coming from various sources
    in order to produce the semantic description.

    Args:
        combining_technique (CombiningTechnique): The technique that will be used
        for combining the embeddings.
        embedding_source (EmbeddingSource):
        Source where the embeddings vectors for the words in field_data are stored.
        granularity (Granularity): It can assume three values, depending on whether
        the framework user wants to combine relatively to words, phrases or documents.
    """

    def __init__(self, combining_technique: CombiningTechnique,
                 embedding_source: EmbeddingSource,
                 granularity: str):
        super().__init__()
        self.__combining_technique: CombiningTechnique = combining_technique
        self.__embedding_source: EmbeddingSource = embedding_source

        self.__granularity: str = granularity.lower()

    def produce_content(self, field_representation_name: str, field_data) -> EmbeddingField:
        """
        Method that builds the semantic content starting from the embeddings contained in
        field_data.

        Args:
            field_representation_name (str): Name of the field representation for which produce
            the content
            field_data: The terms whose embeddings will be combined.

        Returns:
            EmbeddingField
        """

        if self.__granularity == "word":
            doc_matrix = self.__embedding_source.load(field_data)
            return EmbeddingField(field_representation_name, doc_matrix)
        if self.__granularity == "sentence":
            try:
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('punkt')

            sentences = sent_tokenize(field_data)
            for i, sentence in enumerate(sentences):
                sentences[i] = sentence[:len(sentence) - 1]

            sentences_embeddings = \
                np.ndarray(shape=(len(sentences), self.__embedding_source.get_vector_size()))
            for i, sentence in enumerate(sentences):
                sentence_matrix = self.__embedding_source.load(sentence)
                sentences_embeddings[i, :] = self.__combining_technique.combine(sentence_matrix)

            return EmbeddingField(field_representation_name, sentences_embeddings)
        if self.__granularity == "doc":
            doc_matrix = self.__embedding_source.load(field_data)
            return EmbeddingField(
                field_representation_name, self.__combining_technique.combine(doc_matrix))
        else:
            raise ValueError("Must specify a valid embedding technique granularity")

    def __str__(self):
        return "EmbeddingTechnique"

    def __repr__(self):
        return "EmbeddingTechnique " \
               + str(self.__combining_technique) + " " + \
               str(self.__embedding_source) + " " \
               + str(self.__granularity)
