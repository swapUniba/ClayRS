from __future__ import annotations
from abc import abstractmethod

import numpy as np
from typing import Union, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.embeddings.embedding_learner.embedding_learner import WordEmbeddingLearner, \
        SentenceEmbeddingLearner, DocumentEmbeddingLearner
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
        CombiningTechnique
    from clayrs.content_analyzer.embeddings.embedding_loader.embedding_loader import \
        WordEmbeddingLoader, SentenceEmbeddingLoader, DocumentEmbeddingLoader
    from clayrs.content_analyzer.embeddings.embedding_loader.embedding_loader import EmbeddingSource
    from clayrs.content_analyzer.information_processor.information_processor_abstract import InformationProcessor
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.visualpostprocessor import VisualPostProcessor

from clayrs.content_analyzer.embeddings.embedding_learner.embedding_learner import EmbeddingLearner
from clayrs.content_analyzer.content_representation.content import EmbeddingField
from clayrs.content_analyzer.embeddings.embedding_loader.embedding_loader import EmbeddingLoader, EmbeddingSource
from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SingleContentTechnique
from clayrs.content_analyzer.utils.check_tokenization import check_tokenized, tokenize_in_sentences, check_not_tokenized
from clayrs.utils.class_utils import get_all_implemented_subclasses
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar


class EmbeddingTechnique(SingleContentTechnique):
    """
    Abstract class that generalizes the techniques that create embedding vectors. The EmbeddingTechniques can be split
    into different categories depending on the type of granularity the technique has. For example, a word granularity
    embedding will have a resulting matrix where each row refers to a specific word in the text.

    Args:
        embedding_source (EmbeddingSource): Source where the embeddings vectors for the words in field_data
            are stored.
    """

    def __init__(self, embedding_source: EmbeddingSource):

        super().__init__()

        self.__embedding_source = embedding_source

    @staticmethod
    def from_str_to_embedding_source(embedding_source_str: str, loader_class: Type[EmbeddingLoader]) -> EmbeddingSource:
        """
        Method used to convert a string (which represents a model name) to a corresponding Embedding Source that can
        use the defined model
        Given the loader class (which is a class inheriting from EmbeddingSourceLoader), the method checks each
        implemented class inheriting from loader_class and returns the one that is able to load the model

        The method raises FileNotFoundError if no embedding loader is able to load the model

        Be careful when passing a string, because the first Loader able to load the model will be kept. So if there are
        multiple loaders capable of loading a model, the first one found will be used

        Args:
            embedding_source_str (str): string representing the model name ('twitter-glove-25' for example)
            loader_class (Type): class that inherits from EmbeddingSourceLoader, each subclass of this class will be
                tested with the model name (stored in embedding_source_str)

        Returns:
            embedding_source (EmbeddingSource): embedding source which can load the model
        """
        # retrieves all implementations (meaning not abstract classes) inheriting from loader_class
        possible_implementations = get_all_implemented_subclasses(loader_class)
        # each implementation is tested and, when one of the implementations loads the model successfully, the loader
        # instance is returned
        for implementation in possible_implementations:
            embedding_source = implementation(embedding_source_str)
            if embedding_source.model is not None:
                logger.info("The EmbeddingSource %s was found for the %s reference"
                            % (implementation.__name__, embedding_source_str))
                return embedding_source
        # if no class was found to process the model an exception is raised
        raise FileNotFoundError("The system couldn't process %s as a valid embedding reference"
                                % embedding_source_str)

    @property
    def embedding_source(self):
        return self.__embedding_source

    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        postprocessor_list: List[VisualPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        representation_list: List[FieldRepresentation] = []

        if isinstance(self.__embedding_source, EmbeddingLoader) and self.__embedding_source.model is None:
            raise FileNotFoundError("The reference %s was not valid for the %s source" %
                                    (self.__embedding_source.reference, self.__embedding_source))

        # if the embedding source is an EmbeddingLearner (meaning it can be trained) and the source has no model
        # the source is trained
        if isinstance(self.__embedding_source, EmbeddingLearner) and self.__embedding_source.model is None:
            logger.warning("The model %s wasn't found, so it will be created and trained now" %
                           self.__embedding_source.reference)
            logger.warning("The model will be trained on the %s field "
                           "and the data will be processed with %s" % (field_name, preprocessor_list))
            self.__embedding_source.fit(source, [field_name], preprocessor_list)

        # it iterates over all contents contained in the source in order to retrieve the raw data
        # the data contained in the field_name is processed using each information processor in the processor_list
        # the data is passed to the method that will create the single representation
        with get_progbar(list(source)) as pbar:

            for content_data in pbar:

                pbar.set_description(f"Processing and producing contents with {self.__embedding_source}")

                processed_data = self.process_data(content_data[field_name], preprocessor_list)
                representation_list.append(self.produce_single_repr(processed_data))

            representation_list = self.postprocess_representations(representation_list, postprocessor_list)

        self.embedding_source.unload_model()
        return representation_list

    @abstractmethod
    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        """
        Method that builds the semantic content starting from the embeddings contained in field_data which can
        be either a list of tokens or the entire text in string form (each technique will then have to apply
        processing operations to the data, in order to obtain it in the desired form)

        Args:
            field_data (List[str], str): Data contained in the field, it can be either a list of tokens or the
                original data in string form

        Returns:
            EmbeddingField: complex representation created using the field data
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class StandardEmbeddingTechnique(EmbeddingTechnique):
    """
    Class that generalizes the embedding techniques working with the corresponding EmbeddingSource
    StandardEmbeddingTechnique can be extended to consider different types of granularity (word, sentence, ...)
    and each technique should refer to the corresponding EmbeddingSource, so, for example, the
    WordEmbeddingTechnique should refer to the WordEmbeddingSource.

    To sum it up, this class contains the techniques that don't apply any kind of additional operation other
    than loading the embedding from the source
    """

    def __init__(self, embedding_source: EmbeddingSource):
        super().__init__(embedding_source)

    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        return EmbeddingField(self.embedding_source.load(self.process_data_granularity(field_data)))

    @abstractmethod
    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class WordEmbeddingTechnique(StandardEmbeddingTechnique):
    """
    Class that makes use of a *word* granularity embedding source to produce *word* embeddings

    Args:
        embedding_source: Any `WordEmbedding` model
    """

    def __init__(self, embedding_source: Union[WordEmbeddingLoader, WordEmbeddingLearner, str]):
        # if isinstance(embedding_source, str):
        #     embedding_source = self.from_str_to_embedding_source(embedding_source, WordEmbeddingLoader)
        super().__init__(embedding_source)

    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        return check_tokenized(field_data)

    def __str__(self):
        return "WordEmbeddingTechnique"

    def __repr__(self):
        return f'WordEmbeddingTechnique(embedding_source={self.embedding_source})'


class SentenceEmbeddingTechnique(StandardEmbeddingTechnique):
    """
    Class that makes use of a *sentence* granularity embedding source to produce *sentence* embeddings

    Args:
        embedding_source: Any `SentenceEmbedding` model
    """

    def __init__(self, embedding_source: Union[SentenceEmbeddingLoader, SentenceEmbeddingLearner, str]):
        # if isinstance(embedding_source, str):
        #     embedding_source = self.from_str_to_embedding_source(embedding_source, SentenceEmbeddingLoader)
        super().__init__(embedding_source)

    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        return tokenize_in_sentences(field_data)

    def __str__(self):
        return "SentenceEmbeddingTechnique"

    def __repr__(self):
        return f'SentenceEmbeddingTechnique(embedding_source={self.embedding_source})'


class DocumentEmbeddingTechnique(StandardEmbeddingTechnique):
    """
    Class that makes use of a *document* granularity embedding source to produce *document* embeddings

    Args:
        embedding_source: Any `DocumentEmbedding` model
    """

    def __init__(self, embedding_source: Union[DocumentEmbeddingLoader, DocumentEmbeddingLearner, str]):
        # if isinstance(embedding_source, str):
        #     embedding_source = self.from_str_to_embedding_source(embedding_source, DocumentEmbeddingLoader)
        super().__init__(embedding_source)

    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        return [check_not_tokenized(field_data)]

    def __str__(self):
        return "DocumentEmbeddingTechnique"

    def __repr__(self):
        return f'DocumentEmbeddingTechnique(embedding_source={self.embedding_source})'


class CombiningEmbeddingTechnique(EmbeddingTechnique):
    """
    Class that generalizes the embedding techniques working with a combining technique
    CombiningEmbeddingTechnique can be extended to consider different types of granularity (word, sentence, ...)
    and each technique should be further extended in order to consider the different embedding sources that
    can be combined into the defined granularity.
    For example, with the document granularity, it is possible to combine the embedding matrix of any granularity with a
    lower scope, so word or sentence. With the sentence granularity, it is possible to combine the word embeddings but
    not the document ones (because document's scope is bigger than sentence's scope).

    To sum it up, this class contains the techniques that make use of an EmbeddingSource with different granularity from
    theirs and apply a combining technique to create their embedding matrix

    Args:
        combining_technique (CombiningTechnique): technique that will be used to combine the embeddings retrieved from
        the source
    """

    def __init__(self, embedding_source: EmbeddingSource, combining_technique: CombiningTechnique):
        super().__init__(embedding_source)
        self.__combining_technique = combining_technique

    @property
    def combining_technique(self) -> CombiningTechnique:
        return self.__combining_technique

    @abstractmethod
    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        raise NotImplementedError

    @abstractmethod
    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class CombiningSentenceEmbeddingTechnique(CombiningEmbeddingTechnique):
    """
    Class that generalizes the combining embedding techniques with sentence granularity
    """

    def __init__(self, embedding_source: EmbeddingSource, combining_technique: CombiningTechnique):
        super().__init__(embedding_source, combining_technique)

    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        """
        Produces a single representation with sentence granularity by combining the embedding vectors in order to
        create an embedding matrix that represents the document
        """
        # text is split into sentences
        sentences = tokenize_in_sentences(field_data)

        # the sentences embedding matrix is created and the combined vectors added
        sentences_embeddings = np.ndarray(shape=(len(sentences), self.embedding_source.get_vector_size()))
        for i, sentence in enumerate(sentences):
            sentence_matrix = self.embedding_source.load(self.process_data_granularity(sentence))
            sentences_embeddings[i, :] = self.combining_technique.combine(sentence_matrix)

        return EmbeddingField(sentences_embeddings)

    @abstractmethod
    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class Word2SentenceEmbedding(CombiningSentenceEmbeddingTechnique):
    """
    Class that makes use of a word granularity embedding source to produce sentence embeddings

    Args:
        embedding_source: Any `WordEmbedding` model
        combining_technique: Technique used to combine embeddings of finer granularity (word-level) to obtain embeddings
            of coarser granularity (sentence-level)
    """

    def __init__(self, embedding_source: Union[WordEmbeddingLoader, WordEmbeddingLearner, str],
                 combining_technique: CombiningTechnique):
        # if isinstance(embedding_source, str):
        #     embedding_source = self.from_str_to_embedding_source(embedding_source, WordEmbeddingLoader)
        super().__init__(embedding_source, combining_technique)

    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        return check_tokenized(field_data)

    def __str__(self):
        return "Word2SentenceEmbedding"

    def __repr__(self):
        return f"Word2SentenceEmbedding(embedding_source={self.embedding_source}, " \
               f"combining_technique={self.combining_technique})"


class CombiningDocumentEmbeddingTechnique(CombiningEmbeddingTechnique):
    """
    Class that generalizes the combining embedding techniques with document granularity
    """

    def __init__(self, embedding_source: EmbeddingSource, combining_technique: CombiningTechnique):
        super().__init__(embedding_source, combining_technique)

    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        """
        Produces a single representation with document granularity by combining the embedding vectors in order to
        create an embedding matrix that represents the document
        """
        doc_matrix = self.embedding_source.load(self.process_data_granularity(check_not_tokenized(field_data)))
        return EmbeddingField(self.combining_technique.combine(doc_matrix))

    @abstractmethod
    def process_data_granularity(self, data: Union[List[str], str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class Word2DocEmbedding(CombiningDocumentEmbeddingTechnique):
    """
    Class that makes use of a *word* granularity embedding source to produce embeddings of *document* granularity

    Args:
        embedding_source: Any `WordEmbedding` model
        combining_technique: Technique used to combine embeddings of finer granularity (word-level) to obtain embeddings
            of coarser granularity (doc-level)
    """

    def __init__(self, embedding_source: Union[WordEmbeddingLoader, WordEmbeddingLearner, str],
                 combining_technique: CombiningTechnique):
        # if isinstance(embedding_source, str):
        #     embedding_source = self.from_str_to_embedding_source(embedding_source, WordEmbeddingLoader)
        super().__init__(embedding_source, combining_technique)

    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        return check_tokenized(field_data)

    def __str__(self):
        return "Word2DocEmbedding"

    def __repr__(self):
        return f"Word2DocEmbedding(embedding_source={self.embedding_source}, " \
               f"combining_technique={self.combining_technique})"


class Sentence2DocEmbedding(CombiningDocumentEmbeddingTechnique):
    """
    Class that makes use of a *sentence* granularity embedding source to produce embeddings of *document* granularity


    Args:
        embedding_source: Any `SentenceEmbedding` model
        combining_technique: Technique used to combine embeddings of finer granularity (sentence-level) to obtain
            embeddings of coarser granularity (doc-level)
    """

    def __init__(self, embedding_source: Union[SentenceEmbeddingLoader, SentenceEmbeddingLearner, str],
                 combining_technique: CombiningTechnique):
        # if isinstance(embedding_source, str):
        #     embedding_source = self.from_str_to_embedding_source(embedding_source, SentenceEmbeddingLoader)
        super().__init__(embedding_source, combining_technique)

    def process_data_granularity(self, field_data: Union[List[str], str]) -> List[str]:
        return tokenize_in_sentences(field_data)

    def __str__(self):
        return "Sentence2DocEmbedding"

    def __repr__(self):
        return f"Sentence2DocEmbedding(embedding_source={self.embedding_source}, " \
               f"combining_technique={self.combining_technique})"


class DecombiningEmbeddingTechnique(EmbeddingTechnique):
    """
    Class generalizing embedding techniques that contain methods for extracting embedding at finer granularity.
    DecombiningEmbeddingTechnique can be extended to consider Sentence or Document granularity.
    Each technique should be further extended to consider the different sources of embedding that can be combined into
    the defined granularity.
    For example, with Sentence granularity, it is possible, if the model is capable, to extrapolate embedding with word
    granularity.

    To summarize, this class contains techniques that make use of an EmbeddingSource with a granularity different from
    their own, and have the ability to extract the embedding matrix with finer granularity

    """

    def __init__(self, embedding_source: EmbeddingSource):
        super().__init__(embedding_source)

    @abstractmethod
    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:  # return array numpy
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class DecombiningInWordsEmbeddingTechnique(DecombiningEmbeddingTechnique):
    """
    Class that generalizes the decombining embedding techniques from a coarse granularity to the finest granularity word
    """

    def __init__(self, embedding_source: Union[SentenceEmbeddingLoader, SentenceEmbeddingLearner,
                                               DocumentEmbeddingLoader, DocumentEmbeddingLearner]):
        super().__init__(embedding_source)

    @abstractmethod
    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class Sentence2WordEmbedding(DecombiningInWordsEmbeddingTechnique):
    """
    Class that makes use of a sentence granularity embedding source to produce an embedding matrix with word granularity
    """

    def __init__(self, embedding_source: Union[SentenceEmbeddingLoader, SentenceEmbeddingLearner]):
        # if isinstance(embedding_source, str):
        #     embedding_source = self.from_str_to_embedding_source(embedding_source, SentenceEmbeddingLoader)
        super().__init__(embedding_source)

    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        """
        Produces a single matrix where each row is the embedding representation of each token of the sentence,
        while the columns are the hidden dimension of the chosen model

        Args:
            field_data: textual data to complexly represent

        Returns:
            Embedding for each token of the sentence

        """
        field_data = check_not_tokenized(field_data)
        embedding_source: Union[SentenceEmbeddingLoader, SentenceEmbeddingLearner] = self.embedding_source
        words_embeddings = embedding_source.get_embedding_token(field_data)
        return EmbeddingField(words_embeddings)

    def __str__(self):
        return "Sentence2WordEmbedding"

    def __repr__(self):
        return f'Sentence2WordEmbedding(embedding_source={self.embedding_source})'
