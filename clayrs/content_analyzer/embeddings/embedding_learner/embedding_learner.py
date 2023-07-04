from __future__ import annotations
from abc import abstractmethod
from typing import List, Union, TYPE_CHECKING

import numpy as np
from gensim.models import KeyedVectors

if TYPE_CHECKING:
    from clayrs.content_analyzer.information_processor.information_processor_abstract import InformationProcessor
    from clayrs.content_analyzer.raw_information_source import RawInformationSource

from clayrs.content_analyzer.embeddings.embedding_source import EmbeddingSource
from clayrs.content_analyzer.utils.check_tokenization import check_tokenized, tokenize_in_sentences, check_not_tokenized
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar


class EmbeddingLearner(EmbeddingSource):
    """
    Abstract Class for the different kinds of embedding
    The EmbeddingLearner allows to train and store locally embeddings learned from a source on a specific field list.
    So, for example:

        learner = GensimWord2Vec("somedir/word2vec.model")
        learner.fit(JSONFile(file_path), ['Plot', 'Genre'], NLTK())

    The most important method is fit
    It calls the method that creates the model and, if the user wants, it stores the model locally.
    In the example, the model will be fitted on the source located in file_path on the "Plot" and "Genre" fields.
    Preprocessing will also be applied to the text in the source, in this case using NLTK

    Args:
        reference (str): path where the model is stored. If you want to train the model on the spot, pass the path where
            the model will be stored if you save it

            Example:
                I want to load the model locally giving the file_path of the model

                Assume the model file named "example.model" is stored in the dir: "somedir"
                The file_path then will be: "somedir/example.model" or "somedir/example"

                I want to train the model and save it locally

                Assume the dir where the model will be stored is: "somedir"
                The file_path will be as above: "somedir/example.model" or "somedir/example"
                Meaning you have to pass the file name of the model you are going to save

            Alternatively, you can also pass None as reference, in which case, the model won't be saved but it will
            be stored in the Embedding Learner instance (useful if you want to use the trained model only once and then
            discard it)

        auto_save (bool): if True, the model is automatically saved locally after fitting. In particular, you can set
            this to true if you want the model trained during the content analysis process to be saved locally

        extension (str): defines what type of extension the model file will be (for example: .model or .bin). This is
            done so that the user can define a file path with or without the extension (so for example, both
            "somedir/model_file" and "somedir/model_file.model" are acceptable)

        kwargs: arguments that you would pass to any of the models if using them from their original library (for
            example, you can pass "workers=4, min_count=1" to the FastText model)
    """
    def __init__(self, file_path: str, auto_save: bool, extension: str, **kwargs):
        # adds the extension related to the learner if the user didn't pass it
        if file_path is not None and not file_path.endswith(extension):
            file_path += extension

        super().__init__(file_path)

        self._auto_save = auto_save
        self._additional_parameters = kwargs

    @property
    def additional_parameters(self):
        return self._additional_parameters

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    def fit(self, source: RawInformationSource, field_list: List[str],
            preprocessor_list: Union[List[InformationProcessor], InformationProcessor] = None):
        """
        Method that handles the creation and storing of the model
        If the attribute auto_save is True, it automatically stores the model locally after it has been trained

        Args:
            source (RawInformationSource): raw data on which the fitting process will be done
            field_list (List[str]): list of fields to consider from the raw data
            preprocessor_list (Union[List[InformationProcessor], InformationProcessor]): either a list or a single
                information processor that will be used to process the raw data in the fields defined in field list
        """

        if preprocessor_list is None:
            preprocessor_list = []

        if not isinstance(preprocessor_list, list):
            preprocessor_list = [preprocessor_list]

        corpus = self.extract_corpus(source, field_list, preprocessor_list)

        logger.info("Fitting model with extracted corpus...")
        self.fit_model(corpus)

        if self._auto_save and self.reference is not None:
            self.save()

    @abstractmethod
    def fit_model(self, corpus: List):
        """
        This method creates the model, in different ways according to the various implementations.
        The model isn't then returned, but gets stored in the 'model' instance attribute.

        Args:
            corpus (List): data extracted and processed from the raw source which will be used to train the model
        """
        raise NotImplementedError

    def extract_corpus(self, source: RawInformationSource, field_list: List[str],
                       preprocessor_list: List[InformationProcessor]) -> list:
        """
        Extracts the data from the source, from the fields specified in the field_list argument, and processes it
        using the processor_list passed as argument)

        Args:
            source (RawInformationSource): raw data on which the fitting process will be done
            field_list (List[str]): list of fields to consider from the raw data
            preprocessor_list (Union[List[InformationProcessor], InformationProcessor]): either a list or a single
                information processor that will be used to process the raw data in the fields defined in field list

        Returns:
            corpus (list): List of processed data
        """
        corpus = []
        # iter the source
        with get_progbar(list(source)) as pbar:

            for doc in pbar:
                pbar.set_description(f"Preprocessing {', '.join(field_list)} for all contents")
                doc_data = ""
                for field_name in field_list:
                    # apply preprocessing and save the data in the list
                    doc_data += " " + doc[field_name].lower()
                for preprocessor in preprocessor_list:
                    doc_data = preprocessor.process(doc_data)
                corpus.append(self.process_data_granularity(doc_data))
        return corpus

    @abstractmethod
    def process_data_granularity(self, doc_data: str) -> Union[List[str], str]:
        """
        Method that applies modifications to the data in order to fit the granularity of the technique

        Args:
            doc_data (str): data to be modified

        Returns:
            doc_data modified to fit the granularity. For example, if the technique had word granularity and doc_data
                was "this is an example", the output would be ["this", "is", "an", "example"]
        """
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """
        Saves the model in the path stored in the file_path attribute
        """
        raise NotImplementedError


class WordEmbeddingLearner(EmbeddingLearner):
    """
    Extends the EmbeddingLearner to define the embedding learners that work with 'word'
    granularity, meaning that the model expects a list of words as training data
    """

    def __init__(self, reference: str, auto_save: bool, extension: str, **kwargs):
        super().__init__(reference, auto_save, extension, **kwargs)

    def process_data_granularity(self, doc_data: str) -> List[str]:
        return check_tokenized(doc_data)

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def fit_model(self, corpus: List):
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, word: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class GensimWordEmbeddingLearner(WordEmbeddingLearner):
    """
    Class that contains the generic behavior of the Gensim models
    """

    def __init__(self, reference: str, auto_save: bool, extension: str, **kwargs):
        super().__init__(reference, auto_save, extension, **kwargs)

    def get_vector_size(self) -> int:
        return self.model.vector_size

    def get_embedding(self, word: str) -> np.ndarray:
        return self.model[word]

    def load_model(self):
        return KeyedVectors.load_word2vec_format(self.reference, binary=True)

    def save(self):
        self.model.save_word2vec_format(self.reference, binary=True)

    @abstractmethod
    def fit_model(self, corpus: List):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class SentenceEmbeddingLearner(EmbeddingLearner):
    """
    Extends the EmbeddingLearner to define the embedding learners that work with
    'sentence' granularity, meaning that the model expects a list of sentences as training data
    """

    def __init__(self, reference: str, auto_save: bool, extension: str, **kwargs):
        super().__init__(reference, auto_save, extension, **kwargs)

    def process_data_granularity(self, doc_data: str) -> List[str]:
        return tokenize_in_sentences(doc_data)

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def fit_model(self, corpus: List):
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, sentence: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_embedding_token(self, sentence: str):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class DocumentEmbeddingLearner(EmbeddingLearner):
    """
    Extends the EmbeddingLearner to define the embedding learners that work with
    'document' granularity, meaning that the model expects a list of documents as training data
    """

    def __init__(self, reference: str, auto_save: bool, extension: str, **kwargs):
        super().__init__(reference, auto_save, extension, **kwargs)

    def process_data_granularity(self, doc_data: str) -> str:
        return check_not_tokenized(doc_data)

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def fit_model(self, corpus: List):
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, document: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_embedding_sentence(self, document: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_embedding_token(self, document: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class GensimDocumentEmbeddingLearner(DocumentEmbeddingLearner):
    """
    Class that contains the generic behavior of the Gensim models working at document granularity
    """

    def process_data_granularity(self, doc_data: str) -> str:
        # gensim requires document data to be tokenized in a list
        return check_tokenized(doc_data)

    def get_embedding_sentence(self, document_tokenized: List[str]) -> np.ndarray:
        raise NotImplementedError

    def get_embedding_token(self, document_tokenized: List[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self):
        self.model.save(self.reference)
