from orange_cb_recsys.content_analyzer.embeddings.embedding_source import EmbeddingSource
from abc import abstractmethod
import numpy as np


class EmbeddingLoader(EmbeddingSource):
    """
    Abstract class that generalizes the behavior of the techniques that load an embedding model from a local file or
    download it from the internet. In particular, these techniques can only load the model, they don't provide a method
    to train the model
    """

    def __init__(self, reference: str):
        super().__init__(reference)

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, data) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class WordEmbeddingLoader(EmbeddingLoader):
    """
    Defines the granularity of a loader which will be 'word'
    """

    def __init__(self, reference: str):
        super().__init__(reference)

    @abstractmethod
    def load_model(self):
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


class SentenceEmbeddingLoader(EmbeddingLoader):
    """
    Defines the granularity of a loader which will be 'sentence'
    """

    def __init__(self, reference: str):
        super().__init__(reference)

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, sentence: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class DocumentEmbeddingLoader(EmbeddingLoader):
    """
    Defines the granularity of a loader which will be 'document'
    """

    def __init__(self, reference: str):
        super().__init__(reference)

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, document: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
