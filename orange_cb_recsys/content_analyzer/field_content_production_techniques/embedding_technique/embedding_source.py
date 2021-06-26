from typing import List
from abc import ABC, abstractmethod

import gensim.downloader as downloader
from sentence_transformers import SentenceTransformer
import numpy as np
from wikipedia2vec import Wikipedia2Vec


class EmbeddingSource(ABC):
    """
    General class whose purpose is to store the loaded pre-trained embeddings model and extract specified data from it

    The embedding source works in the following way: data -> model -> embedding vector
    The source iterates over all the data and returns the embedding vectors (in a matrix)

    EmbeddingSource has two classes that inherit from it: EmbeddingLoader and EmbeddingLearner. The first is used
    for models downloadable from the internet or stored locally (in particular, models that cannot be trained),
    the second is used for models stored locally that can be trained. Because of this, there
    shouldn't be any need for any other classes

    model: embeddings model loaded from source

    Args:
        reference (str): where to find the model, could be the model name to download or the path where the model is
            located
    """

    def __init__(self, reference: str):
        self.__reference = reference
        try:
            self.__model = self.load_model() if self.__reference is not None else None
        except FileNotFoundError:
            self.__model = None

    @property
    def model(self):
        return self.__model

    @property
    def reference(self):
        return self.__reference

    @model.setter
    def model(self, model):
        self.__model = model

    def load(self, text: List[str]) -> np.ndarray:
        """
        Function that extracts from the embeddings model the vectors of the data contained in text. If the model can't
        return a vector for the data passed as argument, then a vector filled with 0 will be created.

        Args:
            text (list<str>): text from which the embedding vectors will be extracted. The text contents depend from the
                embedding source's granularity. For example, a word embedding source will contain a list of tokens.

        Returns:
            embedding_matrix (np.ndarray): numpy vector, where each row represents the vector for the granularity (if
                the source has sentence granularity, the vector will refer to the corresponding sentence).
                Assuming text is a list of length N (where N depends by the granularity of the technique, so it could
                be the number of words or sentences), embedding_matrix will be N-dimensional.
        """
        embedding_matrix = np.ndarray(shape=(len(text), self.get_vector_size()))

        for i, data in enumerate(text):
            data = data.lower()
            try:
                embedding_matrix[i, :] = self.get_embedding(data)
            except KeyError:
                embedding_matrix[i, :] = np.zeros(self.get_vector_size())

        return embedding_matrix

    @abstractmethod
    def load_model(self):
        """
        Method used to load the model. Each technique should implement this to define how the model is loaded
        """
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        """
        Method that defines the size of a single embedding vector
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, data) -> np.ndarray:
        """
        Method to return the embedding vector of the data passed as argument
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


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


class Gensim(WordEmbeddingLoader):
    """
    This class loads the embeddings using the gensim downloader API.

    Args:
        model_name (str): name of the embeddings model to load
    """

    def __init__(self, model_name: str = 'glove-twitter-25'):
        super().__init__(model_name)

    def get_vector_size(self) -> int:
        return self.model.vector_size

    def get_embedding(self, word: str) -> np.ndarray:
        return self.model[word]

    def load_model(self):
        # if the reference isn't in the possible models, FileNotFoundError is raised
        if self.reference in downloader.info()['models']:
            return downloader.load(self.reference)
        else:
            raise FileNotFoundError

    def __str__(self):
        return "Gensim"

    def __repr__(self):
        return "< Gensim: model = " + str(self.model) + ">"


class Wikipedia2VecLoader(WordEmbeddingLoader):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings using the Wikipedia2Vec binary file loader.
    Can be used for loading of pre-trained wikipedia dump embedding,
    both downloaded or trained on local machine.

    Args:
        path (str): path for the binary file containing the embeddings
    """

    def __init__(self, path: str):
        super().__init__(path)

    def load_model(self):
        try:
            return Wikipedia2Vec.load(self.reference)
        except (FileNotFoundError, KeyError):
            raise FileNotFoundError

    def get_vector_size(self) -> int:
        return self.model.get_word_vector("a").shape[0]

    def get_embedding(self, word: str) -> np.ndarray:
        return self.model.get_word_vector(word)

    def __str__(self):
        return "Wikipedia2Vec"

    def __repr__(self):
        return "< Wikipedia2Vec: model = " + str(self.model) + ">"


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


class Sbert(SentenceEmbeddingLoader):
    """
    This class loads the embeddings using the SentenceTransformer (from sbert).

    Args:
        model_name_or_file_path (str): name of the embeddings model to download or path where the model is stored
            locally
    """

    def __init__(self, model_name_or_file_path: str = 'paraphrase-distilroberta-base-v1'):
        super().__init__(model_name_or_file_path)

    def load_model(self):
        try:
            return SentenceTransformer(self.reference)
        except (OSError, AttributeError):
            raise FileNotFoundError

    def get_vector_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def get_embedding(self, sentence: str) -> np.ndarray:
        return self.model.encode(sentence, show_progress_bar=False)

    def __str__(self):
        return "Sbert"

    def __repr__(self):
        return "< Sbert: model = " + str(self.model) + ">"


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
