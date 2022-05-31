import gc
from typing import List
from abc import ABC, abstractmethod
import numpy as np


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
        self.__model = None

    # this will load/download the model if not already loaded when called
    @property
    def model(self):
        if self.__model is None:
            try:
                self.__model = self.load_model() if self.__reference is not None else None
            except FileNotFoundError:
                self.__model = None
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
        if len(text) > 0:
            embedding_list = []
            for data in text:
                data = data.lower()
                try:
                    embedding_list.append(self.get_embedding(data))
                except KeyError:
                    embedding_list.append(np.zeros(self.get_vector_size()))
            embedding_matrix = np.asarray(embedding_list)
        else:
            # If the text is empty (eg. "") then the embedding matrix is a matrix
            # with 1 row filled with zeros
            embedding_matrix = np.zeros(shape=(1, self.get_vector_size()))

        return embedding_matrix

    @abstractmethod
    def load_model(self):
        """
        Method used to load the model. Each technique should implement this to define how the model is loaded
        """
        raise NotImplementedError

    def unload_model(self):
        self.__model = None
        gc.collect()

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

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
