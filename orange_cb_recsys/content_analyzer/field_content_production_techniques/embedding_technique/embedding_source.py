from typing import List

import gensim.downloader as downloader
from gensim.models import KeyedVectors, Doc2Vec, fasttext, RpModel
from wikipedia2vec import Wikipedia2Vec
import numpy as np

from orange_cb_recsys.content_analyzer.field_content_production_techniques.\
    field_content_production_technique import EmbeddingSource
from orange_cb_recsys.utils.check_tokenization import check_tokenized


class BinaryFile(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings from a binary file
    in a way that depends from embedding_type.

    Args:
        file_path (str): path for the binary file containing the embeddings
        embedding_type (str): Name of the technique used to learn
            the embedding that is being loaded
            the possible values are: "word2vec", "doc2vec", "fasttext", "ri"
    """

    def __init__(self, file_path: str,
                 embedding_type: str):
        super().__init__()
        self.__file_path: str = file_path
        embedding_type = embedding_type.lower()
        if embedding_type == "word2vec":
            self.model = KeyedVectors.load_word2vec_format(self.__file_path, binary=True)
        elif embedding_type == "doc2vec":
            self.model = Doc2Vec.load(self.__file_path)
        elif embedding_type == "fasttext":
            self.model = fasttext.load_facebook_vectors(self.__file_path)
        elif embedding_type == "ri":
            self.model = RpModel.load(self.__file_path)
        else:
            raise ValueError(
                "Must specify a valid embedding model type for loading from binary file")


class GensimDownloader(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings using the gensim downloader API.

    Args:
        name (str): name of the embeddings model to load
    """

    def __init__(self, name: str):
        super().__init__()
        self.__name: str = name
        self.model = downloader.load(self.__name)


class Wikipedia2VecDownloader(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSoruce.
    This class loads the embeddings using the Wikipedia2Vec binary file loader.
    Can be used for loading of pre-trained wikipedia dump embedding,
    both downloaded or trained on local machine.

    Args:
        path (str): path for the binary file containing the embeddings
    """

    def __init__(self, path: str):
        super().__init__()
        self.__path: str = path

        self.model = Wikipedia2Vec.load(self.__path)

    def get_vector_size(self) -> int:
        return self.model.get_word_vector("a").shape[0]

    def load(self, text: List[str]) -> np.ndarray:
        """
        Function that extracts from the embeddings model
        the vectors of the words contained in text

        Args:
            text (list<str>): list of words of which vectors will be extracted

        Returns:
            embedding_matrix (np.ndarray): bi-dimensional numpy vector, each row is a term vector
        """
        embedding_matrix = np.ndarray(shape=(len(text), self.get_vector_size()))

        text = check_tokenized(text)

        for i, word in enumerate(text):
            word = word.lower()
            try:
                embedding_matrix[i, :] = self.model.get_word_vector(word)
            except KeyError:
                embedding_matrix[i, :] = np.zeros(self.get_vector_size())

        return embedding_matrix

# your embedding source
