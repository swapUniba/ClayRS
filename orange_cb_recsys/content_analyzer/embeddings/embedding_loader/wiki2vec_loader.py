from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.embedding_loader import WordEmbeddingLoader
import numpy as np
from wikipedia2vec import Wikipedia2Vec


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
