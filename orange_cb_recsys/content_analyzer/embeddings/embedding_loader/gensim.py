from gensim.models import KeyedVectors

from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.embedding_loader import WordEmbeddingLoader

import gensim.downloader as downloader
import numpy as np

from orange_cb_recsys.utils.const import logger


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
            logger.info("Downloading/Loading gensim model")

            return downloader.load(self.reference)
        else:
            raise FileNotFoundError

    def __str__(self):
        return "Gensim"

    def __repr__(self):
        return "< Gensim: model = " + str(self.model) + ">"
