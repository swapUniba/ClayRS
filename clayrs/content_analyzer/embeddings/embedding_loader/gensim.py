from gensim.models import KeyedVectors

from clayrs.content_analyzer.embeddings.embedding_loader.embedding_loader import WordEmbeddingLoader

import gensim.downloader as downloader
import numpy as np

from clayrs.utils.const import logger


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
            logger.info(f"Downloading/Loading {str(self)}")

            return downloader.load(self.reference)
        else:
            raise FileNotFoundError

    def __str__(self):
        return f"Gensim {self.reference}"

    def __repr__(self):
        return f'Gensim(model name={str(self.model)}'
