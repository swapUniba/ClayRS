from gensim import downloader
import numpy as np

from clayrs.utils.const import logger
from clayrs.content_analyzer.embeddings.embedding_loader.embedding_loader import WordEmbeddingLoader


class Gensim(WordEmbeddingLoader):
    """
    Class that produces word embeddings using gensim pre-trained models.

    The model will be automatically downloaded using the gensim downloader api if not present locally.

    Args:
        model_name: Name of the model to load/download
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
        return f'Gensim(model_name={self.reference}'
