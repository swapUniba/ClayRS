from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.embedding_loader import SentenceEmbeddingLoader

from sentence_transformers import SentenceTransformer
import numpy as np


class Sbert(SentenceEmbeddingLoader):
    """
    This class loads the embeddings using the SentenceTransformer (from sbert).

    Args:
        model_name_or_file_path (str): name of the embeddings model to download or path where the model is stored
            locally
    """

    def get_embedding_token(self, sentence: str) -> np.ndarray:
        raise NotImplementedError

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
