import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.embedding_loader import SentenceEmbeddingLoader
from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.vector_strategy import VectorStrategy, CatStrategy
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique \
    import CombiningTechnique, Centroid


class Transformers(SentenceEmbeddingLoader):
    """
    This class loads the embeddings using the Transformers from pretrained model.

    Args:
        model_name (str): name of the embeddings model to download or path where the model is stored locally
        vec_strategy (VectorStrategy): strategy with which to combine layers for each token
        pooling_strategy (CombiningTechnique): strategy with which to combine embeddings for each token
    """

    def __init__(self, model_name: str, vec_strategy: VectorStrategy = CatStrategy(1),
                 pooling_strategy: CombiningTechnique = Centroid()):
        self.model = None
        self.tokenizer = None
        self.name_model = model_name
        self.vec_strategy = vec_strategy
        self.last_interesting_layers = vec_strategy.last_interesting_layers
        self.pooling_strategy = pooling_strategy
        super().__init__(model_name)

    def load_model(self):
        try:
            self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            return self.model
        except (OSError, AttributeError):
            raise FileNotFoundError

    def get_vector_size(self) -> int:
        return self.model.embeddings.token_type_embeddings.embedding_dim * self.last_interesting_layers

    def get_embedding(self, sentence: str) -> np.ndarray:
        token_vecs = self.get_embedding_token(sentence)
        sentence_vec = self.pooling_strategy.combine(token_vecs)
        return sentence_vec

    def get_embedding_token(self, sentence: str) -> np.ndarray:
        encoded = self.tokenizer.encode_plus(sentence)
        tokens_tensor = torch.tensor([encoded['input_ids']])
        segments_tensors = torch.tensor([encoded['attention_mask']])

        with torch.no_grad():
            model_output = self.model(tokens_tensor, segments_tensors)
            hidden_states = model_output[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_vecs = self.vec_strategy.build_embedding(token_embeddings)

        return token_vecs

    def get_sentence_token(self, sentence: str):
        """
        method that returns the tokenization of the sentence
        Args:
            sentence: sentence to analyze

        Returns: list containing tokens

        """
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        return tokenized_text

    def __str__(self):
        return "Transformers: model = " + str(self.model)

    def __repr__(self):
        return str(self)
