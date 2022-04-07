import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.embedding_loader import SentenceEmbeddingLoader
from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.vector_strategy import VectorStrategy, CatStrategy
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique \
    import CombiningTechnique, Centroid


class Transformers(SentenceEmbeddingLoader):
    """
    This class loads the embeddings using the Transformers from pretrained _model.

    Args:
        model_name (str): name of the embeddings _model to download or path where the _model is stored locally
        vec_strategy (VectorStrategy): strategy with which to combine layers for each token
        pooling_strategy (CombiningTechnique): strategy with which to combine embeddings for each token
    """

    def __init__(self, model_name: str = 'bert-base-uncased',
                 vec_strategy: VectorStrategy = CatStrategy(1),
                 pooling_strategy: CombiningTechnique = Centroid()):
        self._model = None
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._name_model = model_name
        self._vec_strategy = vec_strategy
        self._last_interesting_layers = vec_strategy.last_interesting_layers
        self._pooling_strategy = pooling_strategy
        super().__init__(model_name)

    def load_model(self):
        # we disable logger info on the load of the _model
        original_verb = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            self._model = AutoModel.from_pretrained(self._name_model, output_hidden_states=True)

            transformers.logging.set_verbosity(original_verb)
            return self._model
        except (OSError, AttributeError):
            raise FileNotFoundError

    def get_vector_size(self) -> int:
        if isinstance(self._vec_strategy, CatStrategy):
            return self.model.embeddings.token_type_embeddings.embedding_dim * self._last_interesting_layers
        else:
            return self.model.embeddings.token_type_embeddings.embedding_dim

    def get_embedding(self, sentence: str) -> np.ndarray:
        token_vecs = self.get_embedding_token(sentence)
        sentence_vec = self._pooling_strategy.combine(token_vecs)
        return sentence_vec

    def get_embedding_token(self, sentence: str) -> np.ndarray:

        encoded = self._tokenizer.encode_plus(sentence)
        tokens_tensor = torch.tensor([encoded['input_ids']])
        segments_tensors = torch.tensor([encoded['attention_mask']])

        with torch.no_grad():
            model_output = self.model(tokens_tensor, segments_tensors)
            hidden_states = model_output[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_vecs = self._vec_strategy.build_embedding(token_embeddings)

        return token_vecs

    def get_sentence_token(self, sentence: str):
        """
        method that returns the tokenization of the sentence
        Args:
            sentence: sentence to analyze

        Returns: list containing tokens

        """
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.model.tokenize(marked_text)
        return tokenized_text

    def __str__(self):
        return "Transformers: _model = " + str(self.model)

    def __repr__(self):
        return str(self)
