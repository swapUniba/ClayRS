from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from clayrs.content_analyzer.embeddings.embedding_loader.vector_strategy import VectorStrategy
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique \
        import CombiningTechnique

from clayrs.content_analyzer.embeddings.embedding_loader.embedding_loader import SentenceEmbeddingLoader
from clayrs.content_analyzer.embeddings.embedding_loader.vector_strategy import CatStrategy
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import Centroid


class Transformers(SentenceEmbeddingLoader):
    """
    Abstract class for Transformers
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
            return self.model.config.hidden_size * self._last_interesting_layers
        else:
            return self.model.config.hidden_size

    def get_embedding(self, sentence: str) -> np.ndarray:
        token_vecs = self.get_embedding_token(sentence)
        sentence_vec = self._pooling_strategy.combine(token_vecs)
        return sentence_vec

    @abstractmethod
    def get_embedding_token(self, sentence: str) -> np.ndarray:
        raise NotImplementedError

    def get_sentence_token(self, sentence: str):
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.model.tokenize(marked_text)
        return tokenized_text

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class BertTransformers(Transformers):
    """
    Class that produces sentences/token embeddings using any Bert model from hugging face.

    Args:
        model_name: Name of the embeddings model to download or path where the model is stored locally
        vec_strategy: Strategy which will be used to combine each output layer to obtain a single one
        pooling_strategy: Strategy which will be used to combine the embedding representation of each token into a
            single one, representing the embedding of the whole sentence
    """
    def __init__(self, model_name: str = 'bert-base-uncased',
                 vec_strategy: VectorStrategy = CatStrategy(1),
                 pooling_strategy: CombiningTechnique = Centroid()):
        super().__init__(model_name, vec_strategy, pooling_strategy)

    def get_embedding_token(self, sentence: str) -> np.ndarray:
        encoded = self._tokenizer(sentence, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded)
            hidden_states = model_output['hidden_states']

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_vecs = self._vec_strategy.build_embedding(token_embeddings)

        return token_vecs

    def __str__(self):
        return "BertTransformers"

    def __repr__(self):
        return f"BertTransformers(model_name={self._name_model}, " \
               f"vec_strategy={self._vec_strategy}, " \
               f"pooling_strategy={self._pooling_strategy})"


class T5Transformers(Transformers):
    """
    Class that produces sentences/token embeddings using sbert.

    Args:
        model_name: Name of the embeddings model to download or path where the model is stored locally
        vec_strategy: Strategy which will be used to combine each output layer to obtain a single one
        pooling_strategy: Strategy which will be used to combine the embedding representation of each token into a
            single one, representing the embedding of the whole sentence
    """
    def __init__(self, model_name: str = 't5-small',
                 vec_strategy: VectorStrategy = CatStrategy(1),
                 pooling_strategy: CombiningTechnique = Centroid()):
        super().__init__(model_name, vec_strategy, pooling_strategy)

    def get_embedding_token(self, sentence: str) -> np.ndarray:
        encoded = self._tokenizer(sentence, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model.encoder(**encoded)
            hidden_states = model_output['hidden_states']

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_vecs = self._vec_strategy.build_embedding(token_embeddings)

        return token_vecs

    def __str__(self):
        return "T5Transformers"

    def __repr__(self):
        return f"T5Transformers(model_name={self._name_model}, " \
               f"vec_strategy={self._vec_strategy}, " \
               f"pooling_strategy={self._pooling_strategy})"
