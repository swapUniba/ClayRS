import unittest

import numpy as np
import torch
from transformers import BertModel, BertTokenizer, T5Model, AutoTokenizer

from clayrs.content_analyzer import Centroid, BertTransformers, T5Transformers
from clayrs.content_analyzer.embeddings import SumStrategy, CatStrategy
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    SingleToken


class TestBertTransformers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tiny_model = BertTransformers('prajjwal1/bert-tiny')

    def test_transformer(self):
        # test that the model loaded is the one passed
        transformers_model = self.tiny_model.model
        self.assertIsInstance(transformers_model, BertModel)

        # test load method
        source = self.tiny_model
        vector_size = source.get_vector_size()

        result = source.load(["this is a phrase", "this is another phrase"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(len(result[1]), vector_size)

    def test_embedding(self):
        # Test embedding obtained with framework and embedding obtained with hugging face library
        sentence = 'This is a beautiful model and very tiny'

        model = BertModel.from_pretrained('prajjwal1/bert-tiny', output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

        encoded = tokenizer.encode_plus(sentence)

        tokens_tensor = torch.tensor([encoded['input_ids']])
        segments_tensors = torch.tensor([encoded['attention_mask']])

        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            hidden_states = model_output[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # sentence embedding from model obtained with cat strategy and centroid
        token_vecs = CatStrategy(1).build_embedding(token_embeddings)
        expected = Centroid().combine(token_vecs)

        # sentence embedding from implemented class
        transformers_model = self.tiny_model
        result = transformers_model.get_embedding(sentence)

        self.assertTrue((expected == result).all())

        # sentence embedding from model obtained with cat strategy and centroid
        token_vecs = SumStrategy(2).build_embedding(token_embeddings)
        expected = SingleToken(0).combine(token_vecs)

        # sentence embedding from implemented class
        transformers_model = BertTransformers('prajjwal1/bert-tiny', vec_strategy=SumStrategy(2),
                                              pooling_strategy=SingleToken(0))
        result = transformers_model.get_embedding(sentence)

        self.assertTrue((expected == result).all())

    def test_embedding_token(self):
        transformers_model = self.tiny_model

        tok_1 = transformers_model.get_embedding_token('Hello, all right.')
        tok_2 = transformers_model.get_embedding_token('Hello, all right.')

        # tokens in same sentence should be equal
        self.assertEqual(tok_1.shape[1], transformers_model.get_vector_size())
        self.assertEqual(tok_2.shape[1], transformers_model.get_vector_size())

        self.assertTrue(np.array_equal(tok_1, tok_2))

        tok_1 = transformers_model.get_embedding_token('Hello how are you?')
        tok_2 = transformers_model.get_embedding_token('Hello, all right.')

        # same token ('Hello') in different sentences should be different
        self.assertEqual(tok_1.shape[1], transformers_model.get_vector_size())
        self.assertEqual(tok_2.shape[1], transformers_model.get_vector_size())

        self.assertFalse(np.array_equal(tok_1[1], tok_2[1]))


class TestT5Transformers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tiny_model = T5Transformers('t5-small')

    def test_transformer(self):
        # test that the model loaded is the one passed
        transformers_model = self.tiny_model.model
        self.assertIsInstance(transformers_model, T5Model)

        # test load method
        source = self.tiny_model
        vector_size = source.get_vector_size()

        result = source.load(["this is a phrase", "this is another phrase"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(len(result[1]), vector_size)

    def test_embedding(self):
        # Test embedding obtained with framework and embedding obtained with hugging face library
        sentence = 'This is a beautiful model and very tiny'

        model = T5Model.from_pretrained('t5-small', output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained('t5-small')

        encoded = tokenizer(sentence, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model.encoder(**encoded)
            hidden_states = model_output['hidden_states']

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # sentence embedding from model obtained with cat strategy and centroid
        token_vecs = CatStrategy(1).build_embedding(token_embeddings)
        expected = Centroid().combine(token_vecs)

        # sentence embedding from implemented class
        transformers_model = self.tiny_model
        result = transformers_model.get_embedding(sentence)

        self.assertTrue((expected == result).all())

        # sentence embedding from model obtained with cat strategy and centroid
        token_vecs = SumStrategy(2).build_embedding(token_embeddings)
        expected = SingleToken(0).combine(token_vecs)

        # sentence embedding from implemented class
        transformers_model = T5Transformers('t5-small', vec_strategy=SumStrategy(2),
                                            pooling_strategy=SingleToken(0))
        result = transformers_model.get_embedding(sentence)

        self.assertTrue((expected == result).all())

    def test_embedding_token(self):
        transformers_model = self.tiny_model

        tok_1 = transformers_model.get_embedding_token('Hello, all right.')
        tok_2 = transformers_model.get_embedding_token('Hello, all right.')

        # tokens in same sentence should be equal
        self.assertEqual(tok_1.shape[1], transformers_model.get_vector_size())
        self.assertEqual(tok_2.shape[1], transformers_model.get_vector_size())

        self.assertTrue(np.array_equal(tok_1, tok_2))

        tok_1 = transformers_model.get_embedding_token('Hello how are you?')
        tok_2 = transformers_model.get_embedding_token('Hello, all right.')

        # same token ('Hello') in different sentences should be different
        self.assertEqual(tok_1.shape[1], transformers_model.get_vector_size())
        self.assertEqual(tok_2.shape[1], transformers_model.get_vector_size())

        self.assertFalse(np.array_equal(tok_1[1], tok_2[1]))
