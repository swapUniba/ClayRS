from unittest import TestCase

import os
import gensim

import numpy as np
from gensim.models import LsiModel

from clayrs.content_analyzer.embeddings.embedding_learner.latent_semantic_analysis import GensimLatentSemanticAnalysis
from gensim.corpora import Dictionary
from gensim.test.utils import common_texts

num_topics = 10
model_path = 'test_model_lda'


class TestLda(TestCase):
    def test_all(self):
        my_learner = GensimLatentSemanticAnalysis(model_path, num_topics=num_topics)

        corpus = common_texts
        my_learner.fit_model(corpus)

        # check that vector size is correct
        self.assertEqual(num_topics, my_learner.get_vector_size())

        common_dictionary = Dictionary(common_texts)
        common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
        expected_learner = LsiModel(common_corpus, num_topics=num_topics)

        # test get_embedding not existent document
        unseen_doc_text = ['this', 'is', 'a', 'new', 'document', 'which', 'doesnt', 'exist']

        # check that the doc is unseen (embedding has len 0)
        unseen_doc = common_dictionary.doc2bow(unseen_doc_text)
        expected = expected_learner[unseen_doc]
        self.assertTrue(len(expected) == 0)

        # in our framework if the doc is unseen KeyError is raised
        with self.assertRaises(KeyError):
            my_learner.get_embedding(unseen_doc_text)

        # test get_embedding existent document
        unseen_doc_text = ['human', 'time', 'trees']
        unseen_doc = common_dictionary.doc2bow(unseen_doc_text)
        expected = expected_learner[unseen_doc]
        expected_vector: np.ndarray = gensim.matutils.sparse2full(expected, num_topics)
        result_vector = my_learner.get_embedding(unseen_doc_text)

        # we don't have a way to check if the 2 vectors are the same, because they are build at random.
        # We just check that they are of the same length

        self.assertEqual(len(expected_vector), len(result_vector))

        # test save
        my_learner.save()
        self.assertTrue(os.path.isfile(f"{model_path}.model"))
        self.assertTrue(os.path.isfile(f"{model_path}.model.projection"))

        # test that after load we obtain a valid embedding
        my_learner_loaded = GensimLatentSemanticAnalysis(model_path)
        my_learner_loaded.load_model()
        unseen_doc_text = ['human', 'time', 'trees']
        result_vector = my_learner.get_embedding(unseen_doc_text)

        self.assertTrue(np.any(result_vector))

