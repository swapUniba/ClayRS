from unittest import TestCase

import os

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.test.utils import common_texts
import gensim

from clayrs.content_analyzer.embeddings.embedding_learner.lda import GensimLDA

# we fix random_state for reproducibility
random_state = 42
num_topics = 100
model_path = 'test_model_lda'


class TestLda(TestCase):
    def test_all(self):
        my_learner = GensimLDA(model_path, num_topics=num_topics, random_state=random_state)

        corpus = common_texts
        my_learner.fit_model(corpus)

        # check that vector size is correct
        self.assertEqual(num_topics, my_learner.get_vector_size())

        common_dictionary = Dictionary(common_texts)
        common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
        expected_learner = LdaModel(common_corpus, num_topics=num_topics, random_state=random_state)

        # test get_embedding not existent document
        unseen_doc_text = ['this', 'is', 'a', 'new', 'document', 'which', 'doesnt', 'exist']
        unseen_doc = common_dictionary.doc2bow(unseen_doc_text)
        expected = expected_learner[unseen_doc]
        expected_vector: np.ndarray = gensim.matutils.sparse2full(expected, num_topics)

        result_vector = my_learner.get_embedding(unseen_doc_text)

        self.assertTrue(np.array_equal(expected_vector, result_vector))

        # test get_embedding existent document
        unseen_doc_text = ['human', 'time', 'trees']
        unseen_doc = common_dictionary.doc2bow(unseen_doc_text)
        expected = expected_learner[unseen_doc]
        expected_vector: np.ndarray = gensim.matutils.sparse2full(expected, num_topics)

        result_vector = my_learner.get_embedding(unseen_doc_text)

        self.assertTrue(np.array_equal(expected_vector, result_vector))

        # test save
        my_learner.save()
        self.assertTrue(os.path.isfile(f"{model_path}.model"))
        self.assertTrue(os.path.isfile(f"{model_path}.model.expElogbeta.npy"))
        self.assertTrue(os.path.isfile(f"{model_path}.model.id2word"))
        self.assertTrue(os.path.isfile(f"{model_path}.model.state"))

        # test that after load we obtain a valid embedding
        my_learner_loaded = GensimLDA(model_path)
        my_learner_loaded.load_model()
        unseen_doc_text = ['human', 'time', 'trees']
        result_vector = my_learner.get_embedding(unseen_doc_text)

        self.assertTrue(np.any(result_vector))
