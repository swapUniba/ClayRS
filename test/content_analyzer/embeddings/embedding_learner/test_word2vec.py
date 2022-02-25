from unittest import TestCase
import os
import pathlib as pl

from orange_cb_recsys.content_analyzer.embeddings.embedding_learner import GensimWord2Vec
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

file_path = os.path.join(dir_test_files, 'movies_info_reduced.json')


class TestGensimWord2Vec(TestCase):
    def test_fit(self):
        model_path = "./model_test_Word2Vec"
        learner = GensimWord2Vec(model_path, True)
        learner.fit(source=JSONFile(file_path), field_list=["Plot", "Genre"], preprocessor_list=[NLTK()])
        model_path += ".bin"

        self.assertEqual(learner.get_embedding("ace").any(), True)
        self.assertEqual(pl.Path(model_path).resolve().is_file(), True)

