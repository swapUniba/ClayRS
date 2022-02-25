from unittest import TestCase

import os
import pathlib as pl

from orange_cb_recsys.content_analyzer.embeddings.embedding_learner.random_indexing import GensimRandomIndexing
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

file_path = os.path.join(dir_test_files, 'movies_info_reduced.json')


class TestRandomIndexing(TestCase):
    def test_fit(self):
        model_path = "./model_test_Ri"
        learner = GensimRandomIndexing(model_path, True)
        learner.fit(source=JSONFile(file_path), field_list=["Plot", "Genre"], preprocessor_list=[NLTK()])
        model_path += ".model"

        self.assertEqual(learner.get_embedding("ace").any(), True)
        self.assertEqual(pl.Path(model_path).resolve().is_file(), True)
