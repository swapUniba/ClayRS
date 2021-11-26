from unittest import TestCase

import os
import pathlib as pl

from orange_cb_recsys.content_analyzer.embeddings.embedding_learner.latent_semantic_analysis import GensimLatentSemanticAnalysis
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dir_test_files, 'movies_info_reduced.json')


class TestGensimLatentSemanticAnalysis(TestCase):
    def test_fit(self):
        model_path = os.path.join(THIS_DIR, "/model_test_Lsa")
        learner = GensimLatentSemanticAnalysis(model_path, True)
        learner.fit(source=JSONFile(file_path), field_list=["Plot", "Genre"], preprocessor_list=[NLTK()])
        model_path += ".model"

        self.assertEqual(learner.get_embedding("ace").any(), True)
        self.assertEqual(pl.Path(model_path).resolve().is_file(), True)

