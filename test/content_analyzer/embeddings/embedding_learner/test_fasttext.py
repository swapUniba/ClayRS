from unittest import TestCase
import os
import pathlib as pl

from clayrs.content_analyzer.embeddings.embedding_learner import GensimFastText
from clayrs.content_analyzer.information_processor.nltk_processor import NLTK
from clayrs.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

file_path = os.path.join(dir_test_files, 'movies_info_reduced.json')


class TestGensimFastText(TestCase):
    def test_fit(self):
        model_path = "./model_test_FastText"
        learner = GensimFastText(model_path, True)
        learner.fit(source=JSONFile(file_path), field_list=["Plot", "Genre"], preprocessor_list=[NLTK()])
        model_path += ".kv"

        self.assertEqual(learner.get_embedding("ace").any(), True)
        self.assertEqual(pl.Path(model_path).resolve().is_file(), True)


