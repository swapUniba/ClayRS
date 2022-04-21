import os
from unittest import TestCase
from clayrs.utils.load_content import load_content_instance
from test import dir_test_files

movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')


class Test(TestCase):
    def test_load_content_instance(self):
        self.assertIsNone(load_content_instance("not_existent", "invalid_item"))
        self.assertIsNotNone(load_content_instance(movies_dir, "tt0112281"))
