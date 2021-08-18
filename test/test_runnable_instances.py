from unittest import TestCase
import pathlib as pl
import os
import lzma
import pickle

from orange_cb_recsys.content_analyzer import FieldConfig
from orange_cb_recsys.runnable_instances import serialize_classes, get_classes

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Test(TestCase):
    def test_serialize_get(self):
        serialize_classes(os.path.join(THIS_DIR, 'classes_dir'))

        classes_file_path = os.path.join(THIS_DIR, 'classes_dir/classes.xz')

        self.assertTrue(pl.Path(classes_file_path).is_file())

        with lzma.open(classes_file_path) as f:
            classes_from_file = pickle.load(f)

        classes = get_classes()
        self.assertIsInstance(classes, dict)
        self.assertIsInstance(classes_from_file, dict)
        self.assertEqual(classes['fieldconfig'], FieldConfig)
        self.assertEqual(classes_from_file['fieldconfig'], FieldConfig)

        with self.assertRaises(KeyError):
            cls = classes['not_existing_class_name']

        with self.assertRaises(KeyError):
            cls = classes_from_file['not_existing_class_name']
