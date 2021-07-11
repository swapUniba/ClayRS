from unittest import TestCase
import pathlib as pl
import os

from orange_cb_recsys.content_analyzer import FieldConfig
from orange_cb_recsys.utils.const import root_path
from orange_cb_recsys.utils.runnable_instances import serialize_classes, get_classes

classes_path = os.path.join(root_path, "orange_cb_recsys/classes.xz")


class Test(TestCase):
    def test_serialize_get(self):
        serialize_classes()

        self.assertEqual(pl.Path(classes_path).is_file(), True)

        classes = get_classes()
        self.assertIsInstance(classes, dict)
        self.assertEqual(classes['fieldconfig'], FieldConfig)

        with self.assertRaises(KeyError):
            cls = classes['not_existing_class_name']
