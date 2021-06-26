from unittest import TestCase

from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import PropertiesFromDataset

from orange_cb_recsys.content_analyzer import FieldConfig, ExogenousConfig


class TestFieldConfig(TestCase):
    def test_invalid_id(self):
        with self.assertRaises(ValueError):
            FieldConfig(id='.in.vali.d')

        with self.assertRaises(ValueError):
            FieldConfig(id='#in#vali#d')

        with self.assertRaises(ValueError):
            FieldConfig(id='     ')

        with self.assertRaises(ValueError):
            FieldConfig(id='is invalid')

        with self.assertRaises(ValueError):
            FieldConfig(id='is/inva/lid')

        # ...and many more

    def test_valid_id(self):
        valid_object = FieldConfig(id='test')
        self.assertIsNotNone(valid_object)

        valid_object = FieldConfig(id='test_valid')
        self.assertIsNotNone(valid_object)

        valid_object = FieldConfig(id='test-valid')
        self.assertIsNotNone(valid_object)

        valid_object = FieldConfig(id='test1-valid2')
        self.assertIsNotNone(valid_object)

        valid_object = FieldConfig(id='1_2-3_')
        self.assertIsNotNone(valid_object)

        # ...and many more


class TestExogenousConfig(TestCase):
    def test_invalid_id(self):
        with self.assertRaises(ValueError):
            ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='.in.vali.d')

        with self.assertRaises(ValueError):
            ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='.in.vali.d')

        with self.assertRaises(ValueError):
            ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='.in.vali.d')

        with self.assertRaises(ValueError):
            ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='.in.vali.d')

        with self.assertRaises(ValueError):
            ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='.in.vali.d')

        # ...and many more

    def test_valid_id(self):
        valid_object = ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='test')
        self.assertIsNotNone(valid_object)

        valid_object = ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='test_valid')
        self.assertIsNotNone(valid_object)

        valid_object = ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='test-valid')
        self.assertIsNotNone(valid_object)

        valid_object = ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='test1-valid2')
        self.assertIsNotNone(valid_object)

        valid_object = ExogenousConfig(exogenous_technique=PropertiesFromDataset(), id='1_2-3_')
        self.assertIsNotNone(valid_object)

        # ...and many more
