from unittest import TestCase

from orange_cb_recsys.content_analyzer.content_representation.content import Content, PropertiesDict, FeaturesBagField
from orange_cb_recsys.content_analyzer.content_representation.representation_container import RepresentationContainer


class TestContent(TestCase):
    def test_append_remove_field(self):
        """
        Tests for append, remove and get methods of the content's field instances
        """
        features_bag = dict()
        features_bag["test_key"] = "test_value"

        content_field_repr = FeaturesBagField(features_bag)
        content_field = RepresentationContainer()
        content_field.append(content_field_repr, "test_1")
        content1 = Content("001")
        content1.append_field("test_field", content_field)

        content2 = Content("002")
        content2.append_field("test_field", content_field)
        content_field_repr = FeaturesBagField(features_bag)
        content_field2 = RepresentationContainer()
        content_field2.append(content_field_repr, "test_1")
        content2.append_field("test_field2", content_field2)
        content2.remove_field("test_field2")
        self.assertEqual(content1.field_dict, content2.field_dict)
        self.assertEqual(content1.get_field("test_field"), content2.get_field("test_field"))

    def test_append_remove_field_repr(self):
        """
        Tests for append, remove and get methods of the content's fields' representation instances
        """
        features_bag = dict()
        features_bag["test_key"] = "test_value"

        content_field_repr_1 = FeaturesBagField(features_bag)
        content_field = RepresentationContainer(content_field_repr_1, "test_1")
        content = Content("001")
        content.append_field("test_field", content_field)

        content_field_repr_2 = FeaturesBagField(features_bag)
        content.append_field_representation("test_field_2", content_field_repr_2, "test_2")
        self.assertEqual(content.get_field_representation("test_field_2", "test_2"), content_field_repr_2)
        self.assertEqual(len(content.get_field("test_field_2")), 1)

        content.remove_field_representation("test_field_2", "test_2")
        self.assertEqual(len(content.get_field("test_field_2")), 0)

    def test_append_remove_exo(self):
        """
        Tests for append, remove and get methods of the content's exogenous instances
        """
        exo_features = dict()
        exo_features["test_key"] = "test_value"

        content_exo_repr = PropertiesDict(exo_features)
        content1 = Content("001")
        content1.append_exogenous(content_exo_repr, "test_exo")

        content2 = Content("002")
        content2.append_exogenous(content_exo_repr, "test_exo")
        content_exo_repr = PropertiesDict(exo_features)
        content2.append_exogenous(content_exo_repr, "test_exo2")
        content2.remove_exogenous("test_exo2")
        self.assertEqual(content1.exogenous_rep_container, content2.exogenous_rep_container)
        self.assertEqual(content1.get_exogenous("test_exo"), content2.get_exogenous("test_exo"))
