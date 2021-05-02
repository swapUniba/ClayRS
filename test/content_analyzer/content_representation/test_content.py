from unittest import TestCase

from orange_cb_recsys.content_analyzer.content_representation.content import Content, PropertiesDict
from orange_cb_recsys.content_analyzer.content_representation.content_field import FeaturesBagField, ContentField


class TestContent(TestCase):
    """
    Test for adding and removing a field and an exogenous representation
    """
    def test_append_remove_field(self):
        features_bag = dict()
        features_bag["test_key"] = "test_value"

        content_field_repr = FeaturesBagField(features_bag)
        content_field = ContentField()
        content_field.append(str(0), content_field_repr)
        content1 = Content("001")
        content1.append_field("test_field", content_field)

        content2 = Content("002")
        content2.append_field("test_field", content_field)
        content_field_repr = FeaturesBagField(features_bag)
        content_field2 = ContentField()
        content_field2.append(str(0), content_field_repr)
        content2.append_field("test_field2", content_field2)
        content2.remove_field("test_field2")
        self.assertTrue(content1.field_dict, content2.field_dict)

    def test_append_remove_exo(self):
        exo_features = dict()
        exo_features["test_key"] = "test_value"

        content_exo_repr = PropertiesDict(exo_features)
        content1 = Content("001")
        content1.append_exogenous("test_exo", content_exo_repr)

        content2 = Content("002")
        content2.append_exogenous("test_exo", content_exo_repr)
        content_exo_repr = PropertiesDict(exo_features)
        content2.append_exogenous("test_exo2", content_exo_repr)
        content2.remove_exogenous("test_exo2")
        self.assertTrue(content1.exogenous_rep_dict, content2.exogenous_rep_dict)