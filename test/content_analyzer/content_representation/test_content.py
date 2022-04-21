from unittest import TestCase

import numpy as np

from clayrs.content_analyzer.content_representation.content import Content, PropertiesDict, FeaturesBagField
from clayrs.content_analyzer.content_representation.representation_container import RepresentationContainer


class TestContent(TestCase):
    def test_append_remove_field(self):
        """
        Tests for append, remove and get methods of the content's field instances
        """
        features_bag_vector = np.array([1.0, 0])
        features_bag_pos_features = [(0, 'word_present'), (1, 'word_not_present')]

        content_field_repr = FeaturesBagField(features_bag_vector, features_bag_pos_features)
        content_field = RepresentationContainer()
        content_field.append(content_field_repr, "test_1")
        content1 = Content("001")
        content1.append_field("test_field", content_field)

        content2 = Content("002")
        content2.append_field("test_field", content_field)
        content_field_repr = FeaturesBagField(features_bag_vector, features_bag_pos_features)
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
        features_bag_vector = np.array([1.0, 0])
        features_bag_pos_features = [(0, 'word_present'), (1, 'word_not_present')]

        # test append_field_repr when field already existent
        content_field_repr_1 = FeaturesBagField(features_bag_vector, features_bag_pos_features)
        content_field = RepresentationContainer(content_field_repr_1, "test_1")
        content = Content("001")
        content.append_field("test_field", content_field)

        # test append_field_repr when field not existent
        content_field_repr_2 = FeaturesBagField(features_bag_vector, features_bag_pos_features)
        content.append_field_representation("test_field_2", content_field_repr_2, "test_2")
        self.assertEqual(content.get_field_representation("test_field_2", "test_2"), content_field_repr_2)
        self.assertEqual(len(content.get_field("test_field_2")), 1)

        # test append_field_repr with list of repr
        content_field_repr_3_first = FeaturesBagField(features_bag_vector, features_bag_pos_features)
        content_field_repr_3_second = FeaturesBagField(features_bag_vector, features_bag_pos_features)
        content.append_field_representation("test_field_3", [content_field_repr_3_first, content_field_repr_3_second],
                                            ["test_3_first", "test_3_second"])
        self.assertEqual(content.get_field_representation("test_field_3", "test_3_first"), content_field_repr_3_first)
        self.assertEqual(content.get_field_representation("test_field_3", "test_3_second"), content_field_repr_3_second)
        self.assertEqual(len(content.get_field("test_field_3")), 2)

        # test append_field_repr with list of repr and no id specified
        content_field_repr_4_first = FeaturesBagField(features_bag_vector, features_bag_pos_features)
        content_field_repr_4_second = FeaturesBagField(features_bag_vector, features_bag_pos_features)
        content.append_field_representation("test_field_4", [content_field_repr_4_first, content_field_repr_4_second])
        self.assertEqual(content.get_field_representation("test_field_4", 0), content_field_repr_4_first)
        self.assertEqual(content.get_field_representation("test_field_4", 1), content_field_repr_4_second)
        self.assertEqual(len(content.get_field("test_field_4")), 2)

        # test remove
        content.remove_field_representation("test_field_2", "test_2")
        self.assertEqual(len(content.get_field("test_field_2")), 0)

    def test_append_remove_exo(self):
        """
        Tests for append, remove and get methods of the content's exogenous instances
        """
        exo_features = dict()
        exo_features["test_key"] = "test_value"

        content_exo_repr = PropertiesDict(exo_features)
        content_exo_repr2 = PropertiesDict({"test_key2": 'test_value2'})

        content1 = Content("001")
        content1.append_exogenous_representation(content_exo_repr, "test_exo")

        content2 = Content("002")
        content2.append_exogenous_representation(content_exo_repr, "test_exo")
        content_exo_repr = PropertiesDict(exo_features)
        content2.append_exogenous_representation(content_exo_repr, "test_exo2")
        content2.remove_exogenous_representation("test_exo2")
        self.assertEqual(content1.exogenous_rep_container, content2.exogenous_rep_container)
        self.assertEqual(content1.get_exogenous_representation("test_exo"),
                         content2.get_exogenous_representation("test_exo"))

        # test append list of representations
        content3 = Content("003")

        content3.append_exogenous_representation([content_exo_repr, content_exo_repr2], ["id1", "id2"])
        self.assertEqual(len(content3.exogenous_rep_container), 2)
        self.assertEqual(content3.get_exogenous_representation("id1").value, content_exo_repr.value)
        self.assertEqual(content3.get_exogenous_representation("id2").value, content_exo_repr2.value)

        # test append list of representations without id
        content4 = Content("004")
        content4.append_exogenous_representation([content_exo_repr, content_exo_repr2])
        self.assertEqual(len(content3.exogenous_rep_container), 2)
        self.assertEqual(content3.get_exogenous_representation(0).value, content_exo_repr.value)
        self.assertEqual(content3.get_exogenous_representation(1).value, content_exo_repr2.value)
