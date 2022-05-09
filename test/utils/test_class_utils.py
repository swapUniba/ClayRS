from unittest import TestCase

from clayrs.content_analyzer.config import ContentAnalyzerConfig, FieldConfig, ItemAnalyzerConfig, \
    UserAnalyzerConfig
from clayrs.utils.class_utils import get_all_implemented_classes, get_all_implemented_subclasses


class TestClassUtils(TestCase):

    def test_get_all_implemented_classes(self):

        results = get_all_implemented_classes(ContentAnalyzerConfig)

        expected_results = {ItemAnalyzerConfig, UserAnalyzerConfig}
        self.assertEqual(results, expected_results)

        results = get_all_implemented_classes(FieldConfig)

        expected_results = {FieldConfig}
        self.assertEqual(results, expected_results)

    def test_get_all_implemented_subclasses(self):

        results = get_all_implemented_subclasses(ContentAnalyzerConfig)

        expected_results = {ItemAnalyzerConfig, UserAnalyzerConfig}
        self.assertEqual(results, expected_results)

        results = get_all_implemented_subclasses(FieldConfig)
        expected_results = set()

        self.assertEqual(results, expected_results)