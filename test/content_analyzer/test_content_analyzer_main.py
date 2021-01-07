from unittest import TestCase

from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig, FieldConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.field_content_production_techniques.entity_linking import BabelPyEntityLinking
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestContentsProducer(TestCase):
    def test_create_content(self):
        filepath = '../../datasets/movies_info_reduced.json'
        try:
            with open(filepath):
                pass
        except FileNotFoundError:
            filepath = 'datasets/movies_info_reduced.json'

        entity_linking_pipeline = FieldRepresentationPipeline(BabelPyEntityLinking())
        plot_config = FieldConfig(None)
        plot_config.append_pipeline(entity_linking_pipeline)
        content_analyzer_config = ContentAnalyzerConfig('ITEM', JSONFile(filepath), ["imdbID"], "movielens_test")
        content_analyzer_config.append_field_config("Plot", plot_config)
        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()
