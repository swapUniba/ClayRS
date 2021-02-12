from unittest import TestCase

from orange_cb_recsys.content_analyzer import ContentAnalyzerConfig, ContentAnalyzer
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import CSVFile, JSONFile, DATFile
from orange_cb_recsys.evaluation.graph_metrics import nx_dispersion, nx_degree_centrality
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.content_analyzer.content_representation.content_field import EmbeddingField


class TestNXFullGraph(TestCase):
    def test_all(self):

        ratings_filename = "datasets/examples/new_ratings.csv"
        movies_dir = 'contents/exo_prop/movielens_exo_1612956350.7812138'
        user_dir = 'contents/exo_prop/user_exo_1612956381.4652517'
        try:
            open(ratings_filename)
        except FileNotFoundError:
            ratings_filename = "../../../datasets/examples/new_ratings.csv"
            movies_dir = '../../../contents/exo_prop/movielens_exo_1612956350.7812138'
            user_dir = '../../../contents/exo_prop/user_exo_1612956381.4652517'


        # output_dir = 'test_1m_'
        #
        # movies_ca_config = ContentAnalyzerConfig(
        #     content_type='Item',
        #     source=JSONFile(movies_filename),
        #     id_field_name_list=['imdbID'],
        #     output_directory=output_dir
        # )
        #
        # movies_ca_config.append_exogenous_properties_retrieval(
        #     DBPediaMappingTechnique(
        #         entity_type='Film',
        #         lang='EN',
        #         label_field='Title'
        #     )
        # )
        #
        # ContentAnalyzer(movies_ca_config).fit()
        #
        # users_ca_config = ContentAnalyzerConfig(
        #     content_type='User',
        #     source=DATFile(user_filename),
        #     id_field_name_list=['0'],
        #     output_directory=output_dir
        # )
        #
        # users_ca_config.append_exogenous_properties_retrieval(
        #     PropertiesFromDataset()
        # )
        #
        # ContentAnalyzer(config=users_ca_config).fit()
        #content_analyzer.set_config(users_ca_config).fit()

        ratings_import = RatingsImporter(
            source=CSVFile(ratings_filename),
            rating_configs=[RatingsFieldConfig(
                field_name='points',
                processor=NumberNormalizer(min_=1, max_=5))],
            from_field_name='user_id',
            to_field_name='item_id',
            timestamp_field_name='timestamp',
        )

        ratings_frame = ratings_import.import_ratings()

        print(ratings_frame)

        full_graph = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_properties=['producer', 'starring']
        )

        self.assertFalse(full_graph.graph.has_node('000'))
        full_graph.add_node('000')
        self.assertTrue(full_graph.graph.has_node('000'))

        full_graph.add_edge('000', 'aaa', 0.5)
        result = full_graph.get_edge_data('000', 'aaa')
        expected = {'weight': 0.5, 'label': 'weight'}
        self.assertEqual(expected, result)

        adj = full_graph.get_adj('000')
        expected = ['aaa']
        result = []
        for x in adj:
            result.append(x)
        self.assertEqual(expected, result)

        prec = full_graph.get_predecessors('aaa')
        expected = ['000']
        result = []
        for x in prec:
            result.append(x)
        self.assertEqual(expected, result)

        full_graph.add_edge('000', 'bbb', 0.5)

        succ = full_graph.get_successors('000')
        expected = ['aaa', 'bbb']
        result = []
        for x in succ:
            result.append(x)
        self.assertEqual(expected, result)

        full_graph.add_tree('tt0112281')
        self.assertFalse(full_graph.is_exogenous_property('1'))
        self.assertFalse(full_graph.is_exogenous_property('tt0112281'))
        self.assertTrue(full_graph.is_exogenous_property('http://dbpedia.org/resource/Tom_Hanks'))

        # full_graph.get_voted_contents()
        # full_graph.get_properties()

