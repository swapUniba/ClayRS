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

        ratings_filename = "contents/exo_prop/new_ratings_small.csv"
        movies_dir = 'contents/exo_prop/movielens_exo_1612956350.7812138'
        user_dir = 'contents/exo_prop/user_exo_1612956381.4652517'
        try:
            open(ratings_filename)
        except FileNotFoundError:
            ratings_filename = "../../../contents/exo_prop/new_ratings_small.csv"
            movies_dir = '../../../contents/exo_prop/movielens_exo_1612956350.7812138'
            user_dir = '../../../contents/exo_prop/user_exo_1612956381.4652517'

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

        # Create graph with those properties from that representation
        # EX. create graph with properties 'producer' and 'starring'
        # from representation 0
        full_graph = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_representation="0",
            user_exo_representation="0",
            item_exo_properties=['producer', 'starring'],
            user_exo_properties=['1'] #It's the column in the users .DAT which identifies the gender
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

        full_graph.add_tree('tt0114885')
        self.assertFalse(full_graph.is_exogenous_property('1'))
        self.assertFalse(full_graph.is_exogenous_property('tt0114885'))
        self.assertTrue(full_graph.is_exogenous_property('http://dbpedia.org/resource/Tom_Hanks'))
        self.assertTrue(full_graph.is_exogenous_property('F'))

        # full_graph.get_voted_contents()
        # full_graph.get_properties()


        # Create graph without setting particular representation,
        # EX. Create graph with properties 'producer' and 'starring' from
        # all exo representation, since there can be multiple exo representation
        # containing the same properties
        full_graph = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_properties=['producer', 'starring'],
            user_exo_properties=['1'] #It's the column in the users DAT which identifies the gender
        )

        # simple assert just to make sure the graph is created
        self.assertGreater(full_graph.graph.__len__(), 0)

        # Create graph without setting particular properties,
        # so ALL exo properties of the representation 0 will be retrieved
        full_graph = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_representation="0",
            user_exo_representation="0"
        )

        # simple assert just to make sure the graph is created
        self.assertGreater(full_graph.graph.__len__(), 0)