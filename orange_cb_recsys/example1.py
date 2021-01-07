# CONTENT ANALYZER
from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import DATFile
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset

from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.recsys.ranking_algorithms import NXPageRank

from orange_cb_recsys.evaluation.graph_metrics import nx_degree_centrality, nx_dispersion

from orange_cb_recsys.utils.feature_selection import NXFSPageRank

movies_filename = '/home/Mattia/Documents/ml-1m/movies.dat'
user_filename = '/home/Mattia/Documents/ml-1m/users.dat'
ratings_filename = '/home/Mattia/Documents/ml-1m/ratings.dat'

output_dir = '../../contents/test_1m_'

movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=DATFile(movies_filename),
    id_field_name_list=['0'],
    output_directory=output_dir
)

movies_ca_config.append_exogenous_properties_retrieval(
    DBPediaMappingTechnique(
        entity_type='Film',
        lang='EN',
        label_field='1'
    )
)

content_analyzer = ContentAnalyzer(movies_ca_config).fit()

users_ca_config = ContentAnalyzerConfig(
    content_type='User',
    source=DATFile(user_filename),
    id_field_name_list=['0'],
    output_directory=output_dir
)

users_ca_config.append_exogenous_properties_retrieval(
    PropertiesFromDataset()
)

content_analyzer.set_config(users_ca_config).fit()

ratings_import = RatingsImporter(
    source=DATFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(field_name='2', processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='0',
    to_field_name='1',
    timestamp_field_name='3'
).import_ratings()


full_graph = NXFullGraph(
    source_frame=ratings_import,
    contents_dir=output_dir,
    user_exogenous_properties=None,
    item_exogenous_properties=['director', 'protagonist', 'producer']
)


rank = NXPageRank(graph=full_graph).predict(
    user_id='1',
    ratings=ratings_import,
    recs_number=10,
    feature_selection_algorithm=NXFSPageRank()
)

print(nx_dispersion(full_graph))
print(nx_degree_centrality(full_graph))

