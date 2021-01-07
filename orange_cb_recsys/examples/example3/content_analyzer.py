from orange_cb_recsys.content_analyzer import ContentAnalyzerConfig, ContentAnalyzer
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

movies_filename = '../../../datasets/examples/movies_info_reduced.json'

output_dir_movies = '../../../contents/examples/ex_3/movies_'

movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=output_dir_movies
)


movies_ca_config.append_exogenous_properties_retrieval(
    DBPediaMappingTechnique(
        entity_type='Film',
        lang='EN',
        label_field='Title'
    )
)


content_analyzer = ContentAnalyzer(movies_ca_config).fit()
