from orange_cb_recsys.content_analyzer import ContentAnalyzerConfig, FieldRepresentationPipeline, FieldConfig, \
    ContentAnalyzer
from orange_cb_recsys.content_analyzer.field_content_production_techniques import LuceneTfIdf, BabelPyEntityLinking, \
    EmbeddingTechnique, Centroid, Wikipedia2VecDownloader, GensimDownloader
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

movies_dataset = '../../../datasets/movie_info_reduced.json'
users_dataset = '../../../datasets/users_info_.json'

movies_dir = 'movies_dir'
users_dir = 'Users_Example'

users_config = ContentAnalyzerConfig(
    content_type='User',
    source=JSONFile(users_dataset),
    id_field_name_list=['user_id'],
    output_directory=users_dir,
)

ContentAnalyzer(config=users_config).fit()
