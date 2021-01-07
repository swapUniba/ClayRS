from orange_cb_recsys.content_analyzer import ContentAnalyzerConfig, FieldRepresentationPipeline, FieldConfig, \
    ContentAnalyzer
from orange_cb_recsys.content_analyzer.field_content_production_techniques import LuceneTfIdf
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, DATFile

import lucene

lucene.initVM(vmargs=['-Djava.awt.headless=true'])

movies_filename = '../../../datasets/examples/movies_info.json'

movies_output_dir = '../../../contents/examples/ex_1/movies_'

users_filename = '../../../datasets/examples/users_70.dat'

users_output_dir = '../../../contents/examples/ex_1/users_'

movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=movies_output_dir,

)


movies_ca_config.append_field_config(
    field_name='Title',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=LuceneTfIdf())]
    )
)


movies_ca_config.append_field_config(
    field_name='Year',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=LuceneTfIdf())]
    )
)


movies_ca_config.append_field_config(
    field_name='Genre',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            preprocessor_list=[NLTK(lemmatization=True, stopwords_removal=True)],
            content_technique=LuceneTfIdf())]
    )
)


movies_ca_config.append_field_config(
    field_name='Plot',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            preprocessor_list=[NLTK(lemmatization=True, stopwords_removal=True)],
            content_technique=LuceneTfIdf())]
    )
)

ContentAnalyzer(config=movies_ca_config).fit()


users_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=DATFile(users_filename),
    id_field_name_list=['0'],
    output_directory=users_output_dir,

)

ContentAnalyzer(config=users_ca_config).fit()


# esempio di lettura del contenuto


