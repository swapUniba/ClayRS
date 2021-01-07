## Example 1
In this example we will discuss a simple configuration of the framework via API and configuration files.
The example shown is a simple example to become familiar with the framework. 
We want to use a recommender “CentroidVector” which, as the name suggests, works on the centroids of the vectors of the Items, in this case films.
We also want to initially compute a “Director” field representation of the film dataset; for the field just mentioned, the “SynsetDocumentFrequency” representation was chosen.

Import declaration:
```
from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.content_analyzer.config import FieldConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.field_content_production_techniques.synset_document_frequency import \
    SynsetDocumentFrequency
from orange_cb_recsys.recsys import CosineSimilarity
from orange_cb_recsys.recsys.recsys import RecSys, RecSysConfig
from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVector
```

and constants for directories containing the films (items), ratings and output_dir, 
where the representations created by the Content Analyzer will be stored
```
movies_filename = '../../../datasets/movies_info_reduced.json'
ratings_filename = '../../../datasets/ratings_example.json'

output_dir = '../../../contents/test_1m_easy'
```

## Part 1: content analyzer
The example is splitted in 2 parts: content analyzer and recommender. For the second part of the example you need to know how the system as called the directory which contains the output of the content analyzer because this sistem append a timestamp to the output_dir specified.
We instantiate the Config of the Content Analyzer, defining the variables necessary for the framework to work correctly. In this example, a movie dataset was chosen as ITEM (JSON file); the id_field_name_list field coincides with a list of one or more names that will represent the id of the content.
Finally, the output_dir directory created previously is chosen as output_directory.
```
movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=output_dir
)
```

The following section shows the configuration chosen for a field on which one or more representation techniques can be used; in this example, the "Director" field was chosen, with the "SynsetDocumentFrequency" representation technique, which is a technique that counts, in this case, the occurrences of the names of the directors within the entire dataset.
```
movies_ca_config.append_field_config(
    field_name='Director',         #tag
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=SynsetDocumentFrequency())]
    )
)
```

We can now instantiate the "ContentAnalyzer" object, to which the previously created configuration will be passed, and then train it using the 'fit ()' method.
```
content_analyzer_movies = ContentAnalyzer(
    config=movies_ca_config
)

content_analyzer_movies.fit()
```
## Part 2: recommender
Let's move on to the ratings part: in the next image you import the ratings from a dataset (always JSON file), applying the "NumberNormalizer" technique on the "stars" field.
The "RatingsImporter" object is instantiated, to which the dataset containing the ratings will be passed (via the 'source' parameter), the field name (in this case 'stars') on which to apply the representation technique (quest 'last will be specified in the' processor 'parameter); as mentioned above, in this example we choose the 'NumberNormalizer' technique, which normalizes the score of a rating to a value between -1 and 1. The parameters passed to the 'Number Normalizer' object, that is' min 'and' max ', represent respectively the minimum and maximum value of the score of the selected field (in this case, from 1 to 5 stars).
The 'from_field_name' and 'to_field_name' parameters represent the 'direction' of the ratings, or respectively Who made a rating and on What.
Finally, the 'import_ratings ()' method is called to import the ratings.
```
ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(
        field_name='stars',
        processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()
```

The "CentroidVector" object is instantiated, which will work on the previously selected Items. In the 'field_rappresentation' parameter, the position of the representation of the chosen item_field will be passed (in the following example you can see how there can be more representations). Finally, the "CosineSimilarity" object is passed (via the 'similarity' parameter), which allows you to perform the similarity of the cosine.
```
centroid_config = CentroidVector(
    item_field='Director',
    field_representation='0',
    similarity=CosineSimilarity()
)
```

The image shows how the "RecSysConfig" object is instantiated, to which the imported ratings will be passed, the "CentroidVector" object created in the previous image and the 'output_dir' directory.
```
centroid_recsys_config = RecSysConfig(
    users_directory='contents_dir',             # change with the current dir
    items_directory='contents_dir',             # change with the current dir
    ranking_algorithm=centroid_config,
    rating_frame=ratings_frame
)
```

Finally, the "Recsys" object will be created, an object that allows the framework recommending phase, to which the "RecSysConfig" object will be passed. Finally, the created object will be trained using the 'fit_ranking ()' method, in which the user to make recommendations and the number of recommendations to be output will be passed.
```
centroid_recommender = RecSys(
    config=centroid_recsys_config
)

ranking = centroid_recommender.fit_ranking(
    user_id='01',
    recs_number=2
)

print(ranking)
```
