## Example 2

We can evolve the previous example by increasing its complexity. We also test users with different techniques, also using field preprocessors. As for the ratings we will see how the score combiner works on fields of a different nature. In the recommending phase we will use a random forest classifier.

The imports are as follows:
```
from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.config import FieldConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.field_content_production_techniques import SearchIndexing, LuceneTfIdf, \
    BabelPyEntityLinking
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.recsys import ClassifierRecommender
from orange_cb_recsys.recsys.recsys import RecSys, RecSysConfig
```

then we define the constants of the paths:
```
movies_filename = '../../../datasets/movies_info_reduced.json'
user_filename = '../../../datasets/users_info_.json'
ratings_filename = '../../../datasets/ratings_example.json'

output_dir = '../../../contents/test_1m_medium'
```

## Part 1: content analyzer

We instantiate a ContentAnalyzerConfig object as seen above. The "search_index" flag is validated to start the search index that will be needed later.
```
movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID', 'Title'],
    output_directory=output_dir,
    search_index=True
)
```

The search index is used as one of the two representations of the field Plot (description). The other representation is given by the calculation of the tf-idf through the Apache Lucene system, NLTK instead is the preprocessor chosen to perform the lemmatization.
```
movies_ca_config.append_field_config(
    field_name='Plot',
    field_config=FieldConfig(
        pipelines_list=[
            FieldRepresentationPipeline(content_technique=SearchIndexing()),
            FieldRepresentationPipeline(preprocessor_list=[NLTK(lemmatization=True, lang="english")],
                                        content_technique=LuceneTfIdf())]
    )
)
```

We further specify that the "Actor" field must be processed with the EntityLinking technique via the BabelFy platform with the BabelPyEntityLinking class.
```
movies_ca_config.append_field_config(
    field_name='Actors',
    field_config=FieldConfig(
        pipelines_list=[
            FieldRepresentationPipeline(content_technique=BabelPyEntityLinking())]
    )
)
```

We instantiate the C.A. for the films just described:
```
content_analyzer_movies = ContentAnalyzer(
    config=movies_ca_config
)
```

Now let's move on to configuring users:
```
user_ca_config = ContentAnalyzerConfig(
    content_type='User',
    source=JSONFile(user_filename),
    id_field_name_list=["user_id"],
    output_directory=output_dir
)
```

We choose for the field "name" a representation processed only through an NL preprocessing where URLs and multiple spaces are eventually eliminated
```
user_ca_config.append_field_config(
    field_name="name",
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            preprocessor_list=NLTK(url_tagging=True, strip_multiple_whitespaces=True), content_technique=None)]
    )
)
```

We instantiate the users' content analyzer and launch the fit method on both C.A.
```
content_analyzer_users = ContentAnalyzer(
    config=user_ca_config
)

content_analyzer_movies.fit()
content_analyzer_users.fit()
```

## Part 2: recommender

Let's move on to the ratings part: We instantiate two RatingsFieldConfig, one that carries out the Sentiment Analysis on the title field of the review and the other on the rating (stars) as in the previous example.
```
title_review_config = RatingsFieldConfig(
    field_name='review_title',
    processor=TextBlobSentimentAnalysis()
)

starts_review_config = RatingsFieldConfig(
    field_name='stars',
    processor=NumberNormalizer(min_=1, max_=5))
```

We then instantiate the ratings Importer which returns the ratings frame using the "import_ratings ()" method.
```
ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[title_review_config, starts_review_config],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()
```

Let's move on to the recommending part: we instantiate a classifier with the random forest technique on the tf-idf representation of the field Plot of the item films.
```
classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='1',
    classifier='random_forest'
)
```

The newly created config is then passed to the RecSysConfig.
```
classifier_recsys_config = RecSysConfig(
    users_directory='contents_dir',                 # change with the current dir
    items_directory='contents_dir',                 # change with the current dir 
    ranking_algorithm=classifier_config,
    rating_frame=ratings_frame
)
```

In turn used to instantiate the recommender. We then call the "fit_ranking" method on user 1.
```
classifier_recommender = RecSys(
    config=classifier_recsys_config
)

rank = classifier_recommender.fit_ranking(
    user_id='01',
    recs_number=10
)

print(rank)
```

