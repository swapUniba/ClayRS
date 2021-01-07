# Example 4
In this example we will see how to use the exogenous properties retrieval module and a graph based recommender using 'Page Rank with priors' as algorithm for both recommendation and feature selection on the graph.
NB: The exogenous retrieval module can be used also without a graph based recommender. In this case we will use the properties founded in DBPedia to enrich the graph.
The example is splitted in two parts,in the first half we will see how manage the content analyzer module with the DBPedia Mapping technique to find (exogenous) properties; in the second half, to create the recommender, we need the directory in which the contents processed by the content analyzer module are stored.
NB: local timestamp is added to the output dir specified for the content analizer

First of all we need these modules:
```
from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import DATFile, JSONFile
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset

from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.recsys.ranking_algorithms import NXPageRank
from orange_cb_recsys.utils.feature_selection import NXFSPageRank

from orange_cb_recsys.evaluation.graph_metrics import nx_degree_centrality, nx_dispersion
```

## Part 1: content analyzer and exogenous content retrieval

So we can define the input and output dir of content analyzer
```
movies_filename = '../../../datasets/movies_info_reduced.json'
user_filename = '../../../datasets/users_info_.json'
ratings_filename = '../../../datasets/ratings_example.json'


output_dir_movies = '../../../contents/test_1m_ex_4/movies_'
output_dir_users = '../../../contents/test_1m_ex_4/users_'
```

and define a config for the films, same of the previus examples:
```
movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=output_dir_movies
)
```

but now we introduce the exogenous properties retrieval module: we will use the DBPediaMappingTechnique to find some properties in DBPedia. Entity type is the type of information that we want to find, in this case movies or film; lang is the language of the contents that are already in the our datasets and represent also the language of the (eventually) retrieved properties; Label field is the field used as 'search key' in the DBPedia cloud, in this case Title should be good enough. If we want to specify other filters we can add a list of label_field in "additional_filters". Also we can specify the mode used to retrieve the properties:

- only_retrieved_evaluated (default): only the retrieved properties that have a value are added to the item.
- all: all properties, contains retrieved evaluated and non-evaluated. Also contains the properties in the original dict.
- all_retrieved: only properties retrieved, evaluated and non-evaluated.
- original_retrieved: filter the retrieved properties with the keys already in the dataset.
```
movies_ca_config.append_exogenous_properties_retrieval(
    DBPediaMappingTechnique(
        entity_type='Film',
        lang='EN',
        label_field='Title'
    )
)

content_analyzer = ContentAnalyzer(movies_ca_config).fit()
```

Let's move on the configuration of the users, in this case we don't want to use external properties for enrich th graph, but we add the properties aready in the Dataset.

```
users_ca_config = ContentAnalyzerConfig(
    content_type='User',
    source=JSONFile(user_filename),
    id_field_name_list=['user_id'],
    output_directory=output_dir_users
)

users_ca_config.append_exogenous_properties_retrieval(
    PropertiesFromDataset()
)

content_analyzer = ContentAnalyzer(users_ca_config).fit()
```

## Part 2: Page rank recommender 
Now, let's create ratings dataframe for the graph creation
```
ratings_import = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(field_name='stars', processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp'
).import_ratings()
```

We define the graph with these properties, as we can see only the drector, protagonist and producer properties, which are all items properties, are added to the graph. We can use this graph to do some topological analysis stuff, like analizing the centrality degree:
```
full_graph = NXFullGraph(
    source_frame=ratings_import,
    user_content_dir = '',        # to change
    item_content_dir = '',        # to change
    user_exogenous_properties=None,
    item_exogenous_properties=['director', 'protagonist', 'producer']
)

print(nx_degree_centrality(full_graph))
```

Now we can finally call the Page rank algorithm on the user '01'. We can use also a Feature selection algorithm, in this case the same page rank algorithm is used to "thin out" the graph.
```
rank = NXPageRank(graph=full_graph).predict(
    user_id='01',
    ratings=ratings_import,
    recs_number=10,
    feature_selection_algorithm=NXFSPageRank()
)
```
You can show the dict rank by simply printing it:
```
print(rank)
```
