!!! warning

    Docs are complete, but revision is still a Work in Progress. Sorry for any typos!

# Introduction

The Content Analyzer module has the task to build a complex representation for chosen contents, starting from their
***raw*** representation

The following will introduce you to the standard usage pipeline for this module, showing you all the various operations
that can be performed

## Item Config

Suppose the following **JSON** file which contains information about movies: It will act as *raw source for items*.

```title="JSON items raw source"
[
    {
        "movielens_id": "1",
        "title": "Toy Story",
        "plot": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy"
        "genres": "Animation, Adventure, Comedy, Family, Fantasy",
        "year": "1995",
        "rating": "8.3",
        "directors": "John Lasseter",
        "dbpedia_uri": "http://dbpedia.org/resource/Toy_Story",
        "dbpedia_label": "Toy Story"
    }
]
```

In order to define the *item representation*, the `ItemAnalyzerConfig` class must be instantiated and the following 
parameters should be defined:

* ***source***: the path of the file containing items info
* ***id***: the field that uniquely identifies an item
* ***output_directory***: the path where serialized representations are saved


!!! info

    In the following suppose that the raw source is a `JSON` file, but *ClayRS* is able to read
    from different sources, as CSVFile, DATFile, and more.
    
    * Refer to the [Raw source wrappers](raw_sources.md) section for more

```python
from clayrs import content_analyzer as ca

json_source = ca.JSONFile('items_info.json')

# Configuration of item representation 
movies_ca_config = ca.ItemAnalyzerConfig(
    source=json_source,
    id='movielens_id',
    output_directory='movies_codified/',
)
```

Once we have initialized our *items* configuration, we are ready to complexly represent one or more fields of the
specified *raw source*

### Complex representation

Every field of the *raw source* can be **represented** using several techniques, such as *'tfidf'*, *'embeddings'*, etc.

It is possible to process the content of each field using a **Natural Language Processing (NLP) pipeline**.
The preprocessing will be done *before* assigning a complex representation to said field.
It is also possible to assign a **custom id** for each generated representation, in order to allow a simpler reference 
in the recommendation phase.

* Both NLP pipeline and custom id are optional parameters
* If a list of *NLP preprocessors* is passed to the `preprocessing` parameter, then all operations specified will be
performed in order

So, for example, we could represent the 'plot' field by performing **lemmatization** and **stopwords removal**, 
and represent it using **tfidf**:

```python
movies_ca_config.add_single_config(
    'plot',
    ca.FieldConfig(ca.SkLearnTfIdf(),
                   preprocessing=ca.NLTK(stopwords_removal=True, lemmatization=True),
                   id='tfidf')  # Custom id
)
```

But we could also specify for the same field ***multiple*** complex representations at once with the 
`add_multiple_config()` method:

* In this case each representation can be preceded by different preprocessing operations!

So, for example, we could represent the 'genres' field by:

1. **Removing punctuation** and representing it using the pre-trained *glove-twitter-50* model from **Gensim**;
2. Performing **lemmatization** and representing it by using the **Word2Vec** model which will be trained from scratch 
on our corpus
```python
movies_ca_config.add_multiple_config(
    'genres',
    [   
        # first representation
        ca.FieldConfig(ca.WordEmbeddingTechnique(ca.Gensim('glove-twitter-50')),
                       preprocessing=ca.NLTK(remove_punctuation=True),
                       id='glove'),
        
        # second representation
        ca.FieldConfig(ca.WordEmbeddingTechnique(ca.GensimWord2Vec()),
                       preprocessing=ca.Spacy(lemmatization=True),
                       id='word2vec')
    ]
)
```

### Exogenous representation

We could expand each item by using ***Exogenous techniques***: they are very useful if you plan to use a *graph based
recommender system* later in the experiment.

In order to do that, we call the `add_single_exogenous()` method *(or `add_multiple_exogenous()` in case of multiple 
exogenous techniques)* and pass the instantiated `ExogenousTechnique` object.

!!! info

    Exogenous properties are those extracted from an **external** source, more info 
    [here](https://2017.eswc-conferences.org/sites/default/files/Slides-and-Materials/ESWC-Tutorial-Slides.pdf)

In this case we expand each content with properties extracted from the ***DBPedia ontology***:

* The **first parameter** of the `DBPediaMappingTechnique` object is the entity type of every content 
(*dbo:Film* in this case).
Multiple prefixes such as `rdf`, `rdfs`, `foaf`, `dbo` are imported by default, but if you need another type of entity 
you can pass its uri directly

`
'dbo:Film' <-EQUIVALENT-> '<http://dbpedia.org/ontology/Film>'
`

* The **second parameter** instead is the field in the raw source which must exactly match the string representation of the 
*rdfs:label* of the content on DBPedia

```python
movies_ca_config.add_single_exogenous(
    ca.ExogenousConfig(ca.DBPediaMappingTechnique('dbo:Film', 'dbpedia_label'),
                       id='dbpedia')
)
```

### Store in an index

You could also store in a complex data structure certain representation codified for the contents.

In the following we are exporting the textual data "as is" and preprocessed with *stopwords_removal* and *stemming*
in a [`Whoosh`](https://github.com/mchaput/whoosh) index

!!! info
    
    Textual representations stored in an index can be exploited later in the RecSys phase by the 
    [IndexQuery][clayrs.recsys.content_based_algorithm.index_query.index_query.IndexQuery] algorithm

```python
movies_ca_config.add_multiple_config(
    'genres',
    [   
        # first representation - no preprocessing
        ca.FieldConfig(ca.OriginalData(),
                       memory_interfaces=ca.SearchIndex('index_folder'),
                       id='index_original'),
        
        # first representation - with preprocessing
        ca.FieldConfig(ca.OriginalData(),
                       preprocessing=ca.NLTK(stopwords_removal=True, stemming=True),
                       memory_interfaces=ca.SearchIndex('index_folder'),
                       id='index_original'),
    ]
)
```

## User Config

Suppose the following **JSON** file which contains information about movies: It will act as *raw source for users*.

```title="CSV users raw source"
user_id,age,gender,occupation,zip_code
1,24,M,technician,85711
2,53,F,other,94043
```

In order to define the *user representation*, the `UserAnalyzerConfig` class must be instantiated and the following 
parameters should be defined:

* ***source***: the path of the file containing users info
* ***id***: the field that uniquely identifies an user
* ***output_directory***: the path where serialized representations are saved

```python
# Configuration of user representation
users_ca_config = ca.UserAnalyzerConfig(
    ca.CSVFile('users_info.csv'),
    id='user_id',
    output_directory='users_codified/',
)
```

The operations you could perform for users are exactly the same you could perform on items! So please refer to the 
[above section](#item-config)

For example, we could just expand each user with exogenous properties extracted from local dataset:

`PropertiesFromDataset()` exogenous technique allows specifying which fields to use in order to expand every user info

*   If no field is specified, **all fields** from the raw source will be used

> In this case, we expand every user with `gender` and `occupation`

```python
users_ca_config.add_single_exogenous(
    ca.ExogenousConfig(
        ca.PropertiesFromDataset(field_name_list=['gender', 'occupation'])
    )
)
```

## Serializing Content

At the end of the configuration step, we provide the configuration (regardless if it's for *items* or *users*) to the 
[`ContentAnalyzer` class][content-analyzer-class] and call the `fit()` method:

* The Content Analyzer will **represent** and **serialize** every item.

```python
# complexly represent items
ca.ContentAnalyzer(config=movies_ca_config).fit()

# complexly represent users
ca.ContentAnalyzer(config=users_ca_config).fit()
```

## Exporting to JSON file

There is also the optional parameter `export_json` in the
[`ItemAnalyzerConfig`][clayrs.content_analyzer.config.ItemAnalyzerConfig] or 
[`UserAnalyzerConfig`][clayrs.content_analyzer.config.UserAnalyzerConfig]:

* If set to True, contents complexly represented will also be serialized in a human readable JSON

```python
# Configuration of item representation 
movies_ca_config = ca.ItemAnalyzerConfig(
    source=ca.JSONFile('items_info.json'),
    id='movielens_id',
    output_directory='movies_codified/',
    export_json=True
)
```

After specifying a fitting representation for items and calling the `fit()` method of the `ContentAnalyzer`, the output
folder will have the following structure:

```
📁 movies_codified/
└── 📄 contents.json
└── 📄 1.xz
└── 📄 2.xz
└── 📄 ...
```
