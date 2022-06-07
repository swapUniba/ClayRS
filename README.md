# ClayRS

[![Build Status](https://github.com/swapUniba/ClayRS/actions/workflows/testing_pipeline.yml/badge.svg)](https://github.com/swapUniba/ClayRS/actions/workflows/testing_pipeline.yml)&nbsp;&nbsp;
[![Docs](https://github.com/swapUniba/ClayRS/actions/workflows/docs_building.yml/badge.svg)](https://swapuniba.github.io/ClayRS/)&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/swapUniba/ClayRS/branch/master/graph/badge.svg?token=dftmT3QD8D)](https://codecov.io/gh/swapUniba/ClayRS)&nbsp;&nbsp;
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)


***ClayRS*** is a python framework for (mainly) content-based recommender systems which allows you to perform several operations, starting from a raw representation of users and items to building and evaluating a recommender system. It also supports graph-based recommendation with feature selection algorithms and graph manipulation methods.

The framework has three main modules, which you can also use individually:

<p align="center">
    <img src="https://user-images.githubusercontent.com/26851363/164490523-00d60efd-7b17-4d20-872a-28eaf2323b03.png" alt="ClayRS" style="width:75%;"/>
</p>

Given a raw source, the ***Content Analyzer***:
* Creates and serializes contents,
* Using the chosen configuration

The ***RecSys*** module allows to:
* Instantiate a recommender system
    * *Using items and users serialized by the Content Analyzer*
* Make score *prediction* or *recommend* items for the active user(s)

The ***EvalModel*** has the task of evaluating a recommender system, using several state-of-the-art metrics

Code examples for all three modules will follow in the *Usage* section

Installation
=============
*ClayRS* requires Python **3.8** or later, while package dependencies are in `requirements.txt` and are all installable via `pip`, as *ClayRS* itself.

To install it execute the following command:

``
pip install git+https://github.com/SwapUniba/clayrs.git
``

Usage
=====

### Content Analyzer
The first thing to do is to import the Content Analyzer module
* We will access its methods and classes via dot notation
```python
import clayrs.content_analyzer as ca
```

First let's point to the source containing raw information to process
```python
raw_source = ca.JSONFile('items_info.json')
```

Then let's start building the configuration for the items
* Note that same operations that can be specified for *items*, could be also specified for *users*, via the
`ca.UserAnalyzerConfig` class
```python
# Configuration of item representation
movies_ca_config = ca.ItemAnalyzerConfig(
    source=raw_source,
    id='movielens_id',  # id which uniquely identifies each item
    output_directory='movies_codified/'  # where items complexly represented will be stored
)
```


Let's represent the *plot* field of each content with a TfIdf representation
* Since the `preprocessing` parameter has been specified, then each field is first preprocessed with the specified
operations
```python
movies_ca_config.add_single_config(
    'plot',
    ca.FieldConfig(ca.SkLearnTfIdf(),
                   preprocessing=ca.NLTK(stopwords_removal=True, lemmatization=True),
                   id='tfidf')  # Custom id
)
```

Let's also represent the *description* field of each content with the embedding representation of each word by using the
*glove-twitter-50* model of the Gensim library
* *ClayRS* will download the model for you if it is not already present locally

In this case we are performing a pipeline of *NLP* operations
```python
movies_ca_config.add_single_config(
    'description',
    ca.FieldConfig(ca.WordEmbeddingTechnique(ca.Gensim('glove-twitter-50')),
                   preprocessing=[
                       ca.Spacy(url_tagging=True, remove_punctuation=True),
                       ca.Ekphrasis(corrector='english', spell_correction=True)
                   ],
                   id='gensim')  # Custom id
)
```

To finalize the Content Analyzer part, let's instantiate the `ContentAnalyzer` class by passing the built configuration
and by calling its `fit()` method
* The items will be created with the specified representations and serialized
```python
ca.ContentAnalyzer(movies_ca_config).fit()
```

### RecSys
Similarly above, we must first import the RecSys module
```python
import clayrs.recsys as rs
```

Then we load the rating frame from a TSV file
* In this case in our file the first three columns are user_id, item_id, score in this order
  * If your file has a different structure you must specify how to map the column via parameters, check documentation
  for more

```python
ratings = ca.Ratings(ca.CSVFile('ratings.tsv', separator='\t'))
```

Let's split with the KFold technique the loaded rating frame into train set and test set
* since `n_splits=2`, train_list will contain two *train_sets* and test_list will contain two *test_sets*
```python
train_list, test_list = rs.KFoldPartitioning(n_splits=2).split_all(ratings)
```

In order to recommend items to users, we must choose an algorithm to use
* In this case we are using the `CentroidVector` algorithm which will work by using the first representation
specified for the *plot* field and the representation with *glove* id specified for the *description* field
* You can freely choose which representation to use among all representation codified for the fields in the Content
Analyzer phase
```python
centroid_vec = rs.CentroidVector(
    {'plot': 0,  # the first representation codified for 'plot' field, we didn't specified a custom id
     'description': 'glove'},
  
    similarity=rs.CosineSimilarity()
)
```

Let's now compute the top-10 ranking for each user of the train set
* By default the candidate items are those in the test set of the user, but you can change this behaviour with the
`methodology` parameter

Since we used the kfold technique, we iterate over the train sets and test sets
```python
result_list = []

for train_set, test_set in zip(train_list, test_list):
  
  cbrs = rs.ContentBasedRS(centroid_vec, train_set, 'movies_codified/')
  rank = cbrs.fit_rank(test_set, n_recs=10)

  result_list.append(rank)
```

### EvalModel
Similarly to the Content Analyzer and RecSys module, we must first import the evaluation module
```python
import clayrs.evaluation as eva
```

The Evaluation module needs the following parameters:

*   A list of computed rank/predictions (in case multiple splits must be evaluated)
*   A list of truths (in case multiple splits must be evaluated)
*   List of metrics to compute

Obviously the list of computed rank/predictions and list of truths must have the same length,
and the rank/prediction in position $i$ will be compared with the truth at position $i$

```python
em = eva.EvalModel(
    pred_list=result_list,
    truth_list=test_list,
    metric_list=[
        eva.NDCG(),
        eva.Precision(),
        eva.RecallAtK(k=5)
    ]
)
```

Then simply call the `fit()` method of the instantiated object
* It will return two pandas DataFrame: the first one contains the metrics aggregated for the system,
while the second contains the metrics computed for each user (where possible)

```python
sys_result, users_result =  em.fit()
```

Note that the EvalModel is able to compute evaluation of recommendations generated by other tools/frameworks, check
documentation for more
