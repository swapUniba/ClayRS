# ClayRS

![Build Status](https://github.com/swapUniba/ClayRS/workflows/Testing%20pipeline/badge.svg)&nbsp;&nbsp;[![codecov](https://codecov.io/gh/swapUniba/ClayRS/branch/master/graph/badge.svg?token=dftmT3QD8D)](https://codecov.io/gh/swapUniba/ClayRS)&nbsp;&nbsp;[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)


***ClayRS*** is a python framework for (mainly) content-based recommender systems which allows you to perform several operations, starting from a raw representation of users and items to building and evaluating a recommender system. It also supports graph-based recommendation with feature selection algorithms and graph manipulation methods.

The framework has three main modules, which you can also use individually:

<p align="center">
    <img src="https://user-images.githubusercontent.com/26851363/163631209-2a83da81-9975-4194-9964-0c32a2982434.png" alt="drawing" style="width:75%;"/>
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
```python

import clayrs.content_analyzer as ca

raw_source = ca.JSONFile('items_info.json')

# Configuration of item representation
movies_ca_config = ca.ItemAnalyzerConfig(
    source=raw_source,
    id='movielens_id',
    output_directory='movies_codified/',
)

movies_ca_config.add_single_config(
    'plot',
    ca.FieldConfig(ca.SkLearnTfIdf(),
                   preprocessing=ca.NLTK(stopwords_removal=True, lemmatization=True),
                   id='tfidf')  # Custom id
)
```
