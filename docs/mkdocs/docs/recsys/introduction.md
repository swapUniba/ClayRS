!!! warning

    Docs are complete, but revision is still a Work in Progress. Sorry for any typos!


# Introduction

The Recommender System module lets you easily build a Content Based Recommender System (***CBRS***) or a Graph Based Recommender system
(***GBRS***) with various algorithms.

!!! info
  
    The Recsys module is grounded on contents created with the Content Analyzer

The following will introduce you to the standard usage pipeline for this module, starting from importing the dataset to
generating recommendation lists.

## Importing the dataset

The **Ratings** class allows you to import rating from a source file (or also from an existent dataframe) into a custom object.

**If** the source file contains users, items and ratings in this order, no additional parameters are needed,
**otherwise**  the mapping must be explictly specified using:

* **'user_id'** column,
* **'item_id'** column,
* **'score'** column

In this case the dataset we want to import is a CSV file with the following header:

```
user_id,item_id,rating,timestamp
```

As you can see the *user id column*, *item id column* and *score column* are the first three column and are already in
sequential order, so no additional parameter is required to the `Ratings` class:

```python
import clayrs.content_analyzer as ca

ratings_raw_source = ca.CSVFile('ratings.csv') # (1)

ratings = ca.Ratings(ratings_raw_source)
```

1. In this case our raw source is a CSV file, but ClayRS can also read from JSON files, DAT files and more

## Splitting the dataset

Once you imported the dataset, the first thing you may want to do is to split it with a ***Partitioning technique***

* The output of any partitioning technique are two lists. The first containing all the train set produced by the
partitioning technique (two train set in the below example), the other containing all the test set produced by the
partitioning technique (two test set in the below example)

```python
import clayrs.recsys as rs

# kfold partitioning technique
kf = rs.KFoldPartitioning(n_splits=2)

train_list, test_list = kf.split_all(ratings) # (1)
```

1. You can pass to the `split_all()` method a specific `user_id_list` in case you only want to perform the splitting operation
for a specific subset of users (e.g. select only users with more than x ratings)

## Defining a Content Based Recommender System

A Content Based Recommender System needs an algorithm for ranking or predicting items to users.
There are many available, in the following example we will use the **CentroidVector** algorithm:

*   It computes the centroid vector of the features of items *liked by the user*
*   It computes the similarity between the centroid vector and unrated items

The items liked by a user are those having a rating higher or equal than a specific **threshold**.
If the threshold is not specified, the average score of all items liked by the user is used.

As already said, the Recommender System leverages the representations defined by the Content Analyzer.
Suppose you have complexly represented the 'plot' with a simple TfIdf technique and assigned to this representation
the `tfidf` id:

```python
import clayrs.recsys as rs

centroid_vec = rs.CentroidVector(
    {'plot': 'tfidf'},
    
    similarity=rs.CosineSimilarity()
)
```
You can reference representation for a field also with an integer, in case you didn't assign
any custom id during Content Analyzer phase.

```python
centroid_vec = rs.CentroidVector(
    {'plot': 0},  # (1)
    
    similarity=rs.CosineSimilarity()
)
```

1. This means that you want to use the first representation with which the 'plot' field was complexly represented

Please note that multiple representations could be adopted for a single field, and also multiple representations for
multiple fields can be combined together! Simply specify them in the `item_field` dict that must be passed to any
Content Based algorithm:

```python
centroid_vec = rs.CentroidVector(
    {'plot': [0, 'glove-50', 'glove-100'],
     'genre': ['tfidf', 'fasttext']},
    
    similarity=rs.CosineSimilarity()
)
```

After choosing the algorithm, you are ready to instantiate the `ContentBasedRS` class.

A CBRS needs the following parameters:

* The recommendation  algorithm
* The train set
* The path of the items serialized by the Content Analyzer

```python

train_set = test_list[0] # (1)

cbrs = rs.ContentBasedRS(random_forests, train_set, 'movies_codified/')
```

1. Since every partitioning technique returns a *list* of train sets ([here](#splitting-the-dataset)), in this way we are using only the first train set produced. Just below there's an example
on how to produce recommendation for more than one split

## Defining a Graph Based Recommender System

A Graph Based Recommender System (**GBRS**) requires to first define a *graph*

Ratings imported are used to create a **Full Graph** where property nodes (e.g. *gender* for users, *budget* for movies) can be linked to every node without any restriction

> The framework also allows to create a **Bipartite Graph** (a graph without property node) and a **Tripartite Graph** (where property nodes are only linked to item nodes)

In order to load properties in the graph, we must specify where users and items are serialized and ***which properties to add*** (the following is the same for *item_exo_properties*):

*   If *user_exo_properties* is specified as a **set**, then the graph will try to load **all properties** from **said exogenous representation**
```python
# example
{'my_exo_id'}
```

*   If *user_exo_properties* is specified as a **dict**, then the graph will try to load **said properties** from **said exogenous representation**
```python
# example
{'my_exo_id': ['my_prop1', 'my_prop2']]}
```

Let's now create the graph loading all properties:

```python
full_graph = rs.NXFullGraph(ratings, 
                            user_contents_dir='users_codified/', # (1)
                            item_contents_dir='movies_codified/', # (2)
                            user_exo_properties={0}, # (3)
                            item_exo_properties={'dbpedia'}, # (4)
                            link_label='score')
```

1. Where users complexly represented have been serialized during Content Analyzer phase
2. Where items complexly represented have been serialized during Content Analyzer phase
3. This means that you want to use the first exogenous representation with which each user has been expanded
4. You can also access exogenous representation with custom id, if specified during Content Analyzer phase

The last step to perform before defining the GBRS is to instantiate an algorithm for ranking or predicting items to users.

In the following example we use the **Personalized PageRank** algorithm:

```python
pr = rs.NXPageRank(personalized=True)
```

Finally we can instantiate the GBRS!

```python
gbrs = rs.GraphBasedRS(pr, full_graph)
```

## Generating recommendations

!!! info

	The following procedure works both for ***CBRS*** and ***GBRS***. In the following we will consider a cbrs as an example
    
    * For ***GBRS*** there is no `fit()` method, only `rank()` or `predict()` method must be called

Now the ***cbrs*** must be fit before we can compute the rank:

* We could do this in two separate steps, by first calling the `fit(..)` method and then the `rank(...)` method 

* Or by calling directly the `fit_rank(...)` method, which performs both in one step

In this case we choose the first method:

```python

cbrs.fit()

test_set = test_list[0] # (1)

rank = cbrs.rank(test_set, n_recs=10)  # top-10 recommendation for each user
```

1. Since every partitioning technique returns a *list* of test sets ([here](#splitting-the-dataset)), in this way we are using only the first train set produced. Just below there's an example
on how to produce recommendation for more than one split

In case you perform a splitting of the dataset which returns a multiple train and test sets (KFold technique):

```python
original_rat = ca.Ratings(ca.CSVFile(ratings_path))

train_list, test_list = rs.KFoldPartitioning(n_splits=5).split_all(original_rat)

alg = rs.CentroidVector()  # any cb algorithm

for train_set, test_set in zip(train_list, test_list):

    cbrs = rs.ContentBasedRS(alg, train_set, items_path)
    rank_to_append = cbrs.fit_rank(test_set)

    result_list.append(rank_to_append)
```

`result_list` will contain recommendation lists for each split

### Customizing the ranking process

You can customize the ranking process by changing the parameters of the `rank(...)` method

* You can choice for which users to produce recommendations:

```python
rank = cbrs.rank(test_set, user_list=['u1', 'u23', 'u56'])
```

* If a cut rank list for each user must be produced:

```python
rank = cbrs.rank(test_set, n_recs=10)
```

* If a different *methodology* must be used:
  
!!! info
   
	A *methodology* lets you customize which items must be ranked for each user.
	For each target user $u$, the following 4 different methodologies are available for defining those lists:
	
	1.   **TestRatings** (default): the list of items to be evaluated consists of items rated by $u$ in the test set
	2.   **TestItems**: every item in the test set of every user except those in the training set of $u$ will be predicted
	3.   **TrainingItems**: every item in the training set of every user will be predicted except those in the training set of $u$
	4.   **AllItems**: the whole set of items defined will be predicted, except those in the training set of $u$

	More information on [this paper](https://repositorio.uam.es/bitstream/handle/10486/665121/precision-oriented_bellogin_recsys_2011_ps.pdf;jsessionid=85982302D4DA9FF4DD7F21E4AC4F3391?sequence=1).

By default the methodology used is the **TestRatings** methodology

```python
rank = cbrs.rank(test_set, methodology=rs.TrainingItemsMethodology())
```

## Generating score predictions

Some algorithm (e.g. LinearPredictor algorithm) are able to predict the ***numeric rating*** that a user would give to unseen items.

The usage is exactly the same of [*generating recommendations*](#generating-recommendations) and 
[*customizing the ranking process*](#customizing-the-ranking-process), the only thing that changes is the method to call:

```python
score_prediction = cbrs.fit_predict(test_set)
```

or:

```python

cbrs.fit()

score_prediction = cbrs.predict(test_set)
```

**Note**: if the `predict()` or the `fit_predict()` method is called for an algorithm that is not able to perform score prediction,
the `NotPredictionAlg` exception is raised