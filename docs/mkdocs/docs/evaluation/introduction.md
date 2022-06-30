!!! warning

	Docs are complete, but revision is still a Work in Progress. Sorry for any typos!


# Introduction

The Evaluation module has the task of evaluating a recommender system, using several ***state-of-the-art*** metrics

The usage pipeline it's pretty simple, all the work is done by the 
[`EvalModel` class][clayrs.evaluation.eval_model.EvalModel] class. Suppose you want to
evaluate recommendation lists using ***NDCG***, ***macro Precision***, ***micro Recall@5***, you need to instantiate
the EvalModel class with the following parameters:

* A list of computed rank/predictions (in case multiple splits must be evaluated)
* A list of truths (in case multiple splits must be evaluated)
* List of metrics to compute

Obviously the list of computed rank/predictions and list of truths must have the same length,
and the rank/prediction in position $i$ will be compared with the truth at position $i$


## Usage example

In this case `rank_list` and `truth_list` are results obtained from the RecSys module of the framework 

```python
import clayrs.evaluation as eva

em = eva.EvalModel(
    pred_list=rank_list,
    truth_list=truth_list,
    metric_list=[
        eva.NDCG(),
        eva.Precision(), # (1)
        eva.RecallAtK(k=5, sys_average='micro')
    ]
)
```

1. If not specified, by default system average is computed as *macro*

!!! info

    `Precision`, `Recall`, and in general all classification metrics require a **threshold** which separates relevant
    items from non-relevant.
    
    * If a threshold is specified, then it is *fixed* for all users
    * If no threshold is specified, the mean rating score of each user will be used
    
    Check documentation of each metric for more

Then simply call the `#!python fit()` method of the instantiated object

* It will return two pandas DataFrame: the first one contains the metrics aggregated for the system,
while the second contains the metrics computed for each user (where possible)

```python
sys_result, users_result =  em.fit()
```

## Evaluating external recommendation lists

The evaluation module is completely independent from the Recsys and Content Analyzer module: that means that we can
easily evaluate recommendation lists computed by other frameworks/tools!

Let's suppose we have recommendations (and related truths) generated via other tools in a csv format.
We first import them into the framework and then pass them to the `EvalModel` class

```python
import clayrs.content_analyzer as ca

csv_rank_1 = ca.CSVFile('rank_split_1.csv')
csv_truth_1 = ca.CSVFile('truth_split_1.csv')

csv_rank_2 = ca.CSVFile('rank_split_2.csv')
csv_truth_2 = ca.CSVFile('truth_split_2.csv')

# Importing split 1 (1)
rank_1 = ca.Rank(csv_rank_1)
truth_1 = ca.Ratings(csv_truth_1)

# Importing split 2 (2)
rank_2 = ca.Rank(csv_rank_2)
truth_2 = ca.Ratings(csv_truth_2)

# since multiple splits, we wrap ranks and truths in lists
imported_ranks = [rank_1, rank_2]
imported_truths = [truth_1, truth_2]
```

1. Remember that this instantiation to the `Rank/Ratings` class assumes a certain order of the columns of your
raw source. Otherwise, you need to manually map columns. 
Check [related documentation](/content_analyzer/ratings/ratings/) for more

2. Remember that this instantiation to the `Rank/Ratings` class assumes a certain order of the columns of your
raw source. Otherwise, you need to manually map columns.
Check [related documentation](/content_analyzer/ratings/ratings/) for more


Then simply evaluate them exactly in the same way as shown before!
```python
import clayrs.evaluation as eva

em = eva.EvalModel(
    pred_list=imported_ranks,
    truth_list=imported_truths,
    metric_list=[
        # ... Choose your own metrics
    ]
)

sys_results_df, users_results_df = em.fit()
```

## Perform a statistical test

ClayRS lets you also compare different learning schemas by performing statistical tests:

* Simply instantiate the desired test and call its `#!python perform()` method. The parameter it expects is the
list of `user_results` dataframe obtained in the evaluation step, one for each learning schema to compare.

```python
ttest = eva.Ttest()

all_comb_df = ttest.perform([user_result1, user_result2, user_result3])
```

!!! info
    
    In this case since the Ttest it's a paired test, the final result is a pandas DataFrame which contains learning
    schemas compared in pair:
    
    * (system1, system2)
    * (system1, system3)
    * (system2, system3)