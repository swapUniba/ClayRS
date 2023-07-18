---
hide:
  - navigation
  - toc
---

!!! warning

    Docs are complete, but revision is still a Work in Progress. Sorry for any typos!

<p align="center">
    <img src="https://user-images.githubusercontent.com/26851363/172485577-be6993ef-47c3-4b3c-9187-4988f6c44d94.svg" alt="ClayRS logo" style="width:60%;"/>
</p>

# Welcome to ClayRS's documentation!

[![Build Status](https://github.com/swapUniba/ClayRS/actions/workflows/testing_pipeline.yml/badge.svg)](https://github.com/swapUniba/ClayRS/actions/workflows/testing_pipeline.yml)&nbsp;&nbsp;
[![Docs](https://github.com/swapUniba/ClayRS/actions/workflows/docs_building.yml/badge.svg)](https://swapuniba.github.io/ClayRS/)&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/swapUniba/ClayRS/branch/master/graph/badge.svg?token=dftmT3QD8D)](https://codecov.io/gh/swapUniba/ClayRS)&nbsp;&nbsp;
[![Python versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/)

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

The various sections of this documentation will guide you in becoming a full expert of **ClayRS**!
