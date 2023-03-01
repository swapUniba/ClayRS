# Word Embeddings

Via the following, you can obtain embeddings of ***word*** granularity

```python
from clayrs import content_analyzer as ca

# obtain word embeddings using pre-trained model 'glove-twitter-50'
# from Gensim library
ca.WordEmbeddingTechnique(embedding_source=ca.Gensim('glove-twitter-50'))
```

::: clayrs.content_analyzer.WordEmbeddingTechnique
    handler: python
    options:    
        show_root_toc_entry: true
        show_root_heading: true


## Word Embedding models

::: clayrs.content_analyzer.Gensim
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.GensimDoc2Vec
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.GensimFastText
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.GensimRandomIndexing
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.GensimWord2Vec
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true
