# Combine Embeddings

Via the following, you can obtain embeddings of *coarser* granularity from models which return
embeddings of *finer* granularity (e.g. obtain sentence embeddings from a model which returns word embeddings)

```python
from clayrs import content_analyzer as ca

# obtain sentence embeddings combining token embeddings with a 
# centroid technique
ca.Word2SentenceEmbedding(embedding_source=ca.Gensim('glove-twitter-50'),
                          combining_technique=ca.Centroid())
```

::: clayrs.content_analyzer.Word2SentenceEmbedding
    handler: python
    options:    
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.Word2DocEmbedding
    handler: python
    options:    
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.Sentence2DocEmbedding
    handler: python
    options:    
        show_root_toc_entry: true
        show_root_heading: true

## Combining Techniques

::: clayrs.content_analyzer.Centroid
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.Sum
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.SingleToken
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true