# Contextualized Embeddings

Via the following, you can obtain embeddings of *finer* granularity from models which are able to return also
embeddings of *coarser* granularity (e.g. obtain word embeddings from a model which is also able to return sentence 
embeddings).

For now only models working at sentence and token level are implemented

```python
from clayrs import content_analyzer as ca

# obtain sentence embeddings combining token embeddings with a 
# centroid technique
ca.Sentence2WordEmbedding(embedding_source=ca.BertTransformers('bert-base-uncased'))
```

::: clayrs.content_analyzer.Sentence2WordEmbedding
    handler: python
    options:    
        show_root_toc_entry: true
        show_root_heading: true

## Model able to return sentence and token embeddings

::: clayrs.content_analyzer.BertTransformers
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.T5Transformers
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true
