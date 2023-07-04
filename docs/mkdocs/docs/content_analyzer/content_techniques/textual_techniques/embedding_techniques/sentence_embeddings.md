# Sentence Embeddings

Via the following, you can obtain embeddings of ***sentence*** granularity

```python
from clayrs import content_analyzer as ca

# obtain sentence embeddings using pre-trained model 'glove-twitter-50'
# from SBERT library
ca.SentenceEmbeddingTechnique(embedding_source=ca.Sbert('paraphrase-distilroberta-base-v1'))
```

::: clayrs.content_analyzer.SentenceEmbeddingTechnique
    handler: python
    options:
        show_root_toc_entry: true
        show_root_heading: true

## Sentence Embedding models

::: clayrs.content_analyzer.BertTransformers
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.Sbert
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
