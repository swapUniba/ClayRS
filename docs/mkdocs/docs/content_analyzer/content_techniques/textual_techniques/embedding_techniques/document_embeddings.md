# Document Embeddings

Via the following, you can obtain embeddings of ***document*** granularity

```python
from clayrs import content_analyzer as ca

# obtain document embeddings by training LDA model
# on corpus of contents to complexly represent
ca.DocumentEmbeddingTechnique(embedding_source=ca.GensimLDA())
```

::: clayrs.content_analyzer.DocumentEmbeddingTechnique
    handler: python
    options:    
        show_root_toc_entry: true
        show_root_heading: true

## Document Embedding models

::: clayrs.content_analyzer.GensimLatentSemanticAnalysis
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true

::: clayrs.content_analyzer.GensimLDA
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true
