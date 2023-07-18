# Classification metrics

A classification metric uses confusion matrix terminology (true positive, false positive, true negative, false negative)
to classify each item predicted, and in general it needs a way to discern relevant items from non-relevant items for
users

::: clayrs.evaluation.metrics.classification_metrics
    handler: python
    options:
        filters:
        - "!^_[^_]"
        - "!^ClassificationMetric$"