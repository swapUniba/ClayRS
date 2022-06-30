# Fairness metrics

Fairness metrics evaluate how unbiased the recommendation lists are (e.g. unbiased towards popularity of the items)

::: clayrs.evaluation.metrics.fairness_metrics
    handler: python
    options:
        filters:
        - "!^_[^_]"
        - "!^FairnessMetric$"
        - "!.*def.*"