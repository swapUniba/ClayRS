from typing import List, Tuple

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.metrics import Metric
from orange_cb_recsys.utils.const import logger

import pandas as pd


class MetricCalculator:

    # [ (total_pred, total_truth), (total_pred, total_truth) ...]
    def __init__(self, predictions_truths: List[Split] = None):
        self._split_list = predictions_truths

    def eval_metrics(self, metric_list: List[Metric]) -> Tuple[pd.DataFrame, pd.DataFrame]:

        frames_to_concat = []

        for metric in metric_list:
            metric_result_list = []

            if self._split_list is None:
                split_list = metric._get_pred_truth_list()
            else:
                split_list = self._split_list

            for split in split_list:
                if not split.pred.empty and not split.truth.empty:
                    from_id_valid = split.pred['from_id']
                    # Remove from truth item of which we do not have predictions
                    split.truth = split.truth.query('from_id in @from_id_valid')
                    metric_result = metric.perform(split)
                    metric_result_list.append(metric_result)

            total_results_metric = pd.concat(metric_result_list)

            if not total_results_metric.empty:
                total_results_metric = total_results_metric.groupby('from_id').mean()

                total_results_metric.index.name = 'from_id'

                frames_to_concat.append(total_results_metric)

        final_result = pd.concat(frames_to_concat, axis=1)

        system_results = final_result.loc[['sys']]
        each_user_result = final_result.drop(['sys'])

        each_user_result = each_user_result.dropna(axis=1, how='all')

        return system_results, each_user_result
