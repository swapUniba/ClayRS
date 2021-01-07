from unittest import TestCase

from orange_cb_recsys.evaluation import Precision, Recall, FNMeasure, NDCG, MRR, Correlation, GiniIndex, \
    PopRecsCorrelation, LongTailDistr, CatalogCoverage, PopRatioVsRecs, DeltaGap, Serendipity, Novelty
import pandas as pd

from orange_cb_recsys.evaluation.prediction_metrics import RMSE, MAE

score_frame_fairness = pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
                                               'to_id': ["aaa", "bbb", "aaa", "bbb", "ccc", "aaa", "ddd", "bbb"],
                                               'rating': [1.0, 0.5, 0.0, 0.5, 0.6, 0.2, 0.7, 0.8]})
truth_frame_fairness = pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
                                               'to_id': ["aaa", "bbb", "aaa", "ddd", "ccc", "ccc", "ddd", "ccc"],
                                               'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})


class Test(TestCase):
    def test_perform_ranking_metrics(self):
        col = ["to_id", 'rating']
        truth_rank = {
            "item0": 1.0,
            "item1": 1.0,
            "item2": 0.85,
            "item3": 0.8,
            "item4": 0.7,
            "item5": 0.65,
            "item6": 0.4,
            "item7": 0.35,
            "item8": 0.2,
            "item9": 0.2
        }

        predicted_rank = {
            "item2": 0.9,
            "item5": 0.85,
            "item9": 0.75,
            "item0": 0.7,
            "item4": 0.65,
            "item1": 0.5,
            "item8": 0.2,
            "item7": 0.2,
        }

        truth_rank = pd.DataFrame(truth_rank.items(), columns=col)
        predicted_rank = pd.DataFrame(predicted_rank.items(), columns=col)

        results = {
            "Precision": Precision(0.75).perform(predicted_rank, truth_rank),
            "Recall": Recall(0.75).perform(predicted_rank, truth_rank),
            "F1": FNMeasure(1, 0.75).perform(predicted_rank, truth_rank),
            "F2": FNMeasure(2, 0.75).perform(predicted_rank, truth_rank),
            "NDCG":
                NDCG({0: (-1.0, 0.0), 1: (0.0, 0.3), 2: (0.3, 0.7), 3: (0.7, 1.0)}).perform(predicted_rank, truth_rank),
            "MRR": MRR(0.75).perform(predicted_rank, truth_rank),
            "pearson": Correlation('pearson').perform(predicted_rank, truth_rank),
            "kendall": Correlation('kendall').perform(predicted_rank, truth_rank),
            "spearman": Correlation('spearman').perform(predicted_rank, truth_rank)
        }

        real_results = {
            "Precision": 0.5,
            "Recall": 0.5,
            "F1": 0.5,
            "F2": 0.5,
            "NDCG": 0.908,
            "MRR": 0.8958333333333333,
            "pearson": 0.26,
            "kendall": 0.14,
            "spearman": 0.19,
        }

    def test_NDCG(self):
        score_frame = pd.DataFrame.from_dict({'to_id': ["bbb", "eee", "aaa", "ddd", "ccc", "fff", "hhh", "ggg"],
                                              'rating': [1.0, 1.0, 0.5, 0.5, 0.3, 0.3, 0.7, 0.8]})
        truth_frame = pd.DataFrame.from_dict({'to_id': ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"],
                                              'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})

        NDCG().perform(predictions=score_frame, truth=truth_frame)

        NDCG({}).perform(predictions=score_frame, truth=truth_frame)

        split_dict = {1: (-0.5, 0.0), 2: (0.0, 0.5), 3: (0.5, 1.0)}
        NDCG(split_dict).perform(predictions=score_frame, truth=truth_frame)

        split_dict = {0: (-1.0, -0.5), 1: (-0.5, 0.0), 2: (0.0, 0.5), 3: (0.5, 1.0)}
        NDCG(split_dict).perform(predictions=score_frame, truth=truth_frame)

    def test_perform_fairness_metrics(self):
        GiniIndex().perform(score_frame_fairness)
        PopRecsCorrelation('test', '.').perform(score_frame_fairness, truth_frame_fairness)
        LongTailDistr('test', '.').perform(score_frame_fairness, truth_frame_fairness)
        CatalogCoverage().perform(score_frame_fairness, truth_frame_fairness)
        PopRatioVsRecs('test', '.', {'niche': 0.2, 'diverse': 0.6, 'bb_focused': 0.2}, False).perform(
            score_frame_fairness,
            truth_frame_fairness)
        DeltaGap({'niche': 0.2, 'diverse': 0.6, 'bb_focused': 0.2})

    def test_perform_serendipity(self):
        Serendipity(10).perform(score_frame_fairness, truth_frame_fairness)

    def test_perform_novelty(self):
        Novelty(10).perform(score_frame_fairness, truth_frame_fairness)

    def test_perform_rmse(self):
        predictions = pd.DataFrame.from_dict({'to_id': ["bbb", "eee", "aaa", "ddd", "ccc", "fff", "hhh"],
                                              'rating': [5, 5, 4, 3, 3, 2, 1]})
        truth = pd.DataFrame.from_dict({'to_id': ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"],
                                        'rating': [5, 4, 3, 3, 1, 2, 1]})

        self.assertEqual(RMSE().perform(predictions, truth), 0.9258200997725514)
        self.assertEqual(MAE().perform(predictions, truth), 0.5714285714285714)

        truth = pd.DataFrame.from_dict({'to_id': ["aaa", "bbb", "ccc", "ddd", "eee", "fff"],
                                        'scores': [5, 4, 3, 3, 1, 2]})

        with self.assertRaises(Exception):
            RMSE().perform(predictions, truth)
