from unittest import TestCase
import pandas as pd
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.classification_metrics import Precision, PrecisionAtK, RPrecision, Recall, \
    RecallAtK, FMeasure, FMeasureAtK

from orange_cb_recsys.evaluation.eval_model import MetricCalculator
from orange_cb_recsys.evaluation.metrics.error_metrics import MAE, MSE, RMSE
from orange_cb_recsys.evaluation.metrics.fairness_metrics import PredictionCoverage, CatalogCoverage, DeltaGap, \
    GiniIndex
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric, ScoresNeededMetric
from orange_cb_recsys.evaluation.metrics.plot_metrics import LongTailDistr, PopProfileVsRecs, PopRecsCorrelation
from orange_cb_recsys.evaluation.metrics.ranking_metrics import NDCG, MRR, NDCGAtK, MRRAtK, Correlation


# Every Metric is tested singularly, so we just check that everything goes smoothly at the
# MetricEvaluator level
class TestMetricCalculator(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        rank1 = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3'],
            'to_id': ['i9', 'i6', 'inew1', 'inew2', 'i6', 'i2', 'i1', 'i8',
                      'i10', 'inew3', 'i2', 'i1', 'i8', 'i4', 'i9',
                      'i3', 'i12', 'i2'],

            'score': [500, 450, 400, 350, 300, 250, 200, 150,
                      400, 300, 200, 100, 50, 25, 10,
                      100, 50, 20]
        })

        score1 = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3'],
            'to_id': ['i9', 'i6', 'inew1', 'inew2', 'i6', 'i2', 'i1', 'i8',
                      'i10', 'inew3', 'i2', 'i1', 'i8', 'i4', 'i9',
                      'i3', 'i12', 'i2'],

            'score': [4.36, 2.55, 1.23, 4.36, 3.55, 2.58, 5, 4.2,
                      3.56, 4.22, 4.25, 1.4, 4.4, 3.33, 2.53,
                      2.21, 1.53, 3.32]
        })

        truth1 = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3', 'u3', 'u3'],
            'to_id': ['i1', 'i2', 'i6', 'i8', 'i9',
                      'i1', 'i2', 'i4', 'i9', 'i10',
                      'i2', 'i3', 'i12', 'imissing3', 'imissing4'],

            'score': [3, 3, 4, 1, 1,
                      5, 3, 3, 4, 4,
                      4, 2, 3, 3, 3]
        })

        rank2 = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3'],
            'to_id': ['i10', 'i5', 'i4', 'i3', 'i7',
                      'i70', 'i3', 'i71', 'i8', 'i11',
                      'i10', 'i1', 'i4'],

            'score': [500, 400, 300, 200, 100,
                      400, 300, 200, 100, 50,
                      150, 100, 50]
        })

        score2 = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3'],
            'to_id': ['i10', 'i5', 'i4', 'i3', 'i7',
                      'i70', 'i3', 'i71', 'i8', 'i11',
                      'i10', 'i1', 'i4'],

            'score': [4.4, 3.35, 2.22, 2.56, 3.1,
                      2.55, 1.89, 4.3, 3.77, 3.89,
                      4.23, 4.56, 5]
        })

        truth2 = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3', 'u3', 'u3'],
            'to_id': ['i3', 'i4', 'i5', 'i7', 'i10',
                      'i3', 'i70', 'i71', 'i8', 'i11',
                      'i4', 'i1', 'i10', 'imissing1', 'imissing2'],

            'score': [4, 2, 2, 5, 1,
                      5, 4, 4, 3, 4,
                      2, 3, 1, 1, 1]
        })

        rank_split1 = Split(rank1, truth1)
        rank_split2 = Split(rank2, truth2)

        score_split1 = Split(score1, truth1)
        score_split2 = Split(score2, truth2)

        cls.rank_split_list = [rank_split1, rank_split2]
        cls.score_split_list = [score_split1, score_split2]

        catalog = ['i' + str(num) for num in range(100)]

        cls.catalog = set(catalog)

    def test_eval_ranking_needed_metrics_explicit_split(self):

        # We pass the split_list as a parameter
        c = MetricCalculator(self.rank_split_list)

        system_res, each_user_res = c.eval_metrics([
            Precision(),
            PrecisionAtK(2),
            RPrecision(),
            Recall(),
            RecallAtK(2),
            FMeasure(),
            FMeasureAtK(2),

            NDCG(),
            NDCGAtK(2),
            MRR(),
            MRRAtK(2),
            Correlation('pearson'),
            Correlation('kendall'),
            Correlation('spearman'),

            PredictionCoverage(self.catalog),
            CatalogCoverage(self.catalog, top_n=2),
            GiniIndex(),
            DeltaGap(user_groups={'a': 0.5, 'b': 0.5}),

            LongTailDistr(out_dir='test_plot'),
            PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_plot'),
            PopRecsCorrelation(out_dir='test_plot')
        ])

        self.assertIsInstance(system_res, pd.DataFrame)
        self.assertIsInstance(each_user_res, pd.DataFrame)

    def test_eval_score_needed_metrics_explicit_split(self):
        c = MetricCalculator(self.score_split_list)

        system_res, each_user_res = c.eval_metrics([
            MAE(),
            MSE(),
            RMSE(),
        ])

        self.assertIsInstance(system_res, pd.DataFrame)
        self.assertIsInstance(each_user_res, pd.DataFrame)

    def test_eval_ranking_needed_metrics_implicit_split(self):

        # We set the split_list directly by the class attribute
        c = MetricCalculator()
        RankingNeededMetric.rank_truth_list = self.rank_split_list

        system_res, each_user_res = c.eval_metrics([
            Precision(),
            PrecisionAtK(2),
            RPrecision(),
            Recall(),
            RecallAtK(2),
            FMeasure(),
            FMeasureAtK(2),

            NDCG(),
            NDCGAtK(2),
            MRR(),
            MRRAtK(2),
            Correlation('pearson'),
            Correlation('kendall'),
            Correlation('spearman'),

            PredictionCoverage(self.catalog),
            CatalogCoverage(self.catalog, top_n=2),
            GiniIndex(),
            DeltaGap(user_groups={'a': 0.5, 'b': 0.5}),

            LongTailDistr(out_dir='test_plot'),
            PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_plot'),
            PopRecsCorrelation(out_dir='test_plot')
        ])

        self.assertIsInstance(system_res, pd.DataFrame)
        self.assertIsInstance(each_user_res, pd.DataFrame)

    def test_eval_score_needed_metrics_implicit_split(self):
        c = MetricCalculator()

        ScoresNeededMetric.score_truth_list = self.score_split_list
        system_res, each_user_res = c.eval_metrics([
            MAE(),
            MSE(),
            RMSE(),
        ])

        self.assertIsInstance(system_res, pd.DataFrame)
        self.assertIsInstance(each_user_res, pd.DataFrame)

    def doCleanups(self) -> None:
        RankingNeededMetric._clean_pred_truth_list()
        ScoresNeededMetric._clean_pred_truth_list()
