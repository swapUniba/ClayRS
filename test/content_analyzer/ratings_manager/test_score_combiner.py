from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager.score_combiner import ScoreCombiner


class TestScoreCombiner(TestCase):
    def test_combine(self):
        test_list = [1.0, 0.0, 0.5, 1.0]
        self.assertAlmostEqual(ScoreCombiner("avg").combine(test_list), 0.625)
        self.assertAlmostEqual(ScoreCombiner("min").combine(test_list), 0.0)
        self.assertAlmostEqual(ScoreCombiner("max").combine(test_list), 1.0)
        self.assertAlmostEqual(ScoreCombiner("mode").combine(test_list), 1.0)
