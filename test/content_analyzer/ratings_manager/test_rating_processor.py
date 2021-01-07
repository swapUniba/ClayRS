from unittest import TestCase
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer


class TestNumberNormalizer(TestCase):
    def test_fit(self):
        self.assertAlmostEqual(NumberNormalizer(-10, -5).fit(-6), .6, places=3)
        self.assertAlmostEqual(NumberNormalizer(-5, 4).fit(0.5), 0.222, places=3)
        self.assertAlmostEqual(NumberNormalizer(0, 5).fit(2), -0.2, places=3)
        self.assertAlmostEqual(NumberNormalizer(1, 5).fit(2), -0.5, places=3)
        self.assertAlmostEqual(NumberNormalizer(-7, 0).fit(-6), -0.714, places=3)
        self.assertAlmostEqual(NumberNormalizer(0, 10).fit(0.5), -0.9, places=3)
        self.assertAlmostEqual(NumberNormalizer(0, 10).fit(11), 10, places=3)
        self.assertAlmostEqual(NumberNormalizer(0, 10).fit(-1), 0, places=3)

