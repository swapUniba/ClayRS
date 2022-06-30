from unittest import TestCase
from clayrs.content_analyzer.ratings_manager.score_processor import NumberNormalizer


class TestNumberNormalizer(TestCase):
    def test_fit(self):
        scores = [1, 2, 5, 5, 3, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 10]

        result = []
        for score in scores:
            converted = NumberNormalizer(scale=(1, 10)).fit(score)
            result.append(converted)

        expected = [-1.0, -0.77777777, -0.11111111, -0.11111111,
                    -0.55555555, -0.44444444, -0.42222222, -0.39999999,
                    -0.37777777, -0.35555555, -0.33333333, 1.0]

        for expected_score, result_score in zip(expected, result):
            self.assertAlmostEqual(expected_score, result_score)

        # Test with rounding at the fourth digit
        result_rounded = []
        for score in scores:
            converted_rounded = NumberNormalizer(scale=(1, 10), decimal_rounding=4).fit(score)
            result_rounded.append(converted_rounded)

        expected_rounded = [-1.0, -0.7778, -0.1111, -0.1111, -0.5556,
                            -0.4444, -0.4222, -0.4, -0.3778, -0.3556,
                            -0.3333, 1.0]

        for expected_score_rounded, result_score_rounded in zip(expected_rounded, result_rounded):
            self.assertAlmostEqual(expected_score_rounded, result_score_rounded)

    def test_error(self):

        # 2 numbers must be passed
        with self.assertRaises(ValueError):
            NumberNormalizer(scale=(1,))

        with self.assertRaises(ValueError):
            NumberNormalizer(scale=(1, 2, 3))
