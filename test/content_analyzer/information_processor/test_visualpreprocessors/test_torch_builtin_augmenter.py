import unittest

from torchvision import transforms

import clayrs.content_analyzer.information_processor.visual_preprocessors.torch_builtin_augmenter as clayrs_augments

from test.content_analyzer.information_processor.test_visualpreprocessors.test_torch_builtin_transformer import \
    TestTorchBuiltInTransformer


class TestTorchAutoAugment(TestTorchBuiltInTransformer):

    def setUp(self):
        self.technique = clayrs_augments.TorchAutoAugment()
        self.og_technique = transforms.AutoAugment()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandAugment(TestTorchBuiltInTransformer):

    def setUp(self):
        self.technique = clayrs_augments.TorchRandAugment()
        self.og_technique = transforms.RandAugment()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchTrivialAugmentWide(TestTorchBuiltInTransformer):

    def setUp(self):
        self.technique = clayrs_augments.TorchTrivialAugmentWide()
        self.og_technique = transforms.TrivialAugmentWide()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


if __name__ == "__main__":
    unittest.main()
