import unittest

from torchaudio import transforms

import clayrs.content_analyzer.information_processor.preprocessors.audio_preprocessors.torch_builtin_augmenter as clayrs_augments

from test.content_analyzer.information_processor.preprocessors.test_audiopreprocessors.test_torch_builtin_transformer import \
    TestTorchBuiltInTransformer


class TestTorchAudioFrequencyMasking(TestTorchBuiltInTransformer):

    def setUp(self):
        self.technique = clayrs_augments.TorchFrequencyMasking(freq_mask_param=80)
        self.og_technique = lambda x, y: transforms.FrequencyMasking(freq_mask_param=80)(transforms.Spectrogram()(x))

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioTimeMasking(TestTorchBuiltInTransformer):

    def setUp(self):
        self.technique = clayrs_augments.TorchTimeMasking(time_mask_param=80)
        self.og_technique = lambda x, y: transforms.TimeMasking(time_mask_param=80)(transforms.Spectrogram()(x))

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioTimeStretch(TestTorchBuiltInTransformer):

    def setUp(self):
        self.technique = clayrs_augments.TorchTimeStretch(fixed_rate=0.9)
        self.og_technique = lambda x, y: transforms.TimeStretch(fixed_rate=0.9)(transforms.Spectrogram()(x))

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


if __name__ == "__main__":
    unittest.main()
