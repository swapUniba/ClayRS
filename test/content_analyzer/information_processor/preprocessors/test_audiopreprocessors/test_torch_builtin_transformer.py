import random
import unittest
from typing import Optional

import torch
import torchaudio
from torchaudio import transforms

from clayrs.content_analyzer.information_processor.preprocessors.audio_preprocessors import \
    torch_builtin_transformer as clayrs_transforms


class TestTorchBuiltInTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # 10 waveforms with different sample rates
        sample_rates = [5000 if i < 5 else 8000 for i in range(10)]
        cls.batch_audios = {'waveform': torch.randn(size=(10, 1, 1000)), 'sample_rate': sample_rates}

        # each subclass needs only to overwrite these with setup() method and call
        # assertTrue on expected_result_equal()
        cls.technique: Optional[clayrs_transforms.TorchBuiltInTransformer] = None
        cls.og_technique = None

    def expected_result_equal(self):

        expected = []
        result = []
        for waveform, sr in zip(self.batch_audios['waveform'], self.batch_audios['sample_rate']):
            # setting manual seed for those preprocessors which randomly apply the transformation
            torch.manual_seed(42)
            random.seed(42)
            expected.append(self.og_technique(waveform, sr))
            torch.manual_seed(42)
            random.seed(42)
            result.append(self.technique((waveform, sr)))

        return all(torch.equal(x, y[0]) for x, y in zip(expected, result))


class TestTorchAudioResample(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchResample()
        self.og_technique = lambda x, y: torchaudio.functional.resample(x, y, 16000)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioMuLawEncoding(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchMuLawEncoding()
        self.og_technique = lambda x, y: transforms.MuLawEncoding()(x)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioMuLawDecoding(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchMuLawDecoding()
        self.og_technique = lambda x, y: transforms.MuLawDecoding()(x)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioFade(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchFade()
        self.og_technique = lambda x, y: transforms.Fade()(x)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioVol(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchVol(0.2)
        self.og_technique = lambda x, y: transforms.Vol(0.2)(x)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioAddNoise(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        noise = self.batch_audios['waveform'][0]
        snr = torch.tensor([20, 10, 3, 5, 1, 7, 9, 11, 18, 20])

        self.technique = clayrs_transforms.TorchAddNoise(noise=noise, snr=snr)
        self.og_technique = lambda x, y: transforms.AddNoise()(x, noise=noise, snr=snr)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioConvolve(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        filter = torch.tensor([[1.]])

        self.technique = clayrs_transforms.TorchConvolve(y=filter)
        self.og_technique = lambda x, y: transforms.Convolve()(x, filter)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())

# TO-DO: fix this test

# class TestTorchAudioFFTConvolve(TestTorchBuiltInTransformer):
#
#     def setUp(self) -> None:
#
#         filter = torch.tensor([[1.]])
#
#         self.technique = clayrs_transforms.TorchFFTConvolve(y=filter)
#         self.og_technique = lambda x, y: transforms.Convolve()(x, filter)
#
#     def test_forward(self):
#         self.assertTrue(self.expected_result_equal())


class TestTorchAudioSpeed(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        factor = 0.9

        self.technique = clayrs_transforms.TorchSpeed(factor)
        self.og_technique = lambda x, y: transforms.Speed(y, factor)(x)[0]

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioSpeedPerturbation(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        factors = [0.9, 1.1, 1.0, 1.0, 1.0]

        self.technique = clayrs_transforms.TorchSpeedPerturbation(factors)
        self.og_technique = lambda x, y: transforms.SpeedPerturbation(y, factors)(x)[0]

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioDeemphasis(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchDeemphasis()
        self.og_technique = lambda x, y: transforms.Deemphasis()(x)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchAudioPreemphasis(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchPreemphasis()
        self.og_technique = lambda x, y: transforms.Preemphasis()(x)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestConvertToMono(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.ConvertToMono()
        self.og_technique = lambda x, y: torch.mean(x, dim=0, keepdim=True)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


if __name__ == "__main__":
    unittest.main()
