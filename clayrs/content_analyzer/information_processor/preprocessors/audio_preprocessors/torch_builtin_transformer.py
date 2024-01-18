import inspect
from typing import Optional, Tuple, List

import torch
from torchaudio import transforms
from torchaudio.functional import resample, add_noise

from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import AudioProcessor
from clayrs.utils.automatic_methods import autorepr

__all__ = [
    "TorchResample",
    "TorchAmplitudeToDB",
    "TorchVol",
    "TorchFade",
    "TorchAddNoise",
    "TorchMuLawDecoding",
    "TorchMuLawEncoding",
    "TorchDeemphasis",
    "TorchPreemphasis",
    "TorchFFTConvolve",
    "TorchSpeed",
    "TorchConvolve",
    "TorchSpeedPerturbation",
    "TorchBuiltInTransformer",
    "ConvertToMono"
]


class TorchBuiltInTransformer(AudioProcessor):
    """
    Abstract class for Transformers preprocessors from the torchaudio library
    """

    def __init__(self, torch_builtin_transformer):
        super().__init__()

        self.transformer = torch_builtin_transformer

    def forward(self, field_data: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        Method to apply the given transformer to the data contained in the field

        NOTE: the data is expected to be a waveform Tensor and its associated sample rate
        """

        return self.transformer(field_data[0]), field_data[1]

    def __str__(self):
        return str(self.transformer)

    def __eq__(self, other):
        return self.transformer == other.transformer

    def __repr__(self):
        return repr(self.transformer)


class TorchTransformerRequiresSampleRate(TorchBuiltInTransformer):
    """
    Abstract class for Transformers preprocessors from the torchaudio library that need to know the sample rate
    of the input waveform for processing
    """

    def forward(self, field_data: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        Method to apply the given transformer to the data contained in the field

        NOTE: the data is expected to be an audio converted to a torch tensor and its associated sample rate
        """

        return self.transformer(field_data[0], field_data[1]), field_data[1]


class TorchAmplitudeToDB(TorchBuiltInTransformer):
    """
    Class that implements the AmplitudeToDB Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer AmplitudeToDB directly from torchaudio.

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.AmplitudeToDB.html#torchaudio.transforms.AmplitudeToDB)
    """

    def __init__(self, stype: str = 'power', top_db: Optional[float] = None):
        super().__init__(transforms.AmplitudeToDB(stype=stype, top_db=top_db))


class TorchResample(TorchTransformerRequiresSampleRate):
    """
    Class that implements the Resample Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Resample directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Resample.html#torchaudio.transforms.Resample)
    """

    def __init__(self, new_freq: int = 16000, resampling_method: str = 'sinc_interp_hann',
                 lowpass_filter_width: int = 6,
                 rolloff: float = 0.99, beta: Optional[float] = None):
        super().__init__(lambda x, sr: resample(waveform=x,
                                                orig_freq=sr,
                                                new_freq=new_freq,
                                                resampling_method=resampling_method,
                                                lowpass_filter_width=lowpass_filter_width,
                                                rolloff=rolloff,
                                                beta=beta))

        self.new_sample_rate = new_freq
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta

        self._repr_string = autorepr(self, inspect.currentframe())

    def forward(self, field_data: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        Method to apply the given transformer to the data contained in the field

        NOTE: the data is expected to be an audio converted to a torch tensor and its associated sample rate,
        this technique will not work on textual data
        """

        return self.transformer(field_data[0], field_data[1]), self.new_sample_rate

    def __repr__(self):
        return self._repr_string


class TorchMuLawEncoding(TorchBuiltInTransformer):
    """
    Class that implements the MuLawEncoding Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer MuLawEncoding directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.MuLawEncoding.html#torchaudio.transforms.MuLawEncoding)
    """

    def __init__(self, quantization_channels: int = 256):
        super().__init__(transforms.MuLawEncoding(quantization_channels=quantization_channels))


class TorchMuLawDecoding(TorchBuiltInTransformer):
    """
    Class that implements the MuLawDecoding Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer MuLawDecoding directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.MuLawDecoding.html#torchaudio.transforms.MuLawDecoding)
    """

    def __init__(self, quantization_channels: int = 256):
        super().__init__(transforms.MuLawDecoding(quantization_channels=quantization_channels))


class TorchFade(TorchBuiltInTransformer):
    """
    Class that implements the Fade Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Fade directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Fade.html#torchaudio.transforms.Fade)
    """

    def __init__(self, fade_in_len: int = 0, fade_out_len: int = 0, fade_shape: str = 'linear'):
        super().__init__(transforms.Fade(fade_in_len=fade_in_len, fade_out_len=fade_out_len, fade_shape=fade_shape))


class TorchVol(TorchBuiltInTransformer):
    """
    Class that implements the Vol Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Vol directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Vol.html#torchaudio.transforms.Vol)
    """

    def __init__(self, gain: float, gain_type: str = 'amplitude'):
        super().__init__(transforms.Vol(gain=gain, gain_type=gain_type))


class TorchAddNoise(TorchBuiltInTransformer):
    """
    Class that implements the add noise Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer add noise directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.functional.add_noise.html#torchaudio.functional.add_noise)
    """

    def __init__(self, noise: torch.Tensor, snr: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        super().__init__(lambda x: add_noise(x, noise, snr, lengths))


class TorchConvolve(TorchBuiltInTransformer):
    """
    Class that implements the Convolve Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Convolve directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Convolve.html#torchaudio.transforms.Convolve)
    """

    def __init__(self, y: torch.Tensor, mode: str = 'full'):
        super().__init__(lambda x: transforms.Convolve(mode)(x, y))


class TorchFFTConvolve(TorchBuiltInTransformer):
    """
    Class that implements the FFTConvolve Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer FFTConvolve directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.FFTConvolve.html#torchaudio.transforms.FFTConvolve)
    """

    def __init__(self, y: torch.Tensor, mode: str = 'full'):
        super().__init__(lambda x: transforms.FFTConvolve(mode)(x, y))


class TorchSpeed(TorchTransformerRequiresSampleRate):
    """
    Class that implements the Speed Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Speed directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    Note: lengths parameter of the forward call is not implemented

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Speed.html#torchaudio.transforms.Speed)
    """

    def __init__(self, factor: float):
        super().__init__(lambda x, sr: transforms.Speed(orig_freq=sr, factor=factor)(x))

    def forward(self, field_data: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        Method to apply the given transformer to the data contained in the field

        NOTE: the data is expected to be an audio converted to a torch tensor and its associated sample rate,
        this technique will not work on textual data
        """

        return self.transformer(field_data[0], field_data[1])[0], field_data[1]


class TorchSpeedPerturbation(TorchTransformerRequiresSampleRate):
    """
    Class that implements the SpeedPerturbation from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer SpeedPerturbation directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    Note: lengths parameter of the forward call is not implemented

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.SpeedPerturbation.html#torchaudio.transforms.SpeedPerturbation)
    """

    def __init__(self, factors: List[float]):
        super().__init__(lambda x, sr: transforms.SpeedPerturbation(orig_freq=sr, factors=factors)(x))

    def forward(self, field_data: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        Method to apply the given transformer to the data contained in the field

        NOTE: the data is expected to be an audio converted to a torch tensor and its associated sample rate,
        this technique will not work on textual data
        """

        return self.transformer(field_data[0], field_data[1])[0], field_data[1]


class TorchDeemphasis(TorchBuiltInTransformer):
    """
    Class that implements the Deemphasis Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Deemphasis directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Deemphasis.html#torchaudio.transforms.Deemphasis)
    """

    def __init__(self, coeff: float = 0.97):
        super().__init__(transforms.Deemphasis(coeff))


class TorchPreemphasis(TorchBuiltInTransformer):
    """
    Class that implements the Preemphasis Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Preemphasis directly from torchaudio (except for original_freq which is handled
    directly by the framework).

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Preemphasis.html#torchaudio.transforms.Preemphasis)
    """

    def __init__(self, coeff: float = 0.97):
        super().__init__(transforms.Preemphasis(coeff))


class ConvertToMono(TorchBuiltInTransformer):
    """
    Class that implements a conversion to Mono when dealing with multichannel waveforms (e.g. stereo).
    The mean of the channels is applied and returned.
    """

    def __init__(self):
        super().__init__(lambda x: torch.mean(x, dim=0, keepdim=True))

    def __str__(self):
        return "ConvertToMono"

    def __repr__(self):
        return "ConvertToMono()"
