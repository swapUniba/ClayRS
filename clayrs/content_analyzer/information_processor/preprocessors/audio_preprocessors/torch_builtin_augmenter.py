from torchaudio import transforms

from typing import Optional

from clayrs.content_analyzer.information_processor.preprocessors.audio_preprocessors import \
    TorchBuiltInTransformer

__all__ = [
    "TorchFrequencyMasking",
    "TorchTimeMasking",
    "TorchTimeStretch"
]


class TorchFrequencyMasking(TorchBuiltInTransformer):
    """
    Class that implements the FrequencyMasking Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer FrequencyMasking directly from torchaudio.

    TorchAudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.FrequencyMasking.html#torchaudio.transforms.FrequencyMasking)

    NOTE: the augmented result will SUBSTITUTE the original input
    """
    def __init__(self, freq_mask_param: int, iid_mask: bool = False, **kwargs):
        super().__init__(lambda x: transforms.FrequencyMasking(freq_mask_param, iid_mask)(transforms.Spectrogram(**kwargs)(x)))


class TorchTimeMasking(TorchBuiltInTransformer):
    """
    Class that implements the TimeMasking Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer TimeMasking directly from torchaudio.

    torchaudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.TimeMasking.html#torchaudio.transforms.TimeMasking)

    NOTE: the augmented result will SUBSTITUTE the original input
    """
    def __init__(self, time_mask_param: int, iid_masks: bool = False, p: float = 1.0, **kwargs):
        super().__init__(lambda x: transforms.TimeMasking(time_mask_param, iid_masks, p)(transforms.Spectrogram(**kwargs)(x)))


class TorchTimeStretch(TorchBuiltInTransformer):
    """
    Class that implements the TimeStretch Transformer from torchaudio.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer TimeStretch directly from torchaudio.

    torchaudio documentation: [here](https://pytorch.org/audio/stable/generated/torchaudio.transforms.TimeStretch.html#torchaudio.transforms.TimeStretch)

    NOTE: the augmented result will SUBSTITUTE the original input
    """
    def __init__(self, hop_length: Optional[int] = None, n_freq: int = 201, fixed_rate: Optional[float] = None, **kwargs):
        super().__init__(lambda x: transforms.TimeStretch(hop_length, n_freq, fixed_rate)(transforms.Spectrogram(**kwargs)(x)))
