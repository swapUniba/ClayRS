from __future__ import annotations
import inspect
from abc import abstractmethod
from typing import List, TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torchaudio

from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.audio_content_techniques import \
    AudioContentTechnique
from clayrs.content_analyzer.content_representation.content import EmbeddingField
from clayrs.utils.automatic_methods import autorepr
from clayrs.utils.context_managers import get_progbar

if TYPE_CHECKING:
    from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import AudioProcessor
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource


class LowLevelAudio(AudioContentTechnique):
    """
    Technique which encapsulates the logic for all audio techniques that work at a low level in an algorithmic way
    """

    def __init__(self, contents_dirs: str = "contents_dirs",  time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 max_timeout: int = 2, max_retries: int = 5, max_workers: int = 0, flatten: bool = False):
        super().__init__(contents_dirs, time_tuple, max_timeout, max_retries, max_workers, 1, flatten)

    def produce_content(self, field_name: str, preprocessor_list: List[AudioProcessor], source: RawInformationSource) -> List[FieldRepresentation]:

        dl, _ = self.get_data_loader(field_name, source, preprocessor_list)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:

            for data_batch in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                # preprocessing is done by the dataset directly in the dataloader
                waveform = data_batch['waveform'][0]
                sample_rate = data_batch['sample_rate'][0].item()
                representation_list.append(self.produce_single_repr(waveform, sample_rate))

        return representation_list

    @abstractmethod
    def produce_single_repr(self, field_data: torch.Tensor, sample_rate: int) -> FieldRepresentation:
        """
        Method that will produce the corresponding complex representations from all field data for the batch
        """
        raise NotImplementedError


class MFCC(LowLevelAudio):
    """
    Low level technique which implements the MFCC using the TorchAudio library

    Parameters are the same ones you would pass to the MFCC transformer in TorchAudio together with some framework
    specific parameters, except for the sample rate which is handled directly by the framework

    Arguments for [TorchAudio MFCC](https://pytorch.org/audio/master/generated/torchaudio.transforms.MFCC.html)

    NOTE: differently from TorchAudio, framework output when flatten=False is of shape (x, time, n_mfcc) where x is > 1,
        otherwise if x=1 (time, n_mfcc)

    Args:
        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        time_tuple: start and end second from which the waveform data will be extracted
        flatten: whether the output of the technique should be flattened or not
        mean: performs mean of the coefficients for each channel (this is done only when flatten=False)
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
    """

    def __init__(self, contents_dirs: str = "contents_dirs",  time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 flatten: bool = False, mean: bool = False,
                 max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0,
                 n_mfcc: int = 40, dct_type: int = 2,
                 norm: str = 'ortho', log_mels: bool = False, melkwargs: Optional[dict] = None):

        super().__init__(contents_dirs, time_tuple, max_timeout, max_retries, max_workers, flatten=flatten)

        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_mels = log_mels
        self.melkwargs = melkwargs

        self.mean = mean

        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor, sample_rate: int) -> FieldRepresentation:

        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                          n_mfcc=self.n_mfcc,
                                          dct_type=self.dct_type,
                                          norm=self.norm,
                                          log_mels=self.log_mels,
                                          melkwargs=self.melkwargs)

        if self.flatten:
            repr = mfcc(field_data.float()).numpy().flatten()
        else:
            repr = np.moveaxis(mfcc(field_data.float()).numpy(), 1, -1)

            if self.mean:
                repr = np.mean(repr, axis=0)
            elif repr.shape[0] == 1:
                repr = repr[0]

        return EmbeddingField(repr)

    def __str__(self):
        return "MFCC"

    def __repr__(self):
        return self._repr_string
