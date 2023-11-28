from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import List, TYPE_CHECKING, Callable, Tuple, Optional

import torch
import torchaudio.pipelines as pipe
from torchvision.models.feature_extraction import create_feature_extractor

from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.audio_content_techniques import \
    AudioContentTechnique
from clayrs.content_analyzer.content_representation.content import EmbeddingField
from clayrs.utils.automatic_methods import autorepr
from clayrs.utils.context_managers import get_progbar

if TYPE_CHECKING:
    from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import AudioProcessor
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource


class HighLevelAudio(AudioContentTechnique):
    """
    Technique which encapsulates the logic for all audio techniques that work at a high level, that is they work on
    batches of audios in an efficient way by leveraging pre-trained models
    """

    def __init__(self, model, feature_layer: int = -1, flatten: bool = False, device: str = 'cpu',
                 apply_on_output: Callable[[torch.Tensor], torch.Tensor] = None,
                 contents_dirs: str = "contents_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64):
        super().__init__(contents_dirs, time_tuple, max_timeout, max_retries, max_workers, batch_size, flatten)

        def return_self(x: torch.Tensor) -> torch.Tensor:
            return x

        self.apply_on_output = return_self if apply_on_output is None else apply_on_output

        self.model = model.to(device)
        self.feature_layer = feature_layer
        self.device = device
        self.flatten = flatten

    def apply_and_flatten(self, single_repr):

        single_repr = self.apply_on_output(single_repr)
        single_repr = single_repr.cpu().numpy()

        if self.flatten:

            # do not flatten if already flattened
            if len(single_repr.shape) != 1:
                if single_repr.shape[0] == 1:
                    single_repr = single_repr.flatten()
                else:
                    single_repr = single_repr.reshape(single_repr.shape[0], -1)

        return single_repr

    def produce_content(self, field_name: str, preprocessor_list: List[AudioProcessor], source: RawInformationSource) -> List[FieldRepresentation]:

        dl, _ = self.get_data_loader(field_name, source, preprocessor_list)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:

            for data_batch in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                # preprocessing is done by the dataset directly in the dataloader
                batch_waveform = data_batch['waveform']
                batch_sample_rate = data_batch['sample_rate']
                reprs = self.produce_batch_repr(batch_waveform, batch_sample_rate)

                for single_repr in reprs:
                    out_f_applied_single_repr = self.apply_and_flatten(single_repr)
                    representation_list.append(EmbeddingField(out_f_applied_single_repr))

        return representation_list

    @abstractmethod
    def produce_batch_repr(self, field_data: torch.Tensor, sample_rates: List[int]) -> List[FieldRepresentation]:
        """
        Method that will produce the corresponding complex representations from all field data for the batch
        """
        raise NotImplementedError


class TorchAudioPretrained(HighLevelAudio):
    """
    High level technique which uses the [torchaudio library] (https://pytorch.org/audio/stable/index.html) for feature
    extraction from audios using pre-trained models

    A list of available models can be found [here] (https://pytorch.org/audio/stable/pipelines.html)

    NOTE: the class has been mainly tested to work with wav2vec models, support for other models is not guaranteed

    Args:
        module_name: name of the pipeline module in TorchAudio from which the model will be retrieved
        feature_layer: the layer index from which the features will be retrieved
        flatten: whether the features obtained from the model should be flattened or not
        device: device where the model will be stored ('cpu', 'cuda:0', ...). If not specified, 'cpu' will be set
            by default
        apply_on_output: custom lambda function to be applied to the features output
        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        time_tuple: start and end second from which the data will be extracted
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the dataloader
    """

    def __init__(self, module_name: str = "WAV2VEC2_ASR_BASE_960H", feature_layer: int = -1, flatten: bool = True,
                 device: str = 'cpu', apply_on_output: Callable[[torch.Tensor], torch.Tensor] = None,
                 contents_dirs: str = "contents_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64):

        self.module = getattr(pipe, module_name)
        self.module_name = module_name
        model = self.module.get_model()

        super().__init__(model, feature_layer, flatten, device, apply_on_output, contents_dirs, time_tuple,
                         max_timeout, max_retries, max_workers, batch_size)

        self._repr_string = autorepr(self, inspect.currentframe())

    @torch.no_grad()
    def produce_batch_repr(self, field_data: torch.Tensor, sample_rates: List[int]) -> torch.Tensor:

        if len(field_data.shape) == 3:
            if field_data.shape[0] != 1:
                field_data = field_data.squeeze()
            else:
                field_data = field_data[0]

        return self.model.extract_features(field_data.to(self.device))[0][self.feature_layer]

    def __str__(self):
        return f"TorchAudio Pretrained Model ({self.module_name})"

    def __repr__(self):
        return self._repr_string


class VGGISH(HighLevelAudio):
    """
    Temporary class which implements the VGGISH model, at the moment, this is only supported in the "nightly" build of
    torchaudio.

    It is possible to find more information [here](https://pytorch.org/audio/main/generated/torchaudio.prototype.pipelines.VGGISH.html)

    Args:
        feature_layer: the layer index from which the features will be retrieved
        flatten: whether the features obtained from the model should be flattened or not
        device: device where the model will be stored ('cpu', 'cuda:0', ...). If not specified, 'cpu' will be set
            by default
        apply_on_output: custom lambda function to be applied to the features output
        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        time_tuple: start and end second from which the data will be extracted
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the dataloader
    """

    def __init__(self, feature_layer: int = -1, flatten: bool = True,
                 device: str = 'cpu', apply_on_output: Callable[[torch.Tensor], torch.Tensor] = None,
                 contents_dirs: str = "contents_dirs",  time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64):

        import torchaudio.prototype.pipelines as p

        self.input_processor = p.VGGISH.get_input_processor()
        original_model = p.VGGISH.get_model()

        if isinstance(feature_layer, int):
            feature_layer = list(dict(original_model.named_modules()).keys())[feature_layer]

        model = create_feature_extractor(original_model, {feature_layer: 'feature_layer'}).to(device).eval()

        super().__init__(model, feature_layer, flatten, device, apply_on_output, contents_dirs, time_tuple,
                         max_timeout, max_retries, max_workers, batch_size)

        self._repr_string = autorepr(self, inspect.currentframe())

    @torch.no_grad()
    def produce_batch_repr(self, field_data: torch.Tensor, sample_rates: List[int]) -> torch.Tensor:

        if len(field_data.shape) == 3:
            if field_data.shape[0] != 1:
                field_data = field_data.squeeze()
            else:
                field_data = field_data[0]

        field_data_prepared = [self.input_processor(x) for x in field_data]
        split_size = field_data_prepared[0].size()[0]
        field_data_prepared = torch.vstack(field_data_prepared)

        output = self.model(field_data_prepared.to(self.device))['feature_layer']
        output = output.view(output.shape[0] // split_size, split_size, output.shape[1])

        return output

    def __str__(self):
        return f"TorchAudio Pretrained Model (VGGISH)"

    def __repr__(self):
        return self._repr_string
