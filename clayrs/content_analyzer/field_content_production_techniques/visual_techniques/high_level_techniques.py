from __future__ import annotations
import inspect
import os
from abc import abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING, Callable, Tuple, Optional

import cv2
import timm
import torch
import numpy as np
import torchvision.models.video as v_models
import torchvision.models.optical_flow as opt_f_models
from torchvision.models import get_model_weights
from timm.models._helpers import load_checkpoint
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Lambda, Compose, ToTensor

from clayrs.content_analyzer.content_representation.content import EmbeddingField
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.visual_techniques.visual_content_techniques import \
    VisualContentTechnique
from clayrs.utils.context_managers import get_progbar
from clayrs.utils.automatic_methods import autorepr

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import \
        ImageProcessor


class HighLevelVisual(VisualContentTechnique):
    """
    Technique which encapsulates the logic for all visual techniques that work at a high level, that is they work on
    batches of images in an efficient way by leveraging pre-trained models
    """

    def __init__(self, model, feature_layer: int = -1, flatten: bool = False, device: str = 'cpu',
                 apply_on_output: Callable[[torch.Tensor], torch.Tensor] = None,
                 contents_dirs: str = "contents_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64):
        super().__init__(contents_dirs, time_tuple, max_timeout, max_retries, max_workers, batch_size)

        def return_self(x: torch.Tensor) -> torch.Tensor:
            return x

        self.apply_on_output = return_self if apply_on_output is None else apply_on_output

        self.model = model
        self.feature_layer = feature_layer
        self.device = device
        self.flatten = flatten

    def apply_and_flatten(self, single_repr):

        single_repr = self.apply_on_output(single_repr)
        single_repr = single_repr.numpy()

        if self.flatten:

            # do not flatten if already flattened
            if len(single_repr.shape) != 1:
                # single global feature vector
                if single_repr.shape[0] == 1:
                    single_repr = single_repr.flatten()
                # multiple local descriptors
                else:
                    single_repr = single_repr.reshape(single_repr.shape[0], -1)

        return single_repr

    @abstractmethod
    def produce_batch_repr(self, field_data: torch.Tensor) -> torch.Tensor:
        """
        Method that will produce the corresponding complex representations from all field data for the batch
        """
        raise NotImplementedError


class HighLevelVisualFrame(HighLevelVisual):
    """
    High level visual techniques are categorized into three subclasses:

        - Frame level techniques
        - Clip level techniques
        - Flow level techniques

    Frame level techniques return an output for each single image or single video frame,
    so, for example, if resnet is used as technique, for each input frame the corresponding output
    features will be extracted
    """

    def produce_content(self, field_name: str, preprocessor_list: List[ImageProcessor], source: RawInformationSource) -> List[FieldRepresentation]:

        dl, video_mode = self.get_data_loader(field_name, source, preprocessor_list)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:
            for processed_data_batch in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                if not video_mode:
                    # preprocessing is done by the dataset directly in the dataloader
                    reprs = self.produce_batch_repr(processed_data_batch).unsqueeze(1)
                else:

                    processed_data_batch, idxs = processed_data_batch

                    # clips generated from video
                    if len(processed_data_batch.shape) == 5:
                        frames_size = processed_data_batch.size()[1]
                        processed_data_batch = torch.flatten(processed_data_batch, start_dim=0, end_dim=1)
                        reprs = self.produce_batch_repr(processed_data_batch)
                        reprs = torch.split(reprs, list(np.array(idxs) * frames_size))
                    # entire video
                    else:
                        reprs = self.produce_batch_repr(processed_data_batch)
                        reprs = torch.split(reprs, idxs)

                for single_repr in reprs:
                    out_f_applied_single_repr = self.apply_and_flatten(single_repr)
                    representation_list.append(EmbeddingField(out_f_applied_single_repr))

        return representation_list

    @abstractmethod
    def produce_batch_repr(self, field_data: torch.Tensor) -> torch.Tensor:
        """
        Method that will produce the corresponding complex representations from all field data for the batch
        """
        raise NotImplementedError


class HighLevelVisualClip(HighLevelVisual):
    """
    High level visual techniques are categorized into three subclasses:

        - Frame level techniques
        - Clip level techniques
        - Flow level techniques

    Clip level techniques return an output for each clip extracted from the video.
    These techniques should be used only after pre-processing carried out by a clip sampler (that
    extracts a collection of consecutive video frames).

    NOTE: the technique will not raise an error when processing single images or entire videos, but
    the inputs should be properly processed beforehand (using the ClipSampler preprocessor for instance)
    """

    def __init__(self, model, feature_layer: int = -1, flatten: bool = False, device: str = 'cpu',
                 apply_on_output: Callable[[torch.Tensor], torch.Tensor] = None,
                 contents_dirs: str = "contents_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, mini_batch_size: int = 64):

        super().__init__(model, feature_layer, flatten, device, apply_on_output,
                         contents_dirs, time_tuple, max_timeout, max_retries, max_workers, batch_size)

        self.mini_batch_size = mini_batch_size

    def produce_content(self, field_name: str, preprocessor_list: List[ImageProcessor], source: RawInformationSource) -> List[FieldRepresentation]:

        dl, video_mode = self.get_data_loader(field_name, source, preprocessor_list)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:
            for processed_data_batch in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                if not video_mode:
                    # preprocessing is done by the dataset directly in the dataloader
                    reprs = self.produce_batch_repr(processed_data_batch.unsqueeze(1)).unsqueeze(1)
                else:

                    processed_data_batch, idxs = processed_data_batch

                    # clips generated from video
                    if len(processed_data_batch.shape) == 5:
                        reprs = self.produce_batch_repr(processed_data_batch)
                        reprs = torch.split(reprs, idxs)
                    # entire video
                    else:
                        processed_data_batch = torch.split(processed_data_batch, idxs)
                        reprs = []
                        for video in processed_data_batch:
                            reprs.append(self.produce_batch_repr(video.unsqueeze(0)))

                for single_repr in reprs:
                    out_f_applied_single_repr = self.apply_and_flatten(single_repr)
                    representation_list.append(EmbeddingField(out_f_applied_single_repr))

        return representation_list

    @abstractmethod
    def produce_batch_repr(self, field_data: torch.Tensor) -> torch.Tensor:
        """
        Method that will produce the corresponding complex representations from all field data for the batch
        """
        raise NotImplementedError


class PytorchImageModels(HighLevelVisualFrame):
    """
    High level technique which uses the [timm library] (https://timm.fast.ai/) for feature extraction from images or
    video frames using pre-trained models

    Args:
        model_name: a model name supported by the timm library
        feature_layer: the layer index from which the features will be retrieved
            NOTE: the model is loaded from the timm library with the parameter "features_only" set at True, meaning
            that only feature layers of the model will be available and accessible through the index
        custom_weights_path: path where the weights are stored locally, if not specified the default weights for the
            model on the timm library will be used
        use_default_transforms: if True, the default preprocessing operations for the model will be applied, otherwise
            nothing will be done on the inputs. Defaults to False.
        flatten: whether the features obtained from the model should be flattened or not
        device: device where the model will be stored ('cpu', 'cuda:0', ...). If not specified, 'cpu' will be set
            by default
        apply_on_output: custom lambda function to be applied to the features output
        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the dataloader
    """

    def __init__(self, model_name, feature_layer: int = -1, flatten: bool = False, device: str = 'cpu',
                 apply_on_output: Callable[[torch.Tensor], torch.Tensor] = None,
                 contents_dirs: str = "contents_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 custom_weights_path: str = None,
                 use_default_transforms: bool = False,
                 num_classes: int = None,
                 max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64):

        if custom_weights_path is None:
            original_model = timm.create_model(model_name, pretrained=True)
        else:
            original_model = timm.create_model(model_name,
                                               checkpoint_path=custom_weights_path,
                                               num_classes=num_classes)

        self.default_transform = None

        if use_default_transforms:
            default_transform = timm.data.create_transform(
                **timm.data.resolve_data_config(original_model.pretrained_cfg))
            default_transform = [x if type(x) != ToTensor else Lambda(lambda x: x.div(255.0)) for x in
                                 default_transform.transforms]
            self.default_transform = Compose(default_transform)

        self.original_model = original_model

        if isinstance(feature_layer, int):
            feature_layer = list(dict(original_model.named_modules()).keys())[feature_layer]

        model = create_feature_extractor(original_model, {feature_layer: "feature_layer"}).to(device).eval()

        super().__init__(model, feature_layer, flatten, device, apply_on_output, contents_dirs, time_tuple,
                         max_timeout, max_retries, max_workers, batch_size)

        self.model_name = model_name
        self._repr_string = autorepr(self, inspect.currentframe())

    @torch.no_grad()
    def produce_batch_repr(self, field_data: torch.Tensor) -> torch.Tensor:

        if self.default_transform is not None:
            field_data = self.default_transform(field_data)

        out = self.model(field_data.to(self.device))['feature_layer']

        return out.cpu()

    def __str__(self):
        return f"Pytorch Image Models ({self.model_name})"

    def __repr__(self):
        return self._repr_string


class CaffeImageModels(HighLevelVisualFrame):
    """
    High level technique which use pre-trained Caffe models for feature extraction from images or video frames using
    pre-trained models

    Args:
        prototxt_path: path to the prototxt of the model
        model_path: path to the Caffe model
        feature_layer: the layer index from which the features will be retrieved
            NOTE: the model is loaded from the timm library with the parameter "features_only" set at True, meaning
            that only feature layers of the model will be available and accessible through the index
        mean_file_path: path where the mean pixel file related to the dataset on which the model has been trained on
        flatten: whether the features obtained from the model should be flattened or not
        apply_on_output: custom lambda function to be applied to the features output
        swap_rb: instead of RGB use GBR
        use_gpu: use GPU as device instead of CPU
        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the dataloader
    """

    def __init__(self, prototxt_path: str, model_path: str, feature_layer: str = None, mean_file_path: str = None,
                 apply_on_output: Callable[[torch.Tensor], torch.Tensor] = None,
                 swap_rb: bool = False, flatten: bool = True,
                 time_tuple: Tuple[Optional[int], Optional[int]] = (0, None),
                 use_gpu: bool = False,
                 contents_dirs: str = "contents_dirs",
                 max_timeout: int = 2, max_retries: int = 5, max_workers: int = 0,
                 batch_size: int = 64):

        model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.model_name = Path(model_path).name

        if use_gpu:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        super().__init__(model, feature_layer, flatten, None, apply_on_output,
                         contents_dirs, time_tuple, max_timeout, max_retries, max_workers, batch_size)

        self.mean_file_path = mean_file_path
        self.swapRB = swap_rb
        self.flatten = flatten

        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_batch_repr(self, field_data: torch.Tensor) -> torch.Tensor:

        if self.mean_file_path is not None:
            mean = np.load(self.mean_file_path).mean(1).mean(1)
        else:
            mean = None

        if mean is not None:
            imgs_blob = cv2.dnn.blobFromImages(np.moveaxis(field_data.numpy(), 1, -1), mean=mean, swapRB=self.swapRB)
        else:
            imgs_blob = cv2.dnn.blobFromImages(np.moveaxis(field_data.numpy(), 1, -1), swapRB=self.swapRB)

        self.model.setInput(imgs_blob)

        if self.feature_layer is None:
            features_output = self.model.forward()
        else:
            features_output = self.model.forward(self.feature_layer)

        if self.flatten:
            return torch.from_numpy(features_output).flatten(1, -1)
        else:
            return torch.from_numpy(features_output)

    def __str__(self):
        return f"Caffe Image Models ({self.model_name})"

    def __repr__(self):
        return self._repr_string
