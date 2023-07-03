from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
from typing import Tuple, List, TYPE_CHECKING

import timm
import torch
import cv2
import numpy as np

from clayrs_can_see.content_analyzer.content_representation.content import EmbeddingField
from clayrs_can_see.content_analyzer.field_content_production_techniques.visual_techniques.visual_content_techniques import \
    VisualContentTechnique
from clayrs_can_see.utils.context_managers import get_progbar

if TYPE_CHECKING:
    from clayrs_can_see.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs_can_see.content_analyzer.raw_information_source import RawInformationSource
    from clayrs_can_see.content_analyzer.information_processor.information_processor_abstract import ImageProcessor
    from clayrs_can_see.content_analyzer.information_processor.postprocessors.postprocessor import \
        EmbeddingInputPostProcessor


class HighLevelVisual(VisualContentTechnique):
    """
    Technique which encapsulates the logic for all visual techniques that work at a high level, that is they work on
    batches of images in an efficient way
    """

    def __init__(self, imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):
        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)

    def produce_content(self, field_name: str, preprocessor_list: List[ImageProcessor],
                        postprocessor_list: List[EmbeddingInputPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        dl = self.get_data_loader(field_name, source)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:
            for data_batch in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                processed_data = self.process_data(data_batch, preprocessor_list)
                representation_list.extend(self.produce_batch_repr(processed_data))

        representation_list = self.postprocess_representations(representation_list, postprocessor_list)

        return representation_list

    @abstractmethod
    def produce_batch_repr(self, field_data: torch.Tensor) -> List[FieldRepresentation]:
        """
        Method that will produce the corresponding complex representations from all field data for the batch
        """
        raise NotImplementedError


class PytorchImageModels(HighLevelVisual):
    """
    High level technique which uses the [timm library] (https://timm.fast.ai/) for feature extraction from images using
    pre-trained models

    Args:
        model_name: a model name supported by the timm library
        feature_layer: the layer index from which the features will be retrieved
            NOTE: the model is loaded from the timm library with the parameter "features_only" set at True, meaning
            that only feature layers of the model will be available and accessible through the index
        flatten: whether the features obtained from the model should be flattened or not
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, model_name: str, feature_layer: int = -1, flatten: bool = True, device: str = 'cpu',
                 imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        original_model = timm.create_model(model_name, pretrained=True)

        feature_layer = list(original_model._modules.keys())[feature_layer]

        layers = {}
        for layer_name, layer in original_model._modules.items():
            layers[layer_name] = layer
            if layer_name == feature_layer:
                break

        self.model = torch.nn.Sequential(OrderedDict(layers))

        self.model.to(device)
        self.device = device
        self.flatten = flatten

    def produce_batch_repr(self, field_data: torch.Tensor) -> List[EmbeddingField]:

        if self.flatten:
            return list(map(lambda x: EmbeddingField(x.cpu().detach().numpy().flatten()),
                            self.model(field_data.to(self.device))))
        else:
            return list(map(lambda x: EmbeddingField(x.cpu().detach().numpy()),
                            self.model(field_data.to(self.device))))

    def __str__(self):
        return "Pytorch Image Models"

    def __repr__(self):
        return "Pytorch Image Models"


class CaffeImageModels(HighLevelVisual):

    def __init__(self, prototxt_path: str, model_path: str, feature_layer: str = None, mean_file_path: str = None,
                 swap_rb: bool = False, flatten: bool = True, imgs_dirs: str = "imgs_dirs", use_gpu: bool = False,
                 max_timeout: int = 2, max_retries: int = 5, max_workers: int = 0, batch_size: int = 64,
                 resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self.model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        if use_gpu:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.feature_layer = feature_layer
        self.mean_file_path = mean_file_path
        self.swapRB = swap_rb
        self.flatten = flatten

    def produce_batch_repr(self, field_data: torch.Tensor) -> List[EmbeddingField]:

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
            return [EmbeddingField(x.flatten()) for x in features_output]
        else:
            return [EmbeddingField(x) for x in features_output]

    def __str__(self):
        return "Caffe Image Models"

    def __repr__(self):
        return "Caffe Image Models"
