from __future__ import annotations

import inspect
import io
import os
from typing import Tuple, List, Union, TYPE_CHECKING, Any

from abc import abstractmethod
import PIL.Image
import requests
import validators
import timm
from skimage.feature import hog, canny, SIFT, local_binary_pattern
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import torchvision.transforms.functional as TF
from scipy.ndimage.filters import convolve

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.information_processor_abstract import InformationProcessor, ImageProcessor
    from clayrs.content_analyzer.information_processor.visual_postprocessors.visualpostprocessor import \
        EmbeddingInputPostProcessor

from clayrs.content_analyzer.content_representation.content import EmbeddingField
from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    FieldContentProductionTechnique
from clayrs.utils.automatic_methods import autorepr
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_iterator_thread, get_progbar


class ClasslessImageFolder(Dataset):
    def __init__(self, root, resize_size: Tuple[int, int], all_images_list: list = None):
        self.image_paths = [os.path.join(root, file_name) for file_name in all_images_list]
        self.resize_size = list(resize_size)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        try:
            x = PIL.Image.open(image_path).convert("RGB")
            x = TF.to_tensor(x)
            x = TF.resize(x, self.resize_size)
        except FileNotFoundError:
            x = torch.zeros((3, self.resize_size[0], self.resize_size[1]))

        y = 0

        return x, y

    def __len__(self):
        return len(self.image_paths)


class VisualContentTechnique(FieldContentProductionTechnique):

    def __init__(self, imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, resize_size: Tuple[int, int] = (100, 100)):

        self.imgs_dirs = imgs_dirs
        self.max_timeout = max_timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.resize_size = resize_size

    @staticmethod
    def process_data(data, preprocessor_list: List[ImageProcessor]):
        """
        The data passed as argument is processed using the preprocessor list (also given as argument) and is
        then returned

        Args:
            data (str): data on which each preprocessor, in the preprocessor list, will be used
            preprocessor_list (List[InformationProcessor]): list of preprocessors to apply to the data

        Returns:
            processed_data (Union[List[str], str): it could be a list of tokens created from the original data
            or the data in string form
        """
        transformers_seq = torch.nn.Sequential(*preprocessor_list)
        processed_data = transformers_seq(data)

        return processed_data

    def _retrieve_images(self, field_name: str, raw_source: RawInformationSource):
        def dl_and_save_images(url_or_path):
            if validators.url(url_or_path):

                n_retry = 0

                while True:
                    try:
                        byte_img = requests.get(url_or_path, timeout=self.max_timeout).content
                        img = PIL.Image.open(io.BytesIO(byte_img))
                        img_path = os.path.join(self.imgs_dirs, field_name, url_or_path.split("/")[-1])

                        img.save(img_path)

                        return img_path

                    except PIL.UnidentifiedImageError:
                        logger.warning(f"Found a Url which is not an image! URL: {url_or_path}")
                        return None

                    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                        if n_retry < self.max_retries:
                            n_retry += 1
                        else:
                            logger.warning(f"Max number of retries reached ({n_retry + 1}/{self.max_retries}) for URL: "
                                           f"{url_or_path}\nThe image will be skipped")
                            return None
            else:
                return None

        field_imgs_dir = os.path.join(self.imgs_dirs, field_name)
        try:
            os.makedirs(field_imgs_dir)
        except OSError:
            raise FileExistsError(f'{field_imgs_dir} already exists') from None

        url_images = (content[field_name] for content in raw_source)

        error_count = 0
        with get_iterator_thread(self.max_workers, dl_and_save_images, url_images,
                                 progress_bar=True, total=len(raw_source)) as pbar:

            pbar.set_description("Downloading images")

            for future in pbar:
                if not future:
                    error_count += 1

        if error_count != 0:
            logger.warning(f"Failed requests: {error_count}")

    def get_data_loader(self, field_name: str, raw_source: RawInformationSource):
        field_images_dir = os.path.join(self.imgs_dirs, field_name)

        if not os.path.isdir(field_images_dir):
            self._retrieve_images(field_name, raw_source)

        ds = ClasslessImageFolder(root=field_images_dir,
                                  all_images_list=[content[field_name].split('/')[-1] for content in raw_source],
                                  resize_size=self.resize_size)
        dl = DataLoader(ds, batch_size=32)

        return dl

    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        postprocessor_list: List[EmbeddingInputPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        pass

    def __repr__(self):
        pass


class LowLevelVisual(VisualContentTechnique):

    def produce_content(self, field_name: str, preprocessor_list: List[ImageProcessor],
                        postprocessor_list: List[EmbeddingInputPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:

        dl = self.get_data_loader(field_name, source)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:

            for (data_batch, _) in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                for content_data in data_batch:
                    processed_data = self.process_data(content_data, preprocessor_list)
                    representation_list.append(self.produce_single_repr(processed_data))

        representation_list = self.postprocess_representations(representation_list, postprocessor_list)

        return representation_list

    @abstractmethod
    def produce_single_repr(self, field_data: Union[List[str], str]) -> FieldRepresentation:
        raise NotImplementedError


class HighLevelVisual(VisualContentTechnique):

    def produce_content(self, field_name: str, preprocessor_list: List[ImageProcessor],
                        postprocessor_list: List[EmbeddingInputPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        dl = self.get_data_loader(field_name, source)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:
            for (data_batch, _) in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                processed_data = self.process_data(data_batch, preprocessor_list)
                representation_list.extend(self.produce_batch_repr(processed_data))

        return representation_list

    @abstractmethod
    def produce_batch_repr(self, field_data: Union[List[str], str]) -> List[FieldRepresentation]:
        raise NotImplementedError


class PytorchImageModels(HighLevelVisual):

    def __init__(self, model_name: str, feature_layer: int = -1):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.feature_layer = feature_layer

    def produce_batch_repr(self, field_data: torch.Tensor) -> List[FieldRepresentation]:

        if len(field_data.shape) == 3:
            field_data = field_data.unsqueeze(dim=1)

        return list(map(lambda x: EmbeddingField(x.detach().numpy()),
                        self.model(field_data.squeeze())[self.feature_layer]))

    def __str__(self):
        return "PytorchImageModels"


class SkImageHogDescriptor(LowLevelVisual):

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys',
                 transform_sqrt=False, flatten=False):
        self.hog = lambda x, channel_axis: hog(x, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                               cells_per_block=cells_per_block, block_norm=block_norm,
                                               transform_sqrt=transform_sqrt, feature_vector=True,
                                               channel_axis=channel_axis)
        self.flatten = flatten

        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        channel_axis = None
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            channel_axis = 0

        hog_features = self.hog(field_data.numpy(), channel_axis=channel_axis)
        hog_features = hog_features.flatten() if self.flatten else hog_features
        return EmbeddingField(hog_features)

    def __str__(self):
        return "SkImageHogDescriptor"

    def __repr__(self):
        return self._repr_string


class SkImageCannyEdgeDetector(LowLevelVisual):

    def __init__(self, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False,
                 mode='constant', cval=0.0, flatten=False):
        self.canny = lambda x: canny(image=x, sigma=sigma, low_threshold=low_threshold,
                                     high_threshold=high_threshold, mask=mask,
                                     use_quantiles=use_quantiles, mode=mode, cval=cval)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        canny_features = self.canny(field_data.numpy())
        canny_features = canny_features.flatten() if self.flatten else canny_features
        return EmbeddingField(canny_features.astype(int))

    def __str__(self):
        return "SkImageCannyEdgeDetector"

    def __repr__(self):
        return self._repr_string


class CustomFilterConvolution(LowLevelVisual):

    def __init__(self, weights, mode='reflect', cval=0.0, origin=0, flatten=False):
        self.convolve = lambda x: convolve(x, weights=weights, mode=mode, cval=cval, origin=origin)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        convolution_result = self.convolve(field_data.numpy())
        convolution_result = convolution_result.flatten() if self.flatten else convolution_result
        return EmbeddingField(convolution_result.astype(int))

    def __str__(self):
        return "CustomFilterConvolution"

    def __repr__(self):
        return self._repr_string


class SkImageMainColors(LowLevelVisual):

    def __init__(self):
        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()

        if field_data.shape[0] != 3:
            raise Exception('Image not in RGB')

        numpy_field_data = field_data.numpy() * 255
        colors = np.array([numpy_field_data[0, :, :].flatten(),
                           numpy_field_data[1, :, :].flatten(),
                           numpy_field_data[2, :, :].flatten()])
        return EmbeddingField(colors.astype(int))

    def __str__(self):
        return "SkImageMainColors"

    def __repr__(self):
        return self._repr_string


class SkImageColorQuantization(LowLevelVisual):

    def __init__(self, n_colors: Any = 3, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto",
                 flatten=False):
        self.k_means = KMeans(n_clusters=n_colors, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                              random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()

        if field_data.shape[0] != 3:
            raise Exception('Image in grey scale')

        numpy_field_data = field_data.numpy() * 255
        numpy_field_data = numpy_field_data.reshape((numpy_field_data.shape[1] * numpy_field_data.shape[2], 3))
        self.k_means.fit(numpy_field_data)
        dominant_colors = self.k_means.cluster_centers_
        dominant_colors = dominant_colors.flatten() if self.flatten else dominant_colors
        return EmbeddingField(dominant_colors.astype(int))

    def __str__(self):
        return "SkImageColorQuantization"

    def __repr__(self):
        return self._repr_string


class SkImageSIFT(LowLevelVisual):

    def __init__(self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5, c_dog=0.013333333333333334,
                 c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8,
                 flatten=False):
        self.sift = SIFT(upsampling=upsampling, n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min,
                         sigma_in=sigma_in, c_dog=c_dog, c_edge=c_edge, n_bins=n_bins, lambda_ori=lambda_ori,
                         c_max=c_max, lambda_descr=lambda_descr, n_hist=n_hist, n_ori=n_ori)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        self.sift.detect_and_extract(field_data.numpy())
        descriptors = self.sift.descriptors
        descriptors = descriptors.flatten() if self.flatten else descriptors
        return EmbeddingField(descriptors.astype(int))

    def __str__(self):
        return "SkImageSIFT"

    def __repr__(self):
        return self._repr_string


class SkImageLBP(LowLevelVisual):

    def __init__(self, p: int, r: float, method='default', flatten=False):
        self.lbp = lambda x: local_binary_pattern(x, P=p, R=r, method=method)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        patterns = self.lbp(field_data.numpy())
        patterns = patterns.flatten() if self.flatten else patterns
        return EmbeddingField(patterns.astype(int))

    def __str__(self):
        return "SkImageLBP"

    def __repr__(self):
        return self._repr_string


