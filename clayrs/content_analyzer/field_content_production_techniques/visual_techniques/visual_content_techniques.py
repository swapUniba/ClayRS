from __future__ import annotations

import inspect
import io
import os
from typing import Tuple, List, Union, TYPE_CHECKING

from abc import abstractmethod
import PIL.Image
import requests
import validators
import timm
from skimage.feature import hog, canny
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms.functional as TF

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.information_processor import InformationProcessor, ImageProcessor
    from clayrs.content_analyzer.information_processor.visualpostprocessor import VisualPostProcessor

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
        if not isinstance(processed_data, torch.Tensor):
            processed_data = TF.to_tensor(processed_data)

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
                        postprocessor_list: List[VisualPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        pass

    def __repr__(self):
        pass


class LowLevelVisual(VisualContentTechnique):

    def produce_content(self, field_name: str, preprocessor_list: List[ImageProcessor],
                        postprocessor_list: List[VisualPostProcessor],
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
                        postprocessor_list: List[VisualPostProcessor],
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

    def __init__(self, model_name: str):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')

    def produce_batch_repr(self, field_data: torch.Tensor) -> List[FieldRepresentation]:

        if len(field_data.shape) == 3:
            field_data = field_data.unsqueeze(dim=1)

        return list(map(lambda x: EmbeddingField(x.detach().numpy()), self.model(field_data.squeeze())))

    def __str__(self):
        return "PytorchImageModels"


class SkImageHogDescriptor(LowLevelVisual):

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys',
                 transform_sqrt=False):
        self.hog = lambda x, channel_axis: hog(x, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                               cells_per_block=cells_per_block, block_norm=block_norm,
                                               transform_sqrt=transform_sqrt, feature_vector=True,
                                               channel_axis=channel_axis)

        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        channel_axis = None
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            channel_axis = 0

        hog_features = self.hog(field_data.numpy(), channel_axis=channel_axis)
        return EmbeddingField(hog_features)

    def __str__(self):
        return "SkImageHogDescriptor"

    def __repr__(self):
        return self._repr_string


class SkImageCannyEdgeDetector(LowLevelVisual):

    def __init__(self, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False,
                 mode='constant', cval=0.0):
        self.canny = lambda x: canny(image=x, sigma=sigma, low_threshold=low_threshold,
                                     high_threshold=high_threshold, mask=mask,
                                     use_quantiles=use_quantiles, mode=mode, cval=cval)
        self._repr_string = autorepr(self, inspect.currentframe())
        super().__init__()

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        canny_features = self.canny(field_data.numpy()).flatten()
        return EmbeddingField(canny_features.astype(int))

    def __str__(self):
        return "SkImageCannyEdgeDetector"

    def __repr__(self):
        return self._repr_string
