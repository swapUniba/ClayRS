from __future__ import annotations

import io
import os
from typing import Tuple, List, TYPE_CHECKING

from abc import abstractmethod
import PIL.Image
import requests
import validators
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
import torchvision.transforms.functional as TF

if TYPE_CHECKING:
    from clayrs_can_see.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs_can_see.content_analyzer.raw_information_source import RawInformationSource
    from clayrs_can_see.content_analyzer.information_processor.information_processor_abstract import InformationProcessor, \
        ImageProcessor
    from clayrs_can_see.content_analyzer.information_processor.postprocessors.postprocessor import \
        EmbeddingInputPostProcessor

from clayrs_can_see.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    FieldContentProductionTechnique
from clayrs_can_see.utils.const import logger
from clayrs_can_see.utils.context_managers import get_iterator_thread


class ClasslessImageFolder(Dataset):
    """
    Dataset which is used by torch dataloaders to efficiently handle images.
    In this case, since labels are not of interest, only the image in the form of a Torch tensor will be returned.
    """
    def __init__(self, image_paths: List[str], resize_size: Tuple[int, int]):

        self.image_paths = image_paths

        self.resize_size = list(resize_size)
        self.count = 0

    def __getitem__(self, index):
        """
        open the image from the list of image path of the specified index

        if the image exists at the specified path, it is:

            1. converted to RGB
            2. converted to tensor
            3. resized to a default size

        if the image doesn't exist at the specified path, a blank image is created of the specified default size
        """
        image_path = self.image_paths[index]

        try:
            x = PIL.Image.open(image_path).convert("RGB")
            x = TF.pil_to_tensor(x)
            x = TF.resize(x, self.resize_size)
        except FileNotFoundError:
            self.count += 1
            x = torch.zeros((3, self.resize_size[0], self.resize_size[1]))

        return x

    def __len__(self):
        return len(self.image_paths)


class VisualContentTechnique(FieldContentProductionTechnique):
    """
    Class which encapsulates the logic for techniques which use images as input

    In order for the framework to process images, the input source (a CSV file, for example) should contain
    for a field all values of a certain type, that are either:

        - paths to images stored locally;
        - links to images.

    In the case of file paths, the images will simply be loaded. In the case of links, the images will first be
    downloaded locally and then loaded.

    The Visual techniques will then be extended into two different kinds of techniques:

        - Low Level: low level processing techniques which require analyzing each image separately;
        - High Level: high level processing techniques which can efficiently compute batches of images.

    IMPORTANT NOTE: if the technique can't properly load some images (because the download links are not working or
    because it is not available locally) they will be replaced with a three-dimensional Torch Tensor consisting of zeros
    only

    Args:

        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
        to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline, the
        size specified in the latter will be the final resize size
    """

    def __init__(self, imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        self.imgs_dirs = imgs_dirs
        self.max_timeout = max_timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.batch_size = batch_size
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
        """
        Method which retrieves all the images for the specified field name in the raw source in case images are not
        paths but links
        """
        def dl_and_save_images(url_or_path):

            filename = Path(url_or_path).name
            filename_no_extension = Path(url_or_path).stem

            image_path = os.path.join(field_imgs_dir, filename)
            image_path_lnk = os.path.join(field_imgs_dir, filename_no_extension + ".lnk")

            if os.path.isfile(image_path):
                return image_path
            elif os.path.isfile(image_path_lnk):
                return image_path_lnk

            if validators.url(url_or_path):

                n_retry = 0

                # keep trying to download an image until the max_retries number is reached
                while True:
                    try:
                        byte_img = requests.get(url_or_path, timeout=self.max_timeout).content
                        img = PIL.Image.open(io.BytesIO(byte_img))
                        img_path = os.path.join(self.imgs_dirs, Path(field_name).name)

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

                # if the path is not absolute, we go in this if
                if not os.path.isfile(url_or_path):
                    # we build the absolute path and check again if the image exist
                    path_dir_imgs = str(Path(raw_source.file_path).parent.absolute())
                    url_or_path = str(os.path.join(path_dir_imgs, url_or_path))

                    if not os.path.isfile(url_or_path):
                        return None

                file_link = Path(url_or_path).stem + ".lnk"
                os.link(url_or_path, os.path.join(self.imgs_dirs, field_name, file_link))

                return os.path.join(self.imgs_dirs, field_name, file_link)

        field_imgs_dir = os.path.join(self.imgs_dirs, field_name)

        os.makedirs(field_imgs_dir, exist_ok=True)

        url_images = (content[field_name] for content in raw_source)

        img_paths = []
        error_count = 0
        with get_iterator_thread(self.max_workers, dl_and_save_images, url_images,
                                 keep_order=True, progress_bar=True, total=len(raw_source)) as pbar:

            pbar.set_description("Downloading/Locating images")

            for future in pbar:
                if not future:
                    error_count += 1
                else:
                    img_paths.append(future)

        if error_count != 0:
            logger.warning(f"Number of images that couldn't be retrieved: {error_count}")

        return img_paths

    def get_data_loader(self, field_name: str, raw_source: RawInformationSource):
        """
        Method to retrieve the dataloader for the images in the specified field name of the raw source.

        If the images are organized as paths, the
        """
        # IMPORTANT: we must give to the data loader the same ordering of contents
        # in the raw source, otherwise the content analyzer assigns to an item
        # a representation of another!
        # TO DO: maybe save images with the id of the content rather than the filename of the image?
        image_paths = self._retrieve_images(field_name, raw_source)

        ds = ClasslessImageFolder(image_paths=image_paths,
                                  resize_size=self.resize_size)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        return dl

    @abstractmethod
    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        postprocessor_list: List[EmbeddingInputPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
