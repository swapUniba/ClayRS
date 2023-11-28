from __future__ import annotations
from abc import abstractmethod
from typing import Tuple, List, TYPE_CHECKING, Optional

import PIL.Image
import filetype
import numpy as np
import torch
import torchvision
import itertools
from torch.utils.data import DataLoader, Dataset

from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    FileContentTechnique
from clayrs.content_analyzer.information_processor.preprocessors.visual_preprocessors.image_preprocessors.torch_builtin_transformer import \
    TorchResize
from clayrs.content_analyzer.information_processor.preprocessors.visual_preprocessors.image_preprocessors.torch_builtin_transformer import \
    TorchCenterCrop
from clayrs.utils.const import logger

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import VisualProcessor


class ClasslessImageFolder(Dataset):
    """
    Dataset which is used by torch dataloaders to efficiently handle images.
    In this case, since labels are not of interest, only the image in the form of a Torch tensor will be returned.
    """

    def __init__(self, image_paths: List[str], preprocessor_list: List[VisualProcessor] = None):

        self.image_paths = image_paths
        self.preprocessor_list = preprocessor_list

        if self.preprocessor_list is None:
            self.preprocessor_list = []

        self.count = 0

    def __getitem__(self, index):
        """
        Open the image from the list of image paths of the specified index

        if the image exists at the specified path, it is:

            1. converted to RGB
            2. converted to tensor

        if the image doesn't exist at the specified path, a blank image is created of size (3, 100, 100)

        In both cases, after the torch tensor corresponding to the image has been created, it will be processed
        using the defined preprocessors
        """
        image_path = self.image_paths[index]

        try:
            x = PIL.Image.open(image_path).convert("RGB")
            x = torch.from_numpy(np.array(x)).moveaxis(-1, 0)
        except (FileNotFoundError, AttributeError, Exception):
            self.count += 1
            x = torch.zeros((3, 100, 100))

        x = VisualContentTechnique.process_data(x.unsqueeze(0), self.preprocessor_list)

        return x[0]

    def __len__(self):
        return len(self.image_paths)


class ClasslessImageFromVideoFolder(Dataset):
    """
    Dataset which is used by torch dataloaders to efficiently handle videos.
    In this case, since labels are not of interest, only the video frames in the form of a Torch tensor will be returned.

    Additionally, it is possible to specify the start and end second to retrieve frames from using the "pts" parameter
    """

    def __init__(self, video_paths: List[str], pts: Tuple[Optional[int], Optional[int]],
                 preprocessor_list: List[VisualProcessor] = None):

        self.video_paths = video_paths
        self.preprocessor_list = preprocessor_list

        if self.preprocessor_list is None:
            self.preprocessor_list = []

        self.start_pts, self.end_pts = pts

        self.count = 0

    def __getitem__(self, index):
        """
        Open the video from the list of video paths of the specified index

        if the video exists at the specified path, the video frames are extracted from the range start_pts to end_pts
        if the video doesn't exist at the specified path, a blank frame is created of size (1, 3, 100, 100)

        In both cases, after the torch tensor corresponding to the frames has been created, it will be processed
        using the defined preprocessors
        """

        video_path = self.video_paths[index]

        try:

            with open(video_path, "rb") as f:

                reader = torchvision.io.VideoReader(f.read(), "video")

                if self.end_pts is None:
                    all_frames = [frame['data'] for frame in reader if frame['pts'] >= self.start_pts]
                else:
                    all_frames = [frame['data'] for frame in itertools.takewhile(lambda x: x['pts'] <= self.start_pts, reader.seek(self.end_pts))]

                all_frames = torch.stack(all_frames)
                all_frames = VisualContentTechnique.process_data(all_frames, self.preprocessor_list)

        except Exception as e:
            self.count += 1
            all_frames = torch.zeros((1, 3, 100, 100))
            all_frames = VisualContentTechnique.process_data(all_frames, self.preprocessor_list)

        return all_frames

    def __len__(self):
        return len(self.video_paths)


class VisualContentTechnique(FileContentTechnique):
    """
    Class which encapsulates the logic for techniques which use visual content as input (visual modality)

    This class processes both image and video files, in the case of image input it will simply be loaded, while for videos
    the frames will be extracted from it and loaded

    In order for the framework to process images or videos, the input source (a CSV file, for example) should contain
    a field with all values of a certain type, that are either:

        - paths to files stored locally (either relative or absolute paths)
        - links to files

    In the case of file paths, the contents will simply be loaded. In the case of links, the files will first be
    downloaded locally and then loaded.

    Note that all files should refer to a single type of input (image / video), the framework will process all of them
    according to the type of the first file in the input source.

    IMPORTANT NOTE: if the technique can't properly load some files (because the download links are not working or
    because they are not available locally) they will be replaced with the "default" value returned by the corresponding
    dataset class

    Args:
        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        time_tuple: start and end second from which the frames will be extracted (only for videos, doesn't affect images)
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the dataloader
    """

    def __init__(self, contents_dirs: str = "contents_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (0, None), max_timeout: int = 2,
                 max_retries: int = 5, max_workers: int = 0, batch_size: int = 64):

        super().__init__(contents_dirs, time_tuple, max_timeout, max_retries, max_workers, batch_size)

    @staticmethod
    def process_data(data: torch.Tensor, preprocessor_list: List[VisualProcessor]) -> torch.Tensor:
        """
        The data passed as argument is processed using the preprocessor list (also given as argument) and is
        then returned

        Args:
            data (torch.Tensor): data on which each preprocessor, in the preprocessor list, will be used
            preprocessor_list (List[VisualProcessor]): list of preprocessors to apply to the data

        Returns:
            processed_data (torch.Tensor): tensor representing the final processed data
        """
        transformers_seq = torch.nn.Sequential(*preprocessor_list)
        processed_data = transformers_seq(data)

        return processed_data

    def get_data_loader(self, field_name: str, raw_source: RawInformationSource,
                        preprocessor_list: List[VisualProcessor]) -> Tuple[DataLoader, bool]:
        """
        Method to retrieve the dataloader for the images or videos in the specified field name of the raw source.

        The retrieve_contents method will handle the different type of possible references to the contents in the source
        (paths or links), while this method will check whether the data are images or videos and create a suited
        data loader accordingly
        """
        # IMPORTANT: we must give to the data loader the same ordering of contents
        # in the raw source, otherwise the content analyzer assigns to an item
        # a representation of another!
        # TO DO: maybe save images/videos with the id of the content rather than the filename?

        content_paths = self._retrieve_contents(field_name, raw_source)

        first_not_null_file = next((content_path for content_path in content_paths if content_path is not None), None)

        video_mode = False

        if first_not_null_file is not None:

            try:
                test_file = filetype.guess(first_not_null_file)
                if test_file.MIME.startswith('video'):
                    video_mode = True
            except Exception as e:
                pass

        # if batch size is greater than 1, the technique must be a high level one (since they are the only ones that work on batch),
        # if no resizing operation is specified, the dataloader may not be able to load the data since they would have
        # different dimensions, a warning message is issued to warn the user of this scenario
        if self.batch_size > 1 and not any(isinstance(x, TorchResize) or isinstance(x, TorchCenterCrop) for x in preprocessor_list):
            logger.warning(
                "A high level technique is being used but no frame resize operation is specified, "
                "if the inputs have different dimensions, this will lead to a PyTorch dataloader error!"
                "Please add a TorchResize or TorchCenterCrop pre-processor to avoid this"
            )

        if not video_mode:

            ds = ClasslessImageFolder(image_paths=content_paths, preprocessor_list=preprocessor_list)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        else:

            # return frames stacked over time dimension and idxs of the stacked frames for each video
            def collate_videos(batch):
                return torch.vstack(batch), [x.shape[0] for x in batch]

            ds = ClasslessImageFromVideoFolder(video_paths=content_paths, preprocessor_list=preprocessor_list, pts=self.time_tuple)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_videos)

        return dl, video_mode

    @abstractmethod
    def produce_content(self, field_name: str, preprocessor_list: List[VisualProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
