import inspect
import random

import torch
import pytorchvideo.transforms.functional

from typing import Optional

from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import VideoProcessor
from clayrs.utils.automatic_methods import autorepr

__all__ = [
    "TorchUniformTemporalSubSampler",
    "ClipSampler",
]


class VideoSampler(VideoProcessor):
    """
    Class that encapsulates video sampling techniques
    """

    def __init__(self):
        super().__init__()


class VideoFrameSampler(VideoSampler):
    """
    Class that encapsulates sampling techniques which select frames from videos
    """

    def __init__(self):
        super().__init__()


class VideoClipSampler(VideoSampler):
    """
    Class that encapsulates sampling techniques which select clips of frames from videos
    """

    def __init__(self):
        super().__init__()


class TorchUniformTemporalSubSampler(VideoFrameSampler):
    """
    Class which allows to select a subset of frames sampled over the time dimension of the input video.

    If the video consists of 300 frames and 30 frames are to be sampled (specified through the 'num_samples' parameter),
    then 30 frames will be extracted uniformly (which means that a frame will be extracted for each series of
    10 consecutive frames).

    Args:
        - num_samples (int): number of frames to extract from the input video. If float type or negative number, a
            ValueError exception is raised
    """

    def __init__(self, num_samples: int):

        super().__init__()

        if type(num_samples) != int or num_samples < 0:
            raise ValueError(f"num_samples must be a positive integer but got {str(num_samples)}")

        self.num_samples = num_samples
        self._repr_string = autorepr(self, inspect.currentframe())

    def forward(self, field_data: torch.Tensor) -> torch.Tensor:
        return pytorchvideo.transforms.functional.uniform_temporal_subsample(field_data, self.num_samples, temporal_dim=0)

    def __str__(self):
        return 'TorchUniformTemporalSubSample'

    def __repr__(self):
        return self._repr_string


class ClipSampler(VideoClipSampler):
    """
    Class to sample clips consisting of a certain number of frames from an input video.

    The sampler divides the video frames into equally sized clips consisting of n frames, where
    n can be set through the 'num_frames_for_clip' parameter.

    It is also possible to select only a subset of all frames by specifying a 'selection_strategy':
        - "sequential": the first k clips will be selected;
        - "random": k random clips will be selected.

    k can be set using the 'num_clips' parameter, if this is not set then all clips will be returned.

    Args:
        - num_frames_for_clip (int): number of frames which will be kept for each clip
        - num_clips (Optional[int]): number of clips to keep out of all extracted clips, if None all clips will be kept
        - selection_strategy (str): strategy to select a subset of clips to keep out of all available ones
    """

    available_selection_strategies = ["sequential", "random"]

    def __init__(self, num_frames_for_clip: int, num_clips: Optional[int] = None, selection_strategy: str = "sequential"):

        super().__init__()

        if type(num_frames_for_clip) != int or num_frames_for_clip < 0:
            raise ValueError(f"num_frames_for_clip must be a positive integer but got {str(num_clips)}")

        if num_clips is not None and (type(num_clips) != int or num_clips < 0):
            raise ValueError(f"num_clips must be a positive integer but got {str(num_clips)}")

        if selection_strategy not in self.available_selection_strategies:
            raise ValueError(f"{selection_strategy} not in available selection strategy, choose one of {self.available_selection_strategies}")

        self.num_frames_for_clip = num_frames_for_clip
        self.num_clips = num_clips
        self.selection_strategy = selection_strategy

        self._repr_string = autorepr(self, inspect.currentframe())

    def forward(self, field_data: torch.Tensor) -> torch.Tensor:

        field_data_split = field_data.split(self.num_frames_for_clip)

        # drop last batch of frames if size is not the required one
        if field_data_split[-1].shape[0] != self.num_frames_for_clip:
            field_data_split = field_data_split[:-1]

        if self.num_clips is not None:

            if self.selection_strategy == "random":
                field_data_split = random.sample(field_data_split, self.num_clips)
            elif self.selection_strategy == "sequential":
                field_data_split = field_data_split[:self.num_clips]

        return torch.stack(field_data_split)

    def __str__(self):
        return 'ClipSampler'

    def __repr__(self):
        return self._repr_string
