from __future__ import annotations

import itertools
from typing import List, TYPE_CHECKING, Tuple, Optional

from abc import abstractmethod

import filetype
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import torchaudio

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import InformationProcessor, \
        AudioProcessor

from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    FileContentTechnique


class ClasslessAudioFromFolder(Dataset):
    """
    Dataset which is used by torch dataloaders to efficiently handle audios.
    In this case, since labels are not of interest, a dictionary will be returned with the following data:

        'waveform': Torch tensor containing the waveform associated with the audio
        'sample_rate': Integer, original sample rate for the audio

    Additionally, it is possible to specify the start and end second of the waveform using the "pts" parameter
    """

    def __init__(self, audio_paths: List[str], start_pts=0, end_pts=None, preprocessor_list: List[AudioProcessor] = None):

        self.audio_paths = audio_paths
        self.preprocessor_list = preprocessor_list

        if self.preprocessor_list is None:
            self.preprocessor_list = []

        self.start_pts = start_pts

        if end_pts is None:
            self.end_pts = -1
        else:
            self.end_pts = end_pts

        self.count = 0

    def __getitem__(self, index):
        """
        Load the audio from the list of audio paths of the specified index

        if the audio exists at the specified path, it is loaded and its sample rate stored

        if the audio doesn't exist at the specified path, a blank waveform is created of size (1, 16000) and
        with sample rate 16000

        In both cases, after the torch tensor corresponding to the audio has been created, it will be processed
        using the defined preprocessors
        """

        audio_path = self.audio_paths[index]

        try:

            metadata = torchaudio.info(audio_path)
            sample_rate = metadata.sample_rate
            start_frame = self.start_pts * sample_rate

            num_frames = -1

            if self.end_pts != -1:
                num_frames = self.end_pts * sample_rate

            with open(audio_path, "rb") as f:

                waveform, original_sample_rate = torchaudio.load(f,
                                                                 frame_offset=start_frame,
                                                                 num_frames=num_frames)

        except (FileNotFoundError, AttributeError, Exception):
            self.count += 1
            original_sample_rate = 16000
            waveform = torch.zeros((1, original_sample_rate))

        new_waveform, final_sample_rate = AudioContentTechnique.process_data((waveform, original_sample_rate), self.preprocessor_list)

        x = {'waveform': new_waveform, 'sample_rate': final_sample_rate}

        return x

    def __len__(self):
        return len(self.audio_paths)


class ClasslessAudioFromVideoFolder(Dataset):
    """
    Dataset which is used by torch dataloaders to efficiently handle audio from videos.
    In this case, since labels are not of interest, a dictionary will be returned with the following data:

        'waveform': Torch tensor containing the waveform associated with the audio
        'sample_rate': Integer, original sample rate for the audio

    Additionally, it is possible to specify the start and end second of the waveform using the "pts" parameter
    """

    def __init__(self, video_paths: List[str], start_pts=0, end_pts=None, preprocessor_list: List[AudioProcessor] = None):

        self.video_paths = video_paths
        self.preprocessor_list = preprocessor_list

        if self.preprocessor_list is None:
            self.preprocessor_list = []

        self.start_pts = start_pts
        self.end_pts = end_pts

        self.count = 0

    def __getitem__(self, index):
        """
        Open the video from the list of video paths of the specified index

        if the video exists at the specified path, the audio and the sample rate are extracted

        if the video doesn't exist at the specified path, a blank waveform is created of size (1, 16000) and
        with sample rate 16000

        In both cases, after the torch tensor corresponding to the audio has been created, it will be processed
        using the defined preprocessors
        """

        video_path = self.video_paths[index]

        try:

            with open(video_path, "rb") as f:
                reader = torchvision.io.VideoReader(f.read(), "audio")

                if self.end_pts is None:
                    waveform = [frame['data'] for frame in reader if frame['pts'] >= self.start_pts]
                else:
                    waveform = [frame['data'] for frame in itertools.takewhile(lambda x: x['pts'] >= self.start_pts, reader.seek(self.end_pts))]

                waveform = torch.vstack(waveform).moveaxis(0, 1)

                # if video doesn't have audio treat it as a missing video
                if not waveform.numel():
                    raise AttributeError

                original_sample_rate = reader.get_metadata()['audio']['framerate']

        except (FileNotFoundError, AttributeError, Exception) as e:
            self.count += 1
            original_sample_rate = 16000
            waveform = torch.zeros((1, original_sample_rate))

        new_waveform, final_sample_rate = AudioContentTechnique.process_data((waveform, original_sample_rate), self.preprocessor_list)

        x = {'waveform': new_waveform, 'sample_rate': final_sample_rate}

        return x

    def __len__(self):
        return len(self.video_paths)


class AudioContentTechnique(FileContentTechnique):
    """
    Class which encapsulates the logic for techniques which use audios as input

    This class accepts both video and audio formats, in the case of audio it will simply be loaded, while for videos
    the audio will be extracted from it and loaded

    In order for the framework to process audios or videos, the input source (a CSV file, for example) should contain
    a field with all values of a certain type, that are either:

        - paths to files stored locally (either relative or absolute paths)
        - links to files

    In the case of file paths, the contents will simply be loaded. In the case of links, the files will first be
    downloaded locally and then loaded.

    Note that all files should refer to a single type of input (audio / video), the framework will process all of them
    according to the type of the first file in the input source.

    The Audio techniques will then be extended into two different kinds of techniques:

        - Low Level: low level processing techniques which require analyzing each item separately
        - High Level: high level processing techniques which can efficiently compute batches of items

    IMPORTANT NOTE: if the technique can't properly load some files (because the download links are not working or
    because they are not available locally) they will be replaced with the "default" value returned by the corresponding
    dataset class

    IMPORTANT NOTE: if a batch size greater than 1 is specified, the waveforms will be automatically padded so that
    they can be stacked in a batch (if you need precise features use a batch size equal to 1)

    Args:
        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the dataloader
    """

    def __init__(self, contents_dirs: str = "audios_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (0, None), max_timeout: int = 2,
                 max_retries: int = 5, max_workers: int = 0, batch_size: int = 64, flatten: bool = False):
        super().__init__(contents_dirs, time_tuple, max_timeout, max_retries, max_workers, batch_size, flatten)

    @staticmethod
    def process_data(data: Tuple[torch.Tensor, int], preprocessor_list: List[AudioProcessor]) -> Tuple[torch.Tensor, int]:
        """
        The data passed as argument is processed using the preprocessor list (also given as argument) and is
        then returned

        Args:
            data (Tuple[torch.Tensor, int]): data on which each preprocessor, in the preprocessor list, will be used
            preprocessor_list (List[AudioProcessor]): list of preprocessors to apply to the data

        Returns:
            processed_data (Tuple[torch.Tensor, int]): processed waveform and sample rate (may be modified by some
                pre-processing operation such as resampling)
        """
        transformers_seq = torch.nn.Sequential(*preprocessor_list)
        processed_data = transformers_seq(data)

        return processed_data

    def get_data_loader(self, field_name: str, raw_source: RawInformationSource, preprocessor_list: List[AudioProcessor]) -> Tuple[DataLoader, bool]:
        """
        Method to retrieve the dataloader for the audios in the specified field name of the raw source.

        The retrieve_contents method will handle the different types of possible references to the contents in the source
        (paths or links), while this method will check whether the data are audios or videos and create a suited
        data loader accordingly
        """
        # IMPORTANT: same issue as visual_technique
        audio_paths = self._retrieve_contents(field_name, raw_source)

        first_not_null_file = next((audio_path for audio_path in audio_paths if audio_path is not None), None)

        video_mode = False

        if first_not_null_file is not None:

            try:
                test_file = filetype.guess(first_not_null_file)
                if test_file.MIME.startswith('video'):
                    video_mode = True
            except Exception:
                pass

        # TO-DO: find a way to process audio in batches in a way that doesn't influence output representation
        def pad_sequence(batch):
            waveform_batch = []
            sample_rate_batch = []
            for item in batch:
                waveform_batch.append(item['waveform'].t())
                sample_rate_batch.append(item['sample_rate'])

            waveform_batch = torch.nn.utils.rnn.pad_sequence(waveform_batch, batch_first=True, padding_value=0.)
            return {'waveform': waveform_batch.permute(0, 2, 1), 'sample_rate': sample_rate_batch}

        if not video_mode:
            ds = ClasslessAudioFromFolder(audio_paths=audio_paths, preprocessor_list=preprocessor_list,
                                          start_pts=self.time_tuple[0], end_pts=self.time_tuple[1])
        else:
            ds = ClasslessAudioFromVideoFolder(video_paths=audio_paths, preprocessor_list=preprocessor_list,
                                               start_pts=self.time_tuple[0], end_pts=self.time_tuple[1])

        if self.batch_size > 1:
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=pad_sequence)
        else:
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        return dl, video_mode

    @abstractmethod
    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
