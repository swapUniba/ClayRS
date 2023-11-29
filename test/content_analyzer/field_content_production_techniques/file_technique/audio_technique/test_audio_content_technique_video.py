import os
import torch
import shutil
import requests

from unittest import TestCase

from clayrs.content_analyzer import JSONFile, EmbeddingField
from clayrs.content_analyzer.information_processor.preprocessors.audio_preprocessors import \
    TorchResample
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.low_level_techniques import MFCC
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.high_level_techniques import \
    TorchAudioPretrained
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.audio_content_techniques import \
    ClasslessAudioFromVideoFolder

from test import dir_test_files
from test.content_analyzer.field_content_production_techniques.utils import create_full_path_source, create_video_dataset

raw_source_path_online = os.path.join(dir_test_files, 'test_videos', 'urls_dataset.json')
raw_source_path_local_rel = os.path.join(dir_test_files, 'test_videos', 'paths_dataset.json')
ds_path = os.path.join(dir_test_files, 'test_videos', 'videos', 'videoPath')
ds_path_without_field_name = os.path.join(dir_test_files, 'test_videos', 'videos')
videos_path = os.path.join(dir_test_files, 'test_videos', 'videos_files')

this_file_path = os.path.dirname(os.path.abspath(__file__))


class TestClasslessAudioFromVideoFolder(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_video_dataset()

    def test_dataset_videos(self):

        videos_paths_list = [os.path.join(videos_path, file_name) for file_name in os.listdir(videos_path)]

        # load the dataset and check that the 3 videos in it are correctly loaded
        ds = ClasslessAudioFromVideoFolder(videos_paths_list)
        self.assertEqual(3, len(ds))

        # load the dataset and check that the 3 videos in it are correctly loaded and the audios are
        # extracted and resampled
        ds = ClasslessAudioFromVideoFolder(videos_paths_list, [TorchResample(16000)])
        self.assertEqual(3, len(ds))
        # check that all audios have been resampled
        sample_rates_list = [ds[video_idx]['sample_rate'] for video_idx in range(0, len(ds))]
        self.assertTrue(all(sample_rate == 16000 for sample_rate in sample_rates_list))

        # test the dataset again but this time there will be a non-existing path,
        # the dataset will replace the desired video with a tensor full of zeros
        videos_paths_list.append(os.path.join(ds_path, 'not_existing_video_path'))

        ds = ClasslessAudioFromVideoFolder(videos_paths_list)

        waveforms_list = [ds[video_idx]['waveform'] for video_idx in range(0, len(ds))]
        self.assertEqual(4, len(ds))

        # ds.count keeps track of the number of paths that couldn't be resolved
        self.assertEqual(1, ds.count)

        # check that the last video that has been added only contains zeros
        # (the one associated to the non-existing path)
        self.assertTrue(torch.all(waveforms_list[-1] == 0))

        video_without_audio = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
        r = requests.get(video_without_audio)
        filename = video_without_audio.split('/')[-1]
        with open(filename, 'wb') as f:
            f.write(r.content)

        # test the dataset again but this time there will be a video without audio,
        # the dataset will replace the desired video with a tensor full of zeros
        videos_paths_list.append(filename)

        ds = ClasslessAudioFromVideoFolder(videos_paths_list)

        waveforms_list = [ds[video_idx]['waveform'] for video_idx in range(0, len(ds))]
        self.assertEqual(5, len(ds))

        # ds.count keeps track of the number of paths that couldn't be resolved
        self.assertEqual(2, ds.count)

        # check that the last video that has been added only contains zeros
        # (the one associated to the non-existing path)
        self.assertTrue(torch.all(waveforms_list[-1] == 0))

        os.remove(filename)


class TestVideoProcessingTechniquesAudio(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_video_dataset()
        cls.full_source_path = create_full_path_source(raw_source_path_local_rel, 'videoPath', videos_path)

    def test_video_high_level(self):

        # since videos are handled using audio techniques, it is just checked that a single high level technique
        # works with videos (to check the correctness of the processing done in case of video)

        source = JSONFile(self.full_source_path)

        technique = TorchAudioPretrained(flatten=False, contents_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('videoPath', [], source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))

        shutil.rmtree(ds_path_without_field_name)

    def test_video_low_level(self):

        # since videos are handled using audio techniques, it is just checked that a single low level technique
        # works with videos (to check the correctness of the processing done in case of video)

        source = JSONFile(self.full_source_path)

        technique = MFCC(flatten=False, contents_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('videoPath', [], source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))

        shutil.rmtree(ds_path_without_field_name)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.full_source_path)
