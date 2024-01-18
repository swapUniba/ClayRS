import os
import shutil
from unittest import TestCase
from unittest.mock import patch

import requests

import torch
import torchaudio

from clayrs.content_analyzer.information_processor.preprocessors.audio_preprocessors import \
    TorchResample
from clayrs.content_analyzer.raw_information_source import JSONFile
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.audio_content_techniques import \
    ClasslessAudioFromFolder
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques import MFCC

from test import dir_test_files, dir_root_repo
from test.content_analyzer.field_content_production_techniques.utils import create_full_path_source, MockedGoodResponse, \
    MockedUnidentifiedResponse, create_audio_dataset

raw_source_path_online = os.path.join(dir_test_files, 'test_audios', 'urls_dataset.json')
raw_source_path_local_rel = os.path.join(dir_test_files, 'test_audios', 'paths_dataset.json')
ds_path = os.path.join(dir_test_files, 'test_audios', 'audios', 'audioPath')
ds_path_without_field_name = os.path.join(dir_test_files, 'test_audios', 'audios')
audios_path = os.path.join(dir_test_files, 'test_audios', 'audios_files')

this_file_path = os.path.dirname(os.path.abspath(__file__))


class TestClasslessAudioFromFolder(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_audio_dataset()

    def test_dataset(self):

        audios_paths_list = [os.path.join(audios_path, file_name) for file_name in os.listdir(audios_path)]

        # load the dataset and check that the 3 audios in it are correctly loaded
        ds = ClasslessAudioFromFolder(audios_paths_list)
        self.assertEqual(3, len(ds))

        # load the dataset and check that the 3 audios in it are correctly loaded and resampled
        ds = ClasslessAudioFromFolder(audios_paths_list, preprocessor_list=[TorchResample(16000)])
        self.assertEqual(3, len(ds))
        # check that all audios have been resampled
        sample_rates_list = [ds[audio_idx]['sample_rate'] for audio_idx in range(0, len(ds))]
        self.assertTrue(all(sample_rate == 16000 for sample_rate in sample_rates_list))

        # test the dataset again but this time there will be a non-existing path,
        # the dataset will replace the desired audio with a tensor full of zeros
        audios_paths_list.append(os.path.join(ds_path, 'not_existing_audio_path'))

        ds = ClasslessAudioFromFolder(audios_paths_list)
        waveforms_list = [ds[audio_idx]['waveform'] for audio_idx in range(0, len(ds))]
        self.assertEqual(4, len(ds))

        # ds.count keeps track of the number of paths that couldn't be resolved
        self.assertEqual(1, ds.count)

        # check that the last audio that has been added only contains zeros
        # (the one associated to the non-existing path)
        self.assertTrue(torch.all(waveforms_list[3] == 0))


class TestAudioProcessingTechniques(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_audio_dataset()
        cls.full_source_path = create_full_path_source(raw_source_path_local_rel, 'audioPath', audios_path)

    @patch.object(requests, "get", return_value=MockedGoodResponse(audios_path))
    def test_downloading_audios_good_response(self, mocked_response):

        # use the json file containing online links to locally download the audios

        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since AudioContentTechnique class is abstract
        dl, _ = MFCC(contents_dirs=os.path.join(this_file_path, 'audios_dirs')).get_data_loader('audioUrl', source, [])

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'audios_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'audios_dirs', 'audioUrl')))
        self.assertEqual(3, len([audio for audio in dl]))

        for audio in dl:
            self.assertFalse(torch.all(audio['waveform'] == 0))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'audios_dirs'))

    @patch.object(requests, "get", return_value=MockedUnidentifiedResponse())
    def test_downloading_audios_not_audio_response(self, mocked_response):

        # use the json file containing online links to locally download the audios

        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since AudioContentTechnique class is abstract
        dl, _ = MFCC(contents_dirs=os.path.join(this_file_path, 'audios_dirs')).get_data_loader('audioUrl', source, [])

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'audios_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'audios_dirs', 'audioUrl')))
        self.assertEqual(3, len([audio for audio in dl]))

        for audio in dl:
            self.assertTrue(torch.all(audio['waveform'] == 0))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'audios_dirs'))

    @patch.object(requests, "get", side_effect=requests.exceptions.ConnectionError())
    def test_downloading_audios_connection_error(self, mocked_response):

        # use the json file containing online links to locally download the audios

        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since AudioContentTechnique class is abstract
        dl, _ = MFCC(contents_dirs=os.path.join(this_file_path, 'audios_dirs')).get_data_loader('audioUrl', source, [])

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'audios_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'audios_dirs', 'audioUrl')))
        self.assertEqual(3, len([audio for audio in dl]))

        for audio in dl:
            self.assertTrue(torch.all(audio['waveform'] == 0))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'audios_dirs'))

    def test_loading_audios_full_path(self):

        # use the json file containing local paths to load the audios
        source_with_new_paths = JSONFile(self.full_source_path)

        dl, _ = MFCC(contents_dirs=ds_path_without_field_name).get_data_loader('audioPath', source_with_new_paths, [])

        waveforms_in_dl = [audio['waveform'] for audio in dl]

        self.assertTrue(not all(torch.all(audio == 0) for audio in waveforms_in_dl))
        self.assertEqual(3, len(waveforms_in_dl))

        # check that at least one of the loaded audios matches one of the audios from the dataset

        test_audio_path = os.path.join(ds_path, os.listdir(ds_path)[0])

        test_audio, sr = torchaudio.load(test_audio_path)
        test_audio = test_audio.unsqueeze(0)

        self.assertTrue(any(torch.equal(test_audio, waveform_in_dl) for waveform_in_dl in waveforms_in_dl))
        shutil.rmtree(ds_path_without_field_name)

    def test_loading_audios_relative_path(self):

        # this test uses relative path so change the working dir
        current_workdir = os.getcwd()
        os.chdir(dir_root_repo)

        # use the json file containing local paths to load the audios

        source_with_new_paths = JSONFile(raw_source_path_local_rel)

        dl, _ = MFCC(contents_dirs=ds_path_without_field_name).get_data_loader('audioPath', source_with_new_paths, [])
        audios_in_dl = [audio['waveform'] for audio in dl]

        self.assertTrue(not all(torch.all(audio[0][0] == 0) for audio in audios_in_dl))
        self.assertEqual(3, len(audios_in_dl))

        # check that at least one of the loaded audios matches one of the audios from the dataset

        audio_filename = os.listdir(audios_path)[0]
        test_audio_path = os.path.join(audios_path, audio_filename)
        test_audio, original_sample_rate = torchaudio.load(test_audio_path)
        test_audio = test_audio.unsqueeze(0)

        self.assertTrue(any(torch.equal(test_audio, audio_in_dl) for audio_in_dl in audios_in_dl))

        shutil.rmtree(ds_path_without_field_name)

        # reset the previous workdir
        os.chdir(current_workdir)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.full_source_path)
