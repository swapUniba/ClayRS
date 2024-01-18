import os
import shutil
import unittest

import numpy as np
import torch
import torchaudio

from clayrs.content_analyzer import JSONFile, EmbeddingField
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.low_level_techniques import MFCC
from test.content_analyzer.field_content_production_techniques.file_technique.audio_technique.test_audio_content_technique import \
    create_full_path_source, ds_path_without_field_name, raw_source_path_local_rel, audios_path

this_file_path = os.path.dirname(os.path.abspath(__file__))


class TestLowLevelAudioTechnique(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        full_path_source = create_full_path_source(raw_source_path_local_rel, 'audioPath', audios_path)

        # pre-build the list of audios from the source, this is done for
        # comparing outputs obtained from the techniques
        new_source = JSONFile(full_path_source)

        audios_list = []
        audios_list_resampled = []
        sample_rates = []

        for data in new_source:
            audio_path = data['audioPath']
            audio, original_sample_rate = torchaudio.load(audio_path)
            resampled_audio = torchaudio.functional.resample(audio, original_sample_rate, 16000)
            audios_list.append(audio)
            audios_list_resampled.append(resampled_audio)
            sample_rates.append(original_sample_rate)

        cls.audios_list = audios_list
        cls.audios_list_padded = torch.nn.utils.rnn.pad_sequence([x.t() for x in audios_list],
                                                                 batch_first=True,
                                                                 padding_value=0.).permute(0, 2, 1)

        cls.audios_list_resampled = audios_list_resampled
        cls.audios_list_padded_resampled = torch.nn.utils.rnn.pad_sequence([x.t() for x in audios_list_resampled],
                                                                           batch_first=True,
                                                                           padding_value=0.).permute(0, 2, 1)

        cls.sample_rates = sample_rates
        cls.full_path_source = full_path_source

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ds_path_without_field_name)
        os.remove(cls.full_path_source)


class TestMFCC(TestLowLevelAudioTechnique):

    def test_mfcc(self):

        # test output of MFCC technique

        source = JSONFile(self.full_path_source)

        n_mfcc = 10

        technique = MFCC(flatten=False, contents_dirs=ds_path_without_field_name, n_mfcc=n_mfcc)
        framework_output = technique.produce_content('audioPath', [], source)

        # check that there are 3 outputs and all of them match the expected number of dimensions (corresponding to the n_mfcc)
        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(out.value.shape[1] == n_mfcc for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        mfcc_output = []
        for audio, sample_rate in zip(self.audios_list, self.sample_rates):
            mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
            mfcc_output.append(np.transpose(mfcc(audio.float()).numpy()[0]))

        for single_f_out, single_h_out in zip(framework_output, mfcc_output):
            np.testing.assert_array_equal(single_f_out.value, single_h_out)

        # same test as the one done for produce_content but on a single audio, the single audio should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(self.audios_list[0], self.sample_rates[0])
        self.assertEqual(2, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true
        technique = MFCC(flatten=True, contents_dirs=ds_path_without_field_name, n_mfcc=n_mfcc)
        framework_output = technique.produce_content('audioPath', [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))
