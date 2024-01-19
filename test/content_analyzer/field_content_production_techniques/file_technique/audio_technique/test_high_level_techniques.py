import os
import shutil
import unittest

import numpy as np
import torch
import torchaudio

from clayrs.content_analyzer import JSONFile, EmbeddingField
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.audio_techniques.high_level_techniques import \
    TorchAudioPretrained
from clayrs.content_analyzer.information_processor.preprocessors.audio_preprocessors import TorchResample
from test.content_analyzer.field_content_production_techniques.utils import create_audio_dataset
from test.content_analyzer.field_content_production_techniques.file_technique.audio_technique.test_audio_content_technique import \
    create_full_path_source, ds_path_without_field_name, raw_source_path_local_rel, audios_path

this_file_path = os.path.dirname(os.path.abspath(__file__))


class TestHighLevelAudioTechnique(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_audio_dataset()
        full_path_source = create_full_path_source(raw_source_path_local_rel, 'audioPath', audios_path)

        # pre-build the list of audios from the source, this is done for
        # comparing outputs obtained from the techniques
        new_source = JSONFile(full_path_source)

        audios_list = []
        audios_list_resampled = []

        for data in new_source:
            audio_path = data['audioPath']
            audio, original_sample_rate = torchaudio.load(audio_path)
            resampled_audio = torchaudio.functional.resample(audio, original_sample_rate, 16000)
            audios_list.append(audio)
            audios_list_resampled.append(resampled_audio)

        cls.audios_list = audios_list
        cls.audios_list_padded = torch.nn.utils.rnn.pad_sequence([x.t() for x in audios_list],
                                                                 batch_first=True,
                                                                 padding_value=0.).permute(0, 2, 1)

        cls.audios_list_resampled = audios_list_resampled
        cls.audios_list_padded_resampled = torch.nn.utils.rnn.pad_sequence([x.t() for x in audios_list_resampled],
                                                                           batch_first=True,
                                                                           padding_value=0.).permute(0, 2, 1)
        cls.full_path_source = full_path_source

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ds_path_without_field_name)
        os.remove(cls.full_path_source)


class TestTorchAudioPretrained(TestHighLevelAudioTechnique):

    def test_torchaudio_pretrained(self):

        # test output of TorchAudioPretrained technique

        source = JSONFile(self.full_path_source)

        technique = TorchAudioPretrained(contents_dirs=ds_path_without_field_name, flatten=False)
        framework_output = technique.produce_content('audioPath', [TorchResample(16000)], source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        output = model.extract_features(self.audios_list_padded_resampled.squeeze())[0][-1]

        for single_f_out, single_t_out in zip(framework_output, output):
            np.testing.assert_array_equal(single_f_out.value, single_t_out.squeeze().detach().numpy())

        # same test as the one done for produce_content but on a single audio, the single audio should
        # match the expected number of dimensions in output
        single_out = technique.produce_batch_repr(self.audios_list_resampled[0], [16000])
        self.assertEqual(1, len(single_out))
        self.assertEqual(2, len(single_out[0].shape))

        def remove_1(x):
            return x - 1

        technique_with_function = TorchAudioPretrained(apply_on_output=remove_1,
                                                       contents_dirs=ds_path_without_field_name,
                                                       flatten=False)
        framework_output_with_function = technique_with_function.produce_content('audioPath',
                                                                                 [TorchResample(16000)], source)
        self.assertEqual(3, len(framework_output_with_function))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output_with_function))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output_with_function))

        for single_f_out, single_f_with_func_out in zip(framework_output, framework_output_with_function):
            np.testing.assert_array_equal(single_f_out.value - 1, single_f_with_func_out.value)

        # check that output is flattened when "flatten" parameter is set to true
        technique = TorchAudioPretrained(contents_dirs=ds_path_without_field_name, flatten=True)
        framework_output = technique.produce_content('audioPath', [TorchResample(16000)], source)

        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))
