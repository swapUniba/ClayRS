import unittest

import numpy as np
import os
import shutil
from unittest import TestCase

import torchvision
import torchvision.transforms.functional as TF
from timm.models import create_feature_extractor

from clayrs.content_analyzer import JSONFile, EmbeddingField, TorchResize, TorchLambda, ClipSampler, \
    TorchVisionVideoModels, PytorchVideoModels
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.visual_techniques import \
    PytorchImageModels, SkImageHogDescriptor
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.visual_techniques.visual_content_techniques import \
    ClasslessImageFromVideoFolder

import torch

from test import dir_test_files
from test.content_analyzer.field_content_production_techniques.utils import create_full_path_source, videos_path, create_video_dataset

raw_source_path_online = os.path.join(dir_test_files, 'test_videos', 'urls_dataset.json')
raw_source_path_local_rel = os.path.join(dir_test_files, 'test_videos', 'paths_dataset.json')
ds_path = os.path.join(dir_test_files, 'test_videos', 'videos', 'videoPath')
ds_path_without_field_name = os.path.join(dir_test_files, 'test_videos', 'videos')

this_file_path = os.path.dirname(os.path.abspath(__file__))


def get_videos_list_processed(source, resize_size):
    videos_list = []
    for data in source:
        video_path = data['videoPath']
        reader = torchvision.io.VideoReader(video_path, "video")
        video = torch.stack([frame['data'] for frame in reader])
        video = TF.resize(video, resize_size)
        videos_list.append(video)

    return videos_list


class TestClasslessImageFromVideoFolder(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_video_dataset()

    def test_dataset_videos(self):
        videos_paths_list = [os.path.join(videos_path, file_name) for file_name in os.listdir(videos_path)]

        # load the dataset and check that the 3 videos in it are correctly loaded
        ds = ClasslessImageFromVideoFolder(videos_paths_list, (0, None))
        self.assertEqual(3, len(ds))

        # load the dataset and check that the 3 videos in it are correctly loaded and the frames are
        # extracted and resized
        ds = ClasslessImageFromVideoFolder(videos_paths_list, (0, None), preprocessor_list=[TorchResize((227, 227))])
        self.assertEqual(3, len(ds))
        # check that all frames have been resized
        videos_list = [ds[video_idx] for video_idx in range(0, len(ds))]
        self.assertTrue(all(frame.shape == (3, 227, 227) for video in videos_list for frame in video))

        # test the dataset again but this time there will be a non-existing path,
        # the dataset it will replace the desired video with a tensor full of zeros
        videos_paths_list.append(os.path.join(ds_path, 'not_existing_video_path'))

        ds = ClasslessImageFromVideoFolder(videos_paths_list, (0, None))
        videos_list = [ds[video_idx] for video_idx in range(0, len(ds))]
        self.assertEqual(4, len(ds))

        # ds.count keeps track of the number of paths that couldn't be resolved
        self.assertEqual(1, ds.count)

        # check that the last video that has been added only contains zeros
        # (the one associated to the non-existing path)
        self.assertTrue(torch.all(videos_list[3] == 0))


class TestVideoProcessingTechniquesVisual(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_video_dataset()
        cls.full_path_source = create_full_path_source(raw_source_path_local_rel, 'videoPath', videos_path)

    def test_video_high_level(self):

        # since videos are handled using visual techniques, it is just checked that a single high level technique
        # works with videos (to check the correctness of the processing done in case of video)

        source = JSONFile(self.full_path_source)

        technique = PytorchImageModels('resnet18', feature_layer=-3, contents_dirs=ds_path_without_field_name,
                                       flatten=False, batch_size=3)
        framework_output = technique.produce_content('videoPath',
                                                     [TorchResize((227, 227)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_output))

        shutil.rmtree(ds_path_without_field_name)

    def test_video_low_level(self):

        # since videos are handled using visual techniques, it is just checked that a single low level technique
        # works with videos (to check the correctness of the processing done in case of video)

        source = JSONFile(self.full_path_source)

        technique = SkImageHogDescriptor(flatten=False, contents_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('videoPath', [TorchResize((100, 100))], source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))

        shutil.rmtree(ds_path_without_field_name)

    def test_high_level_frame_technique(self):

        # test expected output when using a frame based technique on a video file without and with sampler as well

        source = JSONFile(self.full_path_source)

        technique = PytorchImageModels('resnet18', feature_layer=-3, contents_dirs=ds_path_without_field_name,
                                       flatten=False)

        framework_output = technique.produce_content('videoPath',
                                                     [TorchResize((256, 256)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_output))

        technique = PytorchImageModels('resnet18', feature_layer=-3, contents_dirs=ds_path_without_field_name,
                                       flatten=False)

        framework_output = technique.produce_content('videoPath',
                                                     [ClipSampler(2, 10),
                                                      TorchResize((256, 256)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_output))
        # 2 * 10
        self.assertTrue(all(out.value.shape[0] == 20 for out in framework_output))

        shutil.rmtree(ds_path_without_field_name)

    def test_high_level_clip_technique(self):

        # check that there are 3 outputs and all of them match the expected number of dimensions
        source = JSONFile(self.full_path_source)

        technique = TorchVisionVideoModels(model_name="r2plus1d_18", feature_layer=-2, device="cpu", batch_size=2,
                                           contents_dirs=ds_path_without_field_name, )
        framework_output = technique.produce_content('videoPath',
                                                     [TorchResize((112, 112)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        # not flattened
        self.assertTrue(all(len(out.value.shape) != 2 for out in framework_output))

        technique = TorchVisionVideoModels(model_name="r2plus1d_18", feature_layer=-2, device="cpu", batch_size=2,
                                           contents_dirs=ds_path_without_field_name, )
        framework_output = technique.produce_content('videoPath',
                                                     [ClipSampler(2, 10),
                                                      TorchResize((112, 112)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        # not flattened
        self.assertTrue(all(len(out.value.shape) != 2 for out in framework_output))
        # clip models process the 2 frames together, so only 10 output dimensionality
        self.assertTrue(all(out.value.shape[0] == 10 for out in framework_output))

        shutil.rmtree(ds_path_without_field_name)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.full_path_source)


class TestHighLevelClipVisualTechnique(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_video_dataset()
        full_path_source = create_full_path_source(raw_source_path_local_rel, 'videoPath', videos_path)
        cls.full_path_source = full_path_source

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ds_path_without_field_name)
        os.remove(cls.full_path_source)


class TestTorchVisionVideoModels(TestHighLevelClipVisualTechnique):

    def test_torchvision_video_models(self):
        # check that there are 3 outputs and all of them match the expected number of dimensions
        source = JSONFile(self.full_path_source)
        videos_list = get_videos_list_processed(source, (112, 112))

        # batch size set to 1 otherwise dl pads automatically
        technique = TorchVisionVideoModels(model_name="r2plus1d_18", contents_dirs=ds_path_without_field_name,
                                           feature_layer=-2, device="cpu", batch_size=1)
        framework_output = technique.produce_content('videoPath',
                                                     [TorchResize((112, 112)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 5 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        original_model = torchvision.models.video.r2plus1d_18(weights="DEFAULT").eval().to("cpu")
        feature_layer = list(dict(original_model.named_modules()).keys())[-2]
        original_model = create_feature_extractor(original_model, {feature_layer: "feature_layer"}).eval()

        with torch.no_grad():
            for i, video in enumerate(videos_list):
                original_output = \
                original_model(torch.moveaxis(video.div(255.0).unsqueeze(0), 1, 2))["feature_layer"].cpu()[0].numpy()
                np.testing.assert_array_equal(framework_output[i].value[0], original_output)

        # check that output is flattened when "flatten" parameter is set to true
        technique = TorchVisionVideoModels(model_name="r2plus1d_18", contents_dirs=ds_path_without_field_name,
                                           feature_layer=-2, device="cpu", batch_size=3, flatten=True)
        framework_output = technique.produce_content('videoPath',
                                                     [TorchResize((112, 112)), TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))


class TestPytorchVideoModels(TestHighLevelClipVisualTechnique):

    def test_pytorch_video_models(self):
        # check that there are 3 outputs and all of them match the expected number of dimensions
        source = JSONFile(self.full_path_source)
        videos_list = get_videos_list_processed(source, (182, 182))

        technique = PytorchVideoModels(model_name="x3d_s", contents_dirs=ds_path_without_field_name,
                                       device="cpu", batch_size=2, feature_layer=-2)
        framework_output = technique.produce_content('videoPath',
                                                     [TorchResize((182, 182)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 5 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        original_model = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_s', pretrained=True)
        feature_layer = list(dict(original_model.named_modules()).keys())[-2]
        original_model = create_feature_extractor(original_model, {feature_layer: "feature_layer"}).to('cpu').eval()

        with torch.no_grad():
            for i, video in enumerate(videos_list):
                original_output = \
                original_model(torch.moveaxis(video.div(255.0).unsqueeze(0), 1, 2))["feature_layer"].cpu()[0].numpy()
                np.testing.assert_array_equal(framework_output[i].value[0], original_output)

        # check that output is flattened when "flatten" parameter is set to true
        technique = PytorchVideoModels(model_name="x3d_s", contents_dirs=ds_path_without_field_name,
                                       device="cpu", batch_size=1, flatten=True)
        framework_output = technique.produce_content('videoPath',
                                                     [TorchResize((182, 182)), TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check the output when using a ClipSampler
        technique = PytorchVideoModels(model_name="x3d_s", contents_dirs=ds_path_without_field_name,
                                       device="cpu", batch_size=1)
        framework_output = technique.produce_content('videoPath',
                                                     [ClipSampler(13, 10),
                                                      TorchResize((182, 182)),
                                                      TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 5 for out in framework_output))
