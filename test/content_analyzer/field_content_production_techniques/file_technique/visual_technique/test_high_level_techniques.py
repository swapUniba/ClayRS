import os
import shutil
import unittest

import PIL.Image
import cv2
import numpy as np
import requests
import timm
import torch
import torchvision.transforms.functional as TF
from timm.models import create_feature_extractor
from tqdm import tqdm

from clayrs.content_analyzer import JSONFile, PytorchImageModels, EmbeddingField, CaffeImageModels, TorchResize, \
    TorchLambda
from test.content_analyzer.field_content_production_techniques.utils import create_image_dataset
from test.content_analyzer.field_content_production_techniques.file_technique.visual_technique.test_visual_content_technique import \
    create_full_path_source, ds_path_without_field_name, raw_source_path_local_rel, images_path

this_file_path = os.path.dirname(os.path.abspath(__file__))


def get_images_list_processed(source, resize_size):
    images_list = []
    for data in source:
        image_path = data['imagePath']
        image = np.array(PIL.Image.open(image_path).convert('RGB'))
        image = TF.resize(torch.from_numpy(image).moveaxis(-1, 0), resize_size)
        image = image.unsqueeze(0)
        images_list.append(image)

    return torch.vstack(images_list)


class TestHighLevelFrameVisualTechnique(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_image_dataset()
        full_path_source = create_full_path_source(raw_source_path_local_rel, 'imagePath', images_path)
        cls.full_path_source = full_path_source

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ds_path_without_field_name)
        os.remove(cls.full_path_source)


class TestPytorchImageModels(TestHighLevelFrameVisualTechnique):

    @classmethod
    def setUpClass(cls) -> None:

        super().setUpClass()
        cls.weights_path = os.path.join(this_file_path, 'resnet101-63fe2227.pth')

        if not os.path.isfile(cls.weights_path):

            response = requests.get("https://download.pytorch.org/models/resnet101-63fe2227.pth")
            if response.status_code == 200:

                with open(cls.weights_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise ValueError("Couldn't download custom weights for tests")

    def test_pytorch_image_models(self):

        # check that there are 3 outputs and all of them match the expected number of dimensions
        source = JSONFile(self.full_path_source)
        images_list = get_images_list_processed(source, (256, 256))

        technique = PytorchImageModels('resnet18', feature_layer=-3, contents_dirs=ds_path_without_field_name,
                                       flatten=False)
        framework_output = technique.produce_content('imagePath', [TorchResize((256, 256)), TorchLambda(lambda x: x / 255.0)], source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        timm_model = timm.create_model('resnet18', pretrained=True)
        feature_layer = list(dict(timm_model.named_modules()).keys())[-3]
        timm_model = create_feature_extractor(timm_model, {feature_layer: "feature_layer"}).eval()
        with torch.no_grad():
            timm_output = timm_model(images_list.div(255.0))["feature_layer"].cpu().unsqueeze(1).numpy()

        for single_f_out, single_t_out in zip(framework_output, timm_output):
            np.testing.assert_array_equal(single_f_out.value, single_t_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_batch_repr(images_list[0].unsqueeze(0).div(255.0))
        self.assertEqual(1, len(single_out))
        self.assertEqual(3, len(single_out[0].shape))

        # check that the specified function is applied to the output of the technique

        def remove_1(x):
            return x - 1

        technique_with_function = PytorchImageModels('resnet18',
                                                     feature_layer=-3,
                                                     apply_on_output=remove_1,
                                                     contents_dirs=ds_path_without_field_name,
                                                     flatten=False)
        framework_output_with_function = technique_with_function.produce_content('imagePath', [TorchResize((256, 256)), TorchLambda(lambda x: x / 255.0)], source)
        self.assertEqual(3, len(framework_output_with_function))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output_with_function))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_output_with_function))

        for single_f_out, single_f_with_func_out in zip(framework_output, framework_output_with_function):
            np.testing.assert_array_equal(single_f_out.value - 1, single_f_with_func_out.value)

        # check that output is flattened when "flatten" parameter is set to true
        technique = PytorchImageModels('resnet18', contents_dirs=ds_path_without_field_name, flatten=True)
        framework_output = technique.produce_content('imagePath', [TorchResize((256, 256)), TorchLambda(lambda x: x / 255.0)], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that technique works correctly with custom weights
        technique = PytorchImageModels('resnet101', contents_dirs=ds_path_without_field_name, flatten=True,
                                       use_default_transforms=True,
                                       custom_weights_path=self.weights_path)
        framework_output = technique.produce_content('imagePath',
                                                     [TorchResize((256, 256)), TorchLambda(lambda x: x / 255.0)],
                                                     source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        custom_state_dict = torch.load(self.weights_path)

        for name, param in technique.model.named_parameters():
            self.assertTrue(torch.equal(param, custom_state_dict[name]))

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        os.remove(cls.weights_path)


class TestCaffeImageModels(TestHighLevelFrameVisualTechnique):

    @classmethod
    def setUpClass(cls) -> None:

        super().setUpClass()

        # download necessary files for class instantiation

        caffe_model_dir = os.path.join(this_file_path, "reference_caffenet")

        mean_pixel = os.path.join(caffe_model_dir, "ilsvrc_2012_mean.npy")
        caffe_model = os.path.join(caffe_model_dir, "bvlc_reference_caffenet.caffemodel")
        prototxt = os.path.join(caffe_model_dir, "deploy.prototxt")

        if not os.path.isdir(caffe_model_dir):

            os.makedirs(caffe_model_dir)

            resp = requests.get(
                r"https://github.com/facebookarchive/models/raw/master/bvlc_reference_caffenet/ilsvrc_2012_mean.npy",
                stream=True)

            with open(mean_pixel, 'wb') as file:

                for data in resp.iter_content():
                    file.write(data)

            resp = requests.get(r"http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel", stream=True)

            total = int(resp.headers.get('content-length', 0))
            with open(caffe_model, 'wb') as file, tqdm(desc="Downloading caffe model...",
                                                       total=total, unit='iB', unit_scale=True,
                                                       unit_divisor=1024) as prog_bar:

                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    prog_bar.update(size)

            resp = requests.get(r"https://raw.githubusercontent.com/BVLC/caffe/master/models/"
                                r"bvlc_reference_caffenet/deploy.prototxt", stream=True)

            with open(prototxt, 'wb') as file:

                for data in resp.iter_content():
                    file.write(data)

        cls.mean_pixel = mean_pixel
        cls.caffe_model = caffe_model
        cls.prototxt = prototxt

    def test_caffe_image_models(self):

        # check that there are 3 outputs and all of them match the expected number of dimensions

        source = JSONFile(self.full_path_source)
        images_list = get_images_list_processed(source, (227, 227))

        technique = CaffeImageModels(self.prototxt, self.caffe_model,
                                     feature_layer='conv4',
                                     mean_file_path=self.mean_pixel,
                                     batch_size=3,
                                     swap_rb=True,
                                     contents_dirs=ds_path_without_field_name,
                                     flatten=False)

        framework_output = technique.produce_content('imagePath', [TorchResize((227, 227))], source)

        self.assertEqual(3, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        model = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffe_model)
        mean = np.load(self.mean_pixel).mean(1).mean(1)

        imgs_blob = cv2.dnn.blobFromImages(np.moveaxis(images_list.numpy(), 1, -1), mean=mean, swapRB=True)
        model.setInput(imgs_blob)
        caffe_opencv_output = model.forward('conv4')

        for single_f_out, single_t_out in zip(framework_output, caffe_opencv_output):
            np.testing.assert_array_equal(single_f_out.value, np.expand_dims(single_t_out, 0))

        # same test as the one done for produce_content but on a single image
        single_out = technique.produce_batch_repr(images_list[0].unsqueeze(0))
        self.assertEqual(1, len(single_out))

        # test with swap_rb = False
        technique_no_swap = CaffeImageModels(self.prototxt, self.caffe_model,
                                             feature_layer='conv4',
                                             mean_file_path=self.mean_pixel,
                                             batch_size=3,
                                             swap_rb=False,
                                             contents_dirs=ds_path_without_field_name,
                                             flatten=False)

        framework_no_swap_output = technique_no_swap.produce_content('imagePath', [TorchResize((227, 227))], source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_no_swap_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_no_swap_output))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_no_swap_output))

        for single_f_out, single_f_out_no_swap in zip(framework_output, framework_no_swap_output):
            self.assertFalse(np.array_equal(single_f_out, single_f_out_no_swap))

        # test with no mean file path specified
        technique_no_mean = CaffeImageModels(self.prototxt, self.caffe_model,
                                             feature_layer='conv4',
                                             batch_size=3,
                                             swap_rb=True,
                                             contents_dirs=ds_path_without_field_name,
                                             flatten=False)

        framework_no_mean_output = technique_no_mean.produce_content('imagePath', [TorchResize((227, 227))], source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_no_mean_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_no_mean_output))
        self.assertTrue(all(len(out.value.shape) == 4 for out in framework_no_mean_output))

        for single_f_out, single_f_out_no_mean in zip(framework_output, framework_no_mean_output):
            self.assertFalse(np.array_equal(single_f_out, single_f_out_no_mean))

        # test without specifing layer from which to extract features
        technique_no_feature_layer = CaffeImageModels(self.prototxt, self.caffe_model,
                                                      mean_file_path=self.mean_pixel,
                                                      batch_size=3,
                                                      swap_rb=False,
                                                      contents_dirs=ds_path_without_field_name,
                                                      flatten=False)

        framework_no_feature_layer_output = technique_no_feature_layer.produce_content('imagePath',
                                                                                       [TorchResize((227, 227))],
                                                                                       source)

        # check that there are 3 outputs and all of them match the expected number of dimensions
        self.assertEqual(3, len(framework_no_feature_layer_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_no_feature_layer_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_no_feature_layer_output))

        for single_f_out, single_f_out_no_feature_layer in zip(framework_output, framework_no_feature_layer_output):
            self.assertFalse(np.array_equal(single_f_out, single_f_out_no_feature_layer))

        # check that output is flattened when "flatten" parameter is set to true
        technique = CaffeImageModels(self.prototxt, self.caffe_model,
                                     feature_layer='conv4',
                                     mean_file_path=self.mean_pixel,
                                     batch_size=3,
                                     swap_rb=True,
                                     contents_dirs=ds_path_without_field_name,
                                     flatten=True)
        framework_output = technique.produce_content('imagePath', [TorchResize((227, 227))], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        shutil.rmtree(os.path.join(this_file_path, "reference_caffenet"))
