import os
import shutil
import json
from unittest import TestCase

import numpy as np
import timm
from skimage.feature import canny, hog
from sklearn.cluster import KMeans

from clayrs.content_analyzer import JSONFile, EmbeddingField
from clayrs.content_analyzer.field_content_production_techniques.visual_techniques.visual_content_techniques import \
    ClasslessImageFolder, PytorchImageModels, SkImageHogDescriptor, SkImageCannyEdgeDetector, SkImageMainColors, \
    SkImageColorQuantization

import torch
import PIL.Image
import torchvision.transforms.functional as TF

from test import dir_test_files

raw_source_path_online = os.path.join(dir_test_files, 'test_images', 'tradesy_small_online.json')
raw_source_path_local = os.path.join(dir_test_files, 'test_images', 'tradesy_small_local.json')
ds_path = os.path.join(dir_test_files, 'test_images', 'images')

this_file_path = os.path.dirname(os.path.abspath(__file__))


class TestVisualPostProcessing(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        source = JSONFile(raw_source_path_local)
        new_source_path = os.path.join(dir_test_files, 'tmp_source.json')

        # recreate the source to correctly init paths (with os.path.join)
        data_with_new_paths = []
        for data in source:
            data_with_new_path = data
            data_with_new_path['imagePath'] = os.path.join(dir_test_files, 'test_images',
                                                           data_with_new_path['imagePath'])
            data_with_new_paths.append(data_with_new_path)

        with open(new_source_path, 'w') as f:
            json.dump(data_with_new_paths, f)

        cls.new_source_path = new_source_path

        new_source = JSONFile(new_source_path)

        images_list = []
        for data in new_source:
            image_path = data['imagePath']
            image = PIL.Image.open(image_path).convert('RGB')
            image = TF.resize(TF.to_tensor(image), [227, 227])
            image = image.unsqueeze(0)
            images_list.append(image)

        cls.images_list = torch.vstack(images_list)

    def test_dataset(self):

        ds = ClasslessImageFolder(ds_path, (100, 100))

        self.assertEqual(5, len(ds))

        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]

        self.assertTrue(all(image.shape == (3, 100, 100) for image in images_list))

        images_list = [os.path.join(ds_path, file_name) for file_name in os.listdir(ds_path)]
        images_list.append(os.path.join(ds_path, 'not_existing_image_path'))

        ds = ClasslessImageFolder(ds_path, (100, 100), images_list)

        self.assertEqual(6, len(ds))

        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]

        self.assertTrue(all(image.shape == (3, 100, 100) for image in images_list))
        self.assertTrue(torch.all(images_list[5] == 0))

    def test_downloading_images(self):

        source = JSONFile(raw_source_path_online)

        dl = SkImageHogDescriptor(batch_size=1).get_data_loader('imageUrl', source)

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs', 'imageUrl')))
        self.assertEqual(5, len([image for image in dl]))

        shutil.rmtree(os.path.join(this_file_path, 'imgs_dirs'))

    def test_loading_images(self):

        resize_size = (100, 100)

        source_with_new_paths = JSONFile(self.new_source_path)

        dl = SkImageHogDescriptor(resize_size=resize_size, batch_size=1).get_data_loader('imagePath',
                                                                                         source_with_new_paths)
        images_in_dl = [image for image in dl]

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs', 'imagePath')))
        self.assertTrue(all(image.shape == (1, 3, 100, 100) for image in images_in_dl))
        self.assertEqual(5, len(images_in_dl))

        test_image_path = os.path.join(dir_test_files, 'test_images', 'images',
                                       'anthropologie-skirt-light-pink-434-1.jpg')
        test_image = PIL.Image.open(test_image_path).convert('RGB')
        test_image = TF.resize(TF.to_tensor(test_image), list(resize_size))
        test_image = test_image.unsqueeze(0)

        self.assertTrue(any(torch.equal(test_image, image_in_dl) for image_in_dl in images_in_dl))

        shutil.rmtree(os.path.join(this_file_path, 'imgs_dirs'))
        os.remove(self.new_source_path)

    def test_pytorch_image_models(self):

        source = JSONFile(self.new_source_path)

        technique = PytorchImageModels('resnet34', imgs_dirs=ds_path, flatten=False)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 3 for out in framework_output))

        timm_model = timm.create_model('resnet34', pretrained=True, features_only=True).eval()
        timm_output = timm_model(self.images_list)[-1]

        for single_f_out, single_t_out in zip(framework_output, timm_output):

            np.testing.assert_array_equal(single_f_out.value, single_t_out.squeeze().detach().numpy())

        single_out = technique.produce_batch_repr(self.images_list[0])
        self.assertEqual(1, len(single_out))
        self.assertEqual(3, len(single_out[0].value.shape))

        technique = PytorchImageModels('resnet34', imgs_dirs=ds_path, flatten=True)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

    def test_hog_descriptor(self):

        source = JSONFile(self.new_source_path)

        technique = SkImageHogDescriptor(flatten=True)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        hog_output = []
        for image in self.images_list:
            hog_output.append(hog(image.numpy(), channel_axis=0))

        for single_f_out, single_h_out in zip(framework_output, hog_output):

            np.testing.assert_array_equal(single_f_out.value, single_h_out)

        technique = SkImageHogDescriptor(flatten=False)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertTrue(all(len(out.value.shape) == 5 for out in framework_output))

        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(5, len(single_out.value.shape))

    def test_canny_edge_detector(self):

        source = JSONFile(self.new_source_path)

        technique = SkImageCannyEdgeDetector(flatten=False)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 5 for out in framework_output))

        canny_output = []
        for image in self.images_list:
            grey_image = TF.rgb_to_grayscale(image).squeeze().numpy()
            canny_output.append(canny(image=grey_image))

        for single_f_out, single_c_out in zip(framework_output, canny_output):

            np.testing.assert_array_equal(single_f_out.value, single_c_out)

        single_out = technique.produce_single_repr(TF.rgb_to_grayscale(self.images_list[0]))
        self.assertEqual(2, len(single_out.value.shape))

        technique = SkImageCannyEdgeDetector(flatten=True)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

    def test_main_colors(self):

        source = JSONFile(self.new_source_path)

        technique = SkImageMainColors()
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))

        color_output = []
        for image in self.images_list:
            image = image.numpy() * 255
            colors = np.array([image[0, :, :].flatten(),
                               image[1, :, :].flatten(),
                               image[2, :, :].flatten()])
            color_output.append(colors.astype(int))

        for single_f_out, single_c_out in zip(framework_output, color_output):

            np.testing.assert_array_equal(single_f_out.value, single_c_out)

        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(2, len(single_out.value.shape))

        single_out = technique.produce_single_repr(TF.rgb_to_grayscale(self.images_list[0]))
        self.assertEqual(2, len(single_out.value.shape))

    def test_color_quantization(self):

        source = JSONFile(self.new_source_path)

        technique = SkImageColorQuantization(flatten=False, random_state=42)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))
        self.assertTrue(all(out.value.shape == (3, 3) for out in framework_output))

        quantization_output = []
        k_means = KMeans(3, random_state=42)

        for image in self.images_list:
            reshaped_image = (image.numpy() * 255).reshape(image.shape[1] * image.shape[2], 3)
            k_means.fit(reshaped_image)
            quantization_output.append(k_means.cluster_centers_.astype(int))

        for single_f_out, single_q_out in zip(framework_output, quantization_output):

            np.testing.assert_array_equal(single_f_out.value, single_q_out)

        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(2, len(single_out.value.shape))

        technique = SkImageColorQuantization(flatten=True)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))


