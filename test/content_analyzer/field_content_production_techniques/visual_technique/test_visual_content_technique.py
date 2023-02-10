import os
import shutil
import json
from unittest import TestCase

import numpy as np
import timm
from scipy.ndimage.filters import convolve
from skimage.feature import canny, hog, SIFT, local_binary_pattern
from sklearn.cluster import KMeans

from clayrs.content_analyzer import JSONFile, EmbeddingField, TorchGrayscale
from clayrs.content_analyzer.field_content_production_techniques.visual_techniques.visual_content_techniques import \
    ClasslessImageFolder, PytorchImageModels, SkImageHogDescriptor, SkImageCannyEdgeDetector, ColorsHist, \
    ColorQuantization, SkImageSIFT, SkImageLBP, CustomFilterConvolution

import torch
import PIL.Image
import torchvision.transforms.functional as TF

from test import dir_test_files

raw_source_path_online = os.path.join(dir_test_files, 'test_images', 'tradesy_small_online.json')
raw_source_path_local = os.path.join(dir_test_files, 'test_images', 'tradesy_small_local.json')
ds_path = os.path.join(dir_test_files, 'test_images', 'images', 'imagePath')
ds_path_without_field_name = os.path.join(dir_test_files, 'test_images', 'images')

this_file_path = os.path.dirname(os.path.abspath(__file__))


class TestVisualProcessingTechniques(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        source = JSONFile(raw_source_path_local)
        new_source_path = os.path.join(dir_test_files, 'tmp_source.json')

        # recreate the source to correctly init paths (with os.path.join)
        data_with_new_paths = []
        for data in source:
            data_with_new_path = data
            data_with_new_path['imagePath'] = os.path.join(ds_path, data_with_new_path['imagePath'])
            data_with_new_paths.append(data_with_new_path)

        with open(new_source_path, 'w') as f:
            json.dump(data_with_new_paths, f)

        cls.new_source_path = new_source_path

        new_source = JSONFile(new_source_path)

        # pre-build the list of images from the source, this is done for
        # comparing outputs obtained from the techniques
        images_list = []
        for data in new_source:
            image_path = data['imagePath']
            image = PIL.Image.open(image_path).convert('RGB')
            image = TF.resize(TF.to_tensor(image), [227, 227])
            image = image.unsqueeze(0)
            images_list.append(image)

        cls.images_list = torch.vstack(images_list)

    def test_dataset(self):

        # load the dataset and check that the 5 images in it are correctly loaded

        ds = ClasslessImageFolder(ds_path, (100, 100))

        self.assertEqual(5, len(ds))

        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]

        # dataset automatically resizes images to the specified size, check that all of them have been resized

        self.assertTrue(all(image.shape == (3, 100, 100) for image in images_list))

        # test the dataset again but this time specifying the images to consider through the images_list parameter
        # by doing so, the dataset will check for all of them and if it happens to find one path that doesn't
        # actually exist, it will replace the desired image with a tensor full of zeros
        images_paths_list = [os.path.join(ds_path, file_name) for file_name in os.listdir(ds_path)]
        images_paths_list.append(os.path.join(ds_path, 'not_existing_image_path'))

        ds = ClasslessImageFolder(ds_path, (100, 100), images_paths_list)

        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]

        self.assertEqual(6, len(ds))
        # ds.count keeps track of the number of paths that couldn't be resolved
        self.assertEqual(1, ds.count)

        self.assertTrue(all(image.shape == (3, 100, 100) for image in images_list))
        # check that the last image that has been added only contains zeros
        # (the one associated to the non-existing path)
        self.assertTrue(torch.all(images_list[5] == 0))

    def test_downloading_images(self):

        # use the json file containing online links to locally download the images

        source = JSONFile(raw_source_path_online)

        dl = SkImageHogDescriptor(batch_size=1).get_data_loader('imageUrl', source)

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs', 'imageUrl')))
        self.assertEqual(5, len([image for image in dl]))

        shutil.rmtree(os.path.join(this_file_path, 'imgs_dirs'))

    def test_loading_images(self):

        # use the json file containing local paths to load the images

        resize_size = (100, 100)

        source_with_new_paths = JSONFile(self.new_source_path)

        dl = SkImageHogDescriptor(resize_size=resize_size, batch_size=1).get_data_loader('imagePath',
                                                                                         source_with_new_paths)
        images_in_dl = [image for image in dl]

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs', 'imagePath')))
        self.assertTrue(not all(torch.all(image == 0) for image in images_in_dl))
        self.assertTrue(all(image.shape == (1, 3, 100, 100) for image in images_in_dl))
        self.assertEqual(5, len(images_in_dl))

        # check that at least one of the loaded images matches one of the images from the dataset

        test_image_path = os.path.join(ds_path, 'anthropologie-skirt-light-pink-434-1.jpg')
        test_image = PIL.Image.open(test_image_path).convert('RGB')
        test_image = TF.resize(TF.to_tensor(test_image), list(resize_size))
        test_image = test_image.unsqueeze(0)

        self.assertTrue(any(torch.equal(test_image, image_in_dl) for image_in_dl in images_in_dl))

        shutil.rmtree(os.path.join(this_file_path, 'imgs_dirs'))

    def test_pytorch_image_models(self):

        # test output of PytorchImageModels technique

        source = JSONFile(self.new_source_path)

        technique = PytorchImageModels('resnet18', imgs_dirs=ds_path_without_field_name, flatten=False)
        framework_output = technique.produce_content('imagePath', [], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 3 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        timm_model = timm.create_model('resnet18', pretrained=True, features_only=True).eval()
        timm_output = timm_model(self.images_list)[-1]

        for single_f_out, single_t_out in zip(framework_output, timm_output):

            np.testing.assert_array_equal(single_f_out.value, single_t_out.squeeze().detach().numpy())

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_batch_repr(self.images_list[0].unsqueeze(0))
        self.assertEqual(1, len(single_out))
        self.assertEqual(3, len(single_out[0].value.shape))

        # check that output is flattened when "flatten" parameter is set to true
        technique = PytorchImageModels('resnet18', imgs_dirs=ds_path_without_field_name, flatten=True)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

    def test_hog_descriptor(self):

        # test output of SkImageHogDescriptor technique

        source = JSONFile(self.new_source_path)

        # hog descriptor from skimage provides the argument "feature_vector" to output a 1 dimensional
        # representation of the descriptors, since the parameter is true by default, we first test
        # for the "flatten" set to true case
        technique = SkImageHogDescriptor(flatten=True, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        hog_output = []
        for image in self.images_list:
            hog_output.append(hog(image.numpy(), channel_axis=0))

        for single_f_out, single_h_out in zip(framework_output, hog_output):

            np.testing.assert_array_equal(single_f_out.value, single_h_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

        # check that output is NOT flattened when "flatten" parameter is set to false
        technique = SkImageHogDescriptor(flatten=False, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertTrue(all(len(out.value.shape) == 5 for out in framework_output))

    def test_canny_edge_detector(self):

        # test output of SkImageCannyEdgeDetector technique

        source = JSONFile(self.new_source_path)

        # NOTE: canny edge detector works on grayscale images
        technique = SkImageCannyEdgeDetector(flatten=False, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        canny_output = []
        for image in self.images_list:
            grey_image = TF.rgb_to_grayscale(image).squeeze().numpy()
            canny_output.append(canny(image=grey_image))

        for single_f_out, single_c_out in zip(framework_output, canny_output):

            np.testing.assert_array_equal(single_f_out.value, single_c_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(TF.rgb_to_grayscale(self.images_list[0]))
        self.assertEqual(2, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true
        technique = SkImageCannyEdgeDetector(flatten=True, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that the technique automatically converts the image to grayscale if an RGB one is passed
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

    def test_colors_histogram(self):

        # test output of Color Histogram technique

        source = JSONFile(self.new_source_path)

        technique = ColorsHist(imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        color_output = []
        for image in self.images_list:
            image = image.numpy() * 255
            colors = np.array([image[0, :, :].flatten(),
                               image[1, :, :].flatten(),
                               image[2, :, :].flatten()])
            color_output.append(colors.astype(int))

        for single_f_out, single_c_out in zip(framework_output, color_output):

            np.testing.assert_array_equal(single_f_out.value, single_c_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(2, len(single_out.value.shape))

        # since color histogram works on RGB images (or 3 channel images anyway), it is expected that, by passing
        # a grayscale image, the technique will automatically convert it to RGB to match the requirements
        single_out = technique.produce_single_repr(TF.rgb_to_grayscale(self.images_list[0]))
        self.assertEqual(2, len(single_out.value.shape))

    def test_color_quantization(self):

        # test output of Color Quantization technique

        source = JSONFile(self.new_source_path)

        technique = ColorQuantization(n_colors= 3, flatten=False, random_state=42, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        # in this case the shape is also explicitly checked (since 3 colors are considered it is expected that each
        # output will be similar to [20, 134, 25] where each value represents one of the image channels;
        # so the full output should be similar to
        # [[20, 134, 25],
        # [56, 72, 3],
        # [9, 25, 39]])
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))
        self.assertTrue(all(out.value.shape == (3, 3) for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        quantization_output = []
        k_means = KMeans(3, random_state=42)

        for image in self.images_list:
            reshaped_image = (image.numpy() * 255).reshape(image.shape[1] * image.shape[2], 3)
            k_means.fit(reshaped_image)
            quantization_output.append(k_means.cluster_centers_.astype(int))

        for single_f_out, single_q_out in zip(framework_output, quantization_output):

            np.testing.assert_array_equal(single_f_out.value, single_q_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(2, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true
        technique = ColorQuantization(flatten=True, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # since color quantization works on RGB images (or 3 channel images anyway), it is expected that, by passing
        # a grayscale image, the technique will automatically convert it to RGB to match the requirements
        single_out = technique.produce_single_repr(TF.rgb_to_grayscale(self.images_list[0]))
        self.assertEqual(1, len(single_out.value.shape))

    def test_sift(self):

        # test output of SkImageSIFT technique

        source = JSONFile(self.new_source_path)

        # NOTE: sift works on grayscale images
        technique = SkImageSIFT(flatten=False, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        sift_output = []
        sift = SIFT()

        for image in self.images_list:
            grey_image = TF.rgb_to_grayscale(image).squeeze().numpy()
            sift.detect_and_extract(grey_image)
            sift_output.append(sift.descriptors)

        for single_f_out, single_s_out in zip(framework_output, sift_output):

            np.testing.assert_array_equal(single_f_out.value, single_s_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(2, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true
        technique = SkImageSIFT(flatten=True, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that the technique automatically converts the image to grayscale if an RGB one is passed
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

    def test_LBP(self):

        # test output of SkImageLBP technique

        # NOTE: for the LBP technique, a boolean parameter "as_image" can be specified
        # if as_image is set to false, the output will be a vector containing the number of occurrences of each pattern
        # if as_image is set to true, the output will be the lbp image
        # both cases are tested

        source = JSONFile(self.new_source_path)

        # as_image set to True
        # NOTE: lbp technique works on grayscale images
        technique = SkImageLBP(p=8, r=2, as_image=True, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        lbp_output = []

        for image in self.images_list:
            grey_image = TF.rgb_to_grayscale(image).squeeze().numpy()
            lbp_output.append(local_binary_pattern(grey_image, P=8, R=2))

        for single_f_out, single_l_out in zip(framework_output, lbp_output):

            np.testing.assert_array_equal(single_f_out.value, single_l_out)

        # as_image set to False
        technique = SkImageLBP(p=8, r=2, as_image=False, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        lbp_output = []

        for image in self.images_list:
            grey_image = TF.rgb_to_grayscale(image).squeeze().numpy()
            lbp_image = local_binary_pattern(grey_image, P=8, R=2)
            _, counts = np.unique(lbp_image, return_counts=True)
            lbp_output.append(counts)

        for single_f_out, single_l_out in zip(framework_output, lbp_output):

            np.testing.assert_array_equal(single_f_out.value, single_l_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true and "as_image" is set to True
        technique = SkImageLBP(as_image=True, flatten=True, p=8, r=2)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that the technique automatically converts the image to grayscale if an RGB one is passed
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true and "as_image" is set to False
        technique = SkImageLBP(as_image=False, flatten=True, p=8, r=2)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that the technique automatically converts the image to grayscale if an RGB one is passed
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

    def test_custom_convolution(self):

        # test output of Custom Convolution technique

        source = JSONFile(self.new_source_path)

        weights = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0]
        ])

        # NOTE: custom filter convolution technique works on grayscale images
        technique = CustomFilterConvolution(weights, flatten=False, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        # check that there are 5 outputs and all of them match the expected number of dimensions
        self.assertEqual(5, len(framework_output))
        self.assertTrue(all(isinstance(out, EmbeddingField) for out in framework_output))
        self.assertTrue(all(len(out.value.shape) == 2 for out in framework_output))

        # instantiate the original technique and compare its outputs with the ones obtained by the framework
        filter_output = []

        for image in self.images_list:
            grey_image = TF.rgb_to_grayscale(image).squeeze().numpy()
            convolution_result = convolve(grey_image, weights=weights)
            filter_output.append(convolution_result.astype(int))

        for single_f_out, single_s_out in zip(framework_output, filter_output):

            np.testing.assert_array_equal(single_f_out.value, single_s_out)

        # same test as the one done for produce_content but on a single image, the single image should
        # match the expected number of dimensions in output
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(2, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true
        technique = CustomFilterConvolution(weights, flatten=True, imgs_dirs=ds_path_without_field_name)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that the technique automatically converts the image to grayscale if an RGB one is passed
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.new_source_path)
