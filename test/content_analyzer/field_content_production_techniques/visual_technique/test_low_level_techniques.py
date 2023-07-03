import os
import shutil
import unittest

import PIL.Image
import numpy as np
import torchvision.transforms.functional as TF
import torch
from scipy.ndimage import convolve
from skimage.feature import hog, canny, SIFT, local_binary_pattern
from sklearn.cluster import KMeans

from clayrs.content_analyzer import JSONFile, SkImageHogDescriptor, EmbeddingField, SkImageCannyEdgeDetector, \
    TorchGrayscale, ColorsHist, ColorQuantization, SkImageSIFT, SkImageLBP, CustomFilterConvolution
from test.content_analyzer.field_content_production_techniques.visual_technique.test_visual_content_technique import \
    create_full_path_source, ds_path_without_field_name


class TestLowLevelTechnique(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        full_path_source = create_full_path_source()

        # pre-build the list of images from the source, this is done for
        # comparing outputs obtained from the techniques
        new_source = JSONFile(full_path_source)

        images_list = []
        for data in new_source:
            image_path = data['imagePath']
            image = PIL.Image.open(image_path).convert('RGB')
            image = TF.resize(TF.to_tensor(image), [227, 227])
            image = image.unsqueeze(0)
            images_list.append(image)

        cls.images_list = torch.vstack(images_list)
        cls.full_path_source = full_path_source

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ds_path_without_field_name)
        os.remove(cls.full_path_source)


class TestHogDescriptor(TestLowLevelTechnique):

    def test_hog_descriptor(self):

        # test output of SkImageHogDescriptor technique

        source = JSONFile(self.full_path_source)

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


class TestCannyEdgeDetector(TestLowLevelTechnique):

    def test_canny_edge_detector(self):

        # test output of SkImageCannyEdgeDetector technique

        source = JSONFile(self.full_path_source)

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


class TestColorsHist(TestLowLevelTechnique):

    def test_colors_histogram(self):

        # test output of Color Histogram technique

        source = JSONFile(self.full_path_source)

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


class TestColorQuantization(TestLowLevelTechnique):

    def test_color_quantization(self):

        # test output of Color Quantization technique

        source = JSONFile(self.full_path_source)

        technique = ColorQuantization(n_colors=3, flatten=False, random_state=42, imgs_dirs=ds_path_without_field_name)
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


class TestSkImageSIFT(TestLowLevelTechnique):
    def test_sift(self):

        # test output of SkImageSIFT technique

        source = JSONFile(self.full_path_source)

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


class TestSkImageLBP(TestLowLevelTechnique):

    def test_LBP(self):

        # test output of SkImageLBP technique

        # NOTE: for the LBP technique, a boolean parameter "as_image" can be specified
        # if as_image is set to false, the output will be a vector containing the number of occurrences of each pattern
        # if as_image is set to true, the output will be the lbp image
        # both cases are tested

        source = JSONFile(self.full_path_source)

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
        technique = SkImageLBP(imgs_dirs=ds_path_without_field_name, as_image=True, flatten=True, p=8, r=2)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that the technique automatically converts the image to grayscale if an RGB one is passed
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))

        # check that output is flattened when "flatten" parameter is set to true and "as_image" is set to False
        technique = SkImageLBP(imgs_dirs=ds_path_without_field_name, as_image=False, flatten=True, p=8, r=2)
        framework_output = technique.produce_content('imagePath', [TorchGrayscale()], [], source)

        self.assertTrue(all(len(out.value.shape) == 1 for out in framework_output))

        # check that the technique automatically converts the image to grayscale if an RGB one is passed
        single_out = technique.produce_single_repr(self.images_list[0])
        self.assertEqual(1, len(single_out.value.shape))


class TestCustomFilterConvolution(TestLowLevelTechnique):
    def test_custom_convolution(self):

        # test output of Custom Convolution technique

        source = JSONFile(self.full_path_source)

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


if __name__ == "__main__":
    unittest.main()
