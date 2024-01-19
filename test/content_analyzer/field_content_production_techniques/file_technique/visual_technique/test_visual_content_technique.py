import os
import shutil
import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import requests

from clayrs.content_analyzer import JSONFile, SkImageHogDescriptor, TorchResize, PytorchImageModels
from clayrs.content_analyzer.field_content_production_techniques.file_techniques.visual_techniques.visual_content_techniques import \
    ClasslessImageFolder

import torch
import PIL.Image

from test import dir_test_files, dir_root_repo
from test.content_analyzer.field_content_production_techniques.utils import create_full_path_source, MockedGoodResponse, \
    MockedUnidentifiedResponse, create_image_dataset

raw_source_path_online = os.path.join(dir_test_files, 'test_images', 'urls_dataset.json')
raw_source_path_local_rel = os.path.join(dir_test_files, 'test_images', 'paths_dataset.json')
ds_path = os.path.join(dir_test_files, 'test_images', 'images', 'imagePath')
ds_path_without_field_name = os.path.join(dir_test_files, 'test_images', 'images')
images_path = os.path.join(dir_test_files, 'test_images', 'images_files')

this_file_path = os.path.dirname(os.path.abspath(__file__))


class TestClasslessImageFolder(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_image_dataset()

    def test_dataset(self):
        images_paths_list = [os.path.join(images_path, file_name) for file_name in os.listdir(images_path)]

        # load the dataset and check that the 3 images in it are correctly loaded
        ds = ClasslessImageFolder(images_paths_list)
        self.assertEqual(3, len(ds))

        # load the dataset and check that the 3 images in it are correctly loaded and resized
        ds = ClasslessImageFolder(images_paths_list, [TorchResize((100, 100))])
        self.assertEqual(3, len(ds))
        # check that all of them have been resized
        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]
        self.assertTrue(all(image.shape == (3, 100, 100) for image in images_list))

        # test the dataset again but this time there will be a non-existing path,
        # the dataset will replace the desired image with a tensor full of zeros
        images_paths_list.append(os.path.join(ds_path, 'not_existing_image_path'))

        ds = ClasslessImageFolder(images_paths_list)
        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]
        self.assertEqual(4, len(ds))

        # ds.error_count keeps track of the number of paths that couldn't be resolved
        self.assertEqual(1, ds.count)

        # check that the last image that has been added only contains zeros
        # (the one associated to the non-existing path)
        self.assertTrue(torch.all(images_list[3] == 0))


class TestVisualContentTechnique(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        create_image_dataset()
        cls.full_source_path = create_full_path_source(raw_source_path_local_rel, 'imagePath', images_path)

    @patch.object(requests, "get", return_value=MockedGoodResponse(images_path))
    def test_downloading_images_good_response(self, mocked_response):

        # use the json file containing online links to locally download the images
        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since VisualContentTechnique class is abstract
        dl, _ = SkImageHogDescriptor(contents_dirs=os.path.join(this_file_path, 'contents_dirs')).get_data_loader('imageUrl', source, [])

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'contents_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'contents_dirs', 'imageUrl')))
        self.assertEqual(3, len([image for image in dl]))

        for image in dl:
            self.assertFalse(torch.all(image == 0))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'contents_dirs'))

    @patch.object(requests, "get", return_value=MockedUnidentifiedResponse())
    def test_downloading_images_not_image_response(self, mocked_response):

        # use the json file containing online links to locally download the images

        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since VisualContentTechnique class is abstract
        dl, _ = SkImageHogDescriptor(contents_dirs=os.path.join(this_file_path, 'contents_dirs')).get_data_loader('imageUrl', source, [])

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'contents_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'contents_dirs', 'imageUrl')))
        self.assertEqual(3, len([image for image in dl]))

        for image in dl:
            self.assertTrue(torch.equal(torch.zeros(1, 3, 100, 100), image))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'contents_dirs'))

    @patch.object(requests, "get", side_effect=requests.exceptions.ConnectionError())
    def test_downloading_images_connection_error(self, mocked_response):

        # use the json file containing online links to locally download the images

        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since VisualContentTechnique class is abstract
        dl, _ = SkImageHogDescriptor(contents_dirs=os.path.join(this_file_path, 'contents_dirs')).get_data_loader('imageUrl', source, [])

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'contents_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'contents_dirs', 'imageUrl')))
        self.assertEqual(3, len([image for image in dl]))

        for image in dl:
            self.assertTrue(torch.equal(torch.zeros(1, 3, 100, 100), image))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'contents_dirs'))

    def test_loading_images_full_path(self):

        # use the json file containing local paths to load the images

        source_with_new_paths = JSONFile(self.full_source_path)

        dl, _ = SkImageHogDescriptor(contents_dirs=ds_path_without_field_name).get_data_loader('imagePath', source_with_new_paths, [])
        images_in_dl = [image for image in dl]

        self.assertTrue(not all(torch.all(image == 0) for image in images_in_dl))
        self.assertEqual(3, len(images_in_dl))

        # check that at least one of the loaded images matches one of the images from the dataset

        test_image_path = os.path.join(ds_path, os.listdir(ds_path)[0])
        test_image = np.array(PIL.Image.open(test_image_path).convert('RGB'))
        test_image = torch.from_numpy(test_image).moveaxis(-1, 0).unsqueeze(0)

        self.assertTrue(any(torch.equal(test_image, image_in_dl) for image_in_dl in images_in_dl))
        shutil.rmtree(ds_path_without_field_name)

    def test_loading_images_relative_path(self):

        # this test uses relative path so we should change the working dir
        current_workdir = os.getcwd()
        os.chdir(dir_root_repo)

        # use the json file containing local paths to load the images

        source_with_new_paths = JSONFile(raw_source_path_local_rel)

        dl, _ = SkImageHogDescriptor(contents_dirs=ds_path_without_field_name).get_data_loader('imagePath', source_with_new_paths, [])
        images_in_dl = [image for image in dl]

        self.assertTrue(not all(torch.all(image == 0) for image in images_in_dl))
        self.assertEqual(3, len(images_in_dl))

        # check that at least one of the loaded images matches one of the images from the dataset

        test_image_path = os.path.join(ds_path, os.listdir(ds_path)[0])
        test_image = np.array(PIL.Image.open(test_image_path).convert('RGB'))
        test_image = torch.from_numpy(test_image).moveaxis(-1, 0).unsqueeze(0)

        self.assertTrue(any(torch.equal(test_image, image_in_dl) for image_in_dl in images_in_dl))
        shutil.rmtree(ds_path_without_field_name)

        # reset the previous workdir
        os.chdir(current_workdir)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.full_source_path)


if __name__ == "__main__":
    unittest.main()
