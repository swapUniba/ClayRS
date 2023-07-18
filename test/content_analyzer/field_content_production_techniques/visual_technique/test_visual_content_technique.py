import io
import os
import shutil
import json
import unittest
from unittest import TestCase
from unittest.mock import patch

import requests

from clayrs.content_analyzer import JSONFile, SkImageHogDescriptor
from clayrs.content_analyzer.field_content_production_techniques.visual_techniques.visual_content_techniques import \
    ClasslessImageFolder

import torch
import PIL.Image
import torchvision.transforms.functional as TF

from test import dir_test_files, dir_root_repo

raw_source_path_online = os.path.join(dir_test_files, 'test_images', 'tradesy_small_online.json')
raw_source_path_local_rel = os.path.join(dir_test_files, 'test_images', 'tradesy_small_local_relative_paths.json')
ds_path = os.path.join(dir_test_files, 'test_images', 'images', 'imagePath')
ds_path_without_field_name = os.path.join(dir_test_files, 'test_images', 'images')
images_path = os.path.join(dir_test_files, 'test_images', 'images_files')

this_file_path = os.path.dirname(os.path.abspath(__file__))


class MockedGoodResponse:

    def __init__(self):
        img = PIL.Image.open(os.path.join(images_path, 'anthropologie-skirt-light-pink-434-1.jpg'))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        self.content = img_byte_arr.getvalue()


class MockedUnidentifiedImage:

    def __init__(self):
        self.content = str.encode('not an image')


def create_full_path_source() -> str:
    source = JSONFile(raw_source_path_local_rel)
    full_source_path = os.path.join(dir_test_files, 'full_paths_source.json')

    # recreate the source to correctly init paths (with os.path.join)
    data_with_full_paths = []
    for data in source:
        data_with_full_path = data
        image_name = data_with_full_path['imagePath'].split('/')[-1]
        data_with_full_path['imagePath'] = os.path.join(images_path, image_name)
        data_with_full_paths.append(data_with_full_path)

    with open(full_source_path, 'w') as f:
        json.dump(data_with_full_paths, f)

    return full_source_path


class TestClasslessImageFolder(TestCase):

    def test_dataset(self):
        images_paths_list = [os.path.join(images_path, file_name) for file_name in os.listdir(images_path)]

        # load the dataset and check that the 5 images in it are correctly loaded
        ds = ClasslessImageFolder(images_paths_list, (100, 100))
        self.assertEqual(5, len(ds))

        # dataset automatically resizes images to the specified size, check that all of them have been resized
        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]
        self.assertTrue(all(image.shape == (3, 100, 100) for image in images_list))

        # test the dataset again but this time there will be a non-existing path,
        # the dataset it will replace the desired image with a tensor full of zeros
        images_paths_list.append(os.path.join(ds_path, 'not_existing_image_path'))

        ds = ClasslessImageFolder(images_paths_list, (100, 100))
        images_list = [ds[image_idx] for image_idx in range(0, len(ds))]
        self.assertEqual(6, len(ds))

        # ds.error_count keeps track of the number of paths that couldn't be resolved
        self.assertEqual(1, ds.error_count)

        self.assertTrue(all(image.shape == (3, 100, 100) for image in images_list))

        # check that the last image that has been added only contains zeros
        # (the one associated to the non-existing path)
        self.assertTrue(torch.all(images_list[5] == 0))


class TestVisualContentTechnique(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.full_source_path = create_full_path_source()

    @patch.object(requests, "get", return_value=MockedGoodResponse())
    def test_downloading_images_good_response(self, mocked_response):

        # use the json file containing online links to locally download the images
        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since VisualContentTechnique class is abstract
        dl = SkImageHogDescriptor(imgs_dirs=os.path.join(this_file_path, 'imgs_dirs'),
                                  batch_size=1).get_data_loader('imageUrl', source)

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs', 'imageUrl')))
        self.assertEqual(5, len([image for image in dl]))

        for image in dl:
            self.assertFalse(torch.equal(torch.zeros((1, 3, 227, 227)), image))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'imgs_dirs'))

    @patch.object(requests, "get", return_value=MockedUnidentifiedImage())
    def test_downloading_images_not_image_response(self, mocked_response):

        # use the json file containing online links to locally download the images

        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since VisualContentTechnique class is abstract
        dl = SkImageHogDescriptor(imgs_dirs=os.path.join(this_file_path, 'imgs_dirs'),
                                  batch_size=1).get_data_loader('imageUrl', source)

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs', 'imageUrl')))
        self.assertEqual(5, len([image for image in dl]))

        for image in dl:
            self.assertTrue(torch.equal(torch.zeros(1, 3, 227, 227), image))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'imgs_dirs'))

    @patch.object(requests, "get", side_effect=requests.exceptions.ConnectionError())
    def test_downloading_images_connection_error(self, mocked_response):

        # use the json file containing online links to locally download the images

        source = JSONFile(raw_source_path_online)

        # instantiate a subclass since VisualContentTechnique class is abstract
        dl = SkImageHogDescriptor(imgs_dirs=os.path.join(this_file_path, 'imgs_dirs'),
                                  batch_size=1).get_data_loader('imageUrl', source)

        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs')))
        self.assertTrue(os.path.isdir(os.path.join(this_file_path, 'imgs_dirs', 'imageUrl')))
        self.assertEqual(5, len([image for image in dl]))

        for image in dl:
            self.assertTrue(torch.equal(torch.zeros(1, 3, 227, 227), image))

        self.assertTrue(mocked_response.called)
        shutil.rmtree(os.path.join(this_file_path, 'imgs_dirs'))

    def test_loading_images_full_path(self):

        # use the json file containing local paths to load the images

        resize_size = (100, 100)

        source_with_new_paths = JSONFile(self.full_source_path)

        dl = SkImageHogDescriptor(imgs_dirs=ds_path_without_field_name,
                                  resize_size=resize_size, batch_size=1).get_data_loader('imagePath',
                                                                                         source_with_new_paths)
        images_in_dl = [image for image in dl]

        self.assertTrue(not all(torch.all(image == 0) for image in images_in_dl))
        self.assertTrue(all(image.shape == (1, 3, 100, 100) for image in images_in_dl))
        self.assertEqual(5, len(images_in_dl))

        # check that at least one of the loaded images matches one of the images from the dataset

        test_image_path = os.path.join(ds_path, 'anthropologie-skirt-light-pink-434-1.lnk')
        test_image = PIL.Image.open(test_image_path).convert('RGB')
        test_image = TF.resize(TF.to_tensor(test_image), list(resize_size))
        test_image = test_image.unsqueeze(0)

        self.assertTrue(any(torch.equal(test_image, image_in_dl) for image_in_dl in images_in_dl))
        shutil.rmtree(ds_path_without_field_name)

    def test_loading_images_relative_path(self):

        # this test uses relative path so we should change the working dir
        current_workdir = os.getcwd()
        os.chdir(dir_root_repo)

        # use the json file containing local paths to load the images

        resize_size = (100, 100)

        source_with_new_paths = JSONFile(raw_source_path_local_rel)

        dl = SkImageHogDescriptor(imgs_dirs=ds_path_without_field_name,
                                  resize_size=resize_size, batch_size=1).get_data_loader('imagePath',
                                                                                         source_with_new_paths)
        images_in_dl = [image for image in dl]

        self.assertTrue(not all(torch.all(image == 0) for image in images_in_dl))
        self.assertTrue(all(image.shape == (1, 3, 100, 100) for image in images_in_dl))
        self.assertEqual(5, len(images_in_dl))

        # check that at least one of the loaded images matches one of the images from the dataset

        test_image_path = os.path.join(ds_path, 'anthropologie-skirt-light-pink-434-1.lnk')
        test_image = PIL.Image.open(test_image_path).convert('RGB')
        test_image = TF.resize(TF.to_tensor(test_image), list(resize_size))
        test_image = test_image.unsqueeze(0)

        self.assertTrue(any(torch.equal(test_image, image_in_dl) for image_in_dl in images_in_dl))
        shutil.rmtree(ds_path_without_field_name)

        # reset the previous workdir
        os.chdir(current_workdir)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.full_source_path)


if __name__ == "__main__":
    unittest.main()
