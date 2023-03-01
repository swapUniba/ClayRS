import random
import unittest
from typing import Optional

import torch
from torchvision import transforms

import clayrs.content_analyzer.information_processor.visual_preprocessors.torch_builtin_transformer as clayrs_transforms


class TestTorchBuiltInTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # we test 10 images which are of size 28x56 with 3 channels
        cls.batch_images = torch.randn(size=(10, 3, 28, 56))

        # each subclass needs only to overwrite these with setup() method and call
        # assertTrue on expected_result_equal()
        cls.technique: Optional[clayrs_transforms.TorchBuiltInTransformer] = None
        cls.og_technique = None

    def expected_result_equal(self):

        # setting manual seed for those preprocessors which randomly apply the transformation
        torch.manual_seed(42)
        random.seed(42)
        expected = self.og_technique(self.batch_images)

        torch.manual_seed(42)
        random.seed(42)
        result = self.technique(self.batch_images)

        return torch.equal(expected, result)


class TestTorchCompose(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchCompose(transforms_list=[clayrs_transforms.TorchGrayscale(),
                                                                         clayrs_transforms.TorchCenterCrop(10)])
        self.og_technique = transforms.Compose([transforms.Grayscale(),
                                                transforms.CenterCrop(10)])

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchConvertImageDtype(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchConvertImageDtype(dtype=torch.uint8)
        self.og_technique = transforms.ConvertImageDtype(dtype=torch.uint8)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchCenterCrop(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchCenterCrop(10)
        self.og_technique = transforms.CenterCrop(10)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchGrayScale(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchGrayscale()
        self.og_technique = transforms.Grayscale()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchGaussianBlur(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchGaussianBlur(kernel_size=5)
        self.og_technique = transforms.GaussianBlur(kernel_size=5)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchNormalize(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.og_technique = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchLambda(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchLambda(lambda x: x + 1)
        self.og_technique = transforms.Lambda(lambda x: x + 1)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchPad(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchPad(5)
        self.og_technique = transforms.Pad(5)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomApply(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomApply([clayrs_transforms.TorchGrayscale(),
                                                             clayrs_transforms.TorchPad(5)])
        self.og_technique = transforms.RandomApply([transforms.Grayscale(), transforms.Pad(5)])

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomChoice(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomChoice([clayrs_transforms.TorchGrayscale(),
                                                              clayrs_transforms.TorchPad(5),
                                                              clayrs_transforms.TorchCenterCrop(15)])
        self.og_technique = transforms.RandomChoice([transforms.Grayscale(),
                                                     transforms.Pad(5),
                                                     transforms.CenterCrop(15)])

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomOrder(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomOrder([clayrs_transforms.TorchGrayscale(),
                                                             clayrs_transforms.TorchPad(5),
                                                             clayrs_transforms.TorchCenterCrop(15)])
        self.og_technique = transforms.RandomOrder([transforms.Grayscale(),
                                                    transforms.Pad(5),
                                                    transforms.CenterCrop(15)])

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomCrop(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomCrop(10)
        self.og_technique = transforms.RandomCrop(10)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomHorizontalFlip(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomHorizontalFlip()
        self.og_technique = transforms.RandomHorizontalFlip()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomVeticalFlip(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomVerticalFlip()
        self.og_technique = transforms.RandomVerticalFlip()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomResizedCrop(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomResizedCrop(5)
        self.og_technique = transforms.RandomResizedCrop(5)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchResize(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchResize((10, 10))
        self.og_technique = transforms.Resize((10, 10))

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchLinearTransformation(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        transf_matrix = torch.randn((3*28*56, 3*28*56))
        mean_vector = torch.randn(3*28*56)
        self.technique = clayrs_transforms.TorchLinearTransformation(transf_matrix, mean_vector)
        self.og_technique = transforms.LinearTransformation(transf_matrix, mean_vector)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchColorJitter(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchColorJitter()
        self.og_technique = transforms.ColorJitter()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomRotation(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomRotation(degrees=12)
        self.og_technique = transforms.RandomRotation(degrees=12)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomAffine(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomAffine(degrees=12)
        self.og_technique = transforms.RandomAffine(degrees=12)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomGrayScale(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomGrayscale(p=0.6)
        self.og_technique = transforms.RandomGrayscale(p=0.6)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomPerspective(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomPerspective()
        self.og_technique = transforms.RandomPerspective()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomErasing(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomErasing()
        self.og_technique = transforms.RandomErasing()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomInvert(TestTorchBuiltInTransformer):

    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomInvert()
        self.og_technique = transforms.RandomInvert()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomPosterize(TestTorchBuiltInTransformer):
    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomPosterize(bits=2)
        self.og_technique = transforms.RandomPosterize(bits=2)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomSolarize(TestTorchBuiltInTransformer):
    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomSolarize(threshold=0.55)
        self.og_technique = transforms.RandomSolarize(threshold=0.55)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomAdjustSharpness(TestTorchBuiltInTransformer):
    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomAdjustSharpness(sharpness_factor=0)
        self.og_technique = transforms.RandomAdjustSharpness(sharpness_factor=0)

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TestTorchRandomAutocontrast(TestTorchBuiltInTransformer):
    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomAutocontrast()
        self.og_technique = transforms.RandomAutocontrast()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


class TorchRandomEqualize(TestTorchBuiltInTransformer):
    def setUp(self) -> None:

        self.technique = clayrs_transforms.TorchRandomEqualize()
        self.og_technique = transforms.RandomEqualize()

    def test_forward(self):
        self.assertTrue(self.expected_result_equal())


if __name__ == "__main__":
    unittest.main()
