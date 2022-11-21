from typing import Any, List

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from clayrs.content_analyzer.information_processor.information_processor import ImageProcessor

__all__ = [
    "TorchCompose",
    "TorchConvertImageDtype",
    "TorchNormalize",
    "TorchResize",
    "TorchCenterCrop",
    "TorchPad",
    "TorchLambda",
    "TorchRandomApply",
    "TorchRandomChoice",
    "TorchRandomOrder",
    "TorchRandomCrop",
    "TorchRandomHorizontalFlip",
    "TorchRandomVerticalFlip",
    "TorchRandomResizedCrop",
    "TorchLinearTransformation",
    "TorchColorJitter",
    "TorchRandomRotation",
    "TorchRandomAffine",
    "TorchGrayscale",
    "TorchRandomGrayscale",
    "TorchRandomPerspective",
    "TorchRandomErasing",
    "TorchGaussianBlur",
    "InterpolationMode",
    "TorchRandomInvert",
    "TorchRandomPosterize",
    "TorchRandomSolarize",
    "TorchRandomAdjustSharpness",
    "TorchRandomAutocontrast",
    "TorchRandomEqualize",
]


class TorchBuiltInTransformer(ImageProcessor):

    def __init__(self, torch_builtin_transformer):
        super().__init__()

        self.transformer = torch_builtin_transformer

    def forward(self, field_data):
        return self.transformer(field_data)

    def __str__(self):
        # each built in transformer has as str the string "class name",
        # what clayrs expects
        return str(self.transformer)

    def __eq__(self, other):
        return self.transformer == other.transformer

    def __repr__(self):
        # each built in transformer has as repr the string "class_name(param1=val1, ...)",
        # what clayrs expects
        return repr(self.transformer)


class TorchCompose(TorchBuiltInTransformer):
    def __init__(self, transforms_list: List[ImageProcessor]):
        super().__init__(transforms.Compose(transforms_list))


class TorchConvertImageDtype(TorchBuiltInTransformer):
    def __init__(self, dtype: torch.dtype):
        super().__init__(transforms.ConvertImageDtype(dtype))


class TorchCenterCrop(TorchBuiltInTransformer):

    def __init__(self, size: int):
        super().__init__(transforms.CenterCrop(size))


class TorchGrayscale(TorchBuiltInTransformer):

    def __init__(self, num_output_channels: int = 1):
        super().__init__(transforms.Grayscale(num_output_channels))


class TorchGaussianBlur(TorchBuiltInTransformer):

    def __init__(self, kernel_size: int, sigma: Any = (0.1, 2.0)):
        super().__init__(transforms.GaussianBlur(kernel_size, sigma))


class TorchNormalize(TorchBuiltInTransformer):

    def __init__(self, mean, std, inplace=False):
        super().__init__(transforms.Normalize(mean, std, inplace))


class TorchLambda(TorchBuiltInTransformer):

    def __init__(self, lambd: callable):
        super().__init__(transforms.Lambda(lambd))


class TorchPad(TorchBuiltInTransformer):

    def __init__(self, padding, fill: int = 0, padding_mode: str = "constant"):
        super().__init__(transforms.Pad(padding, fill, padding_mode))


class TorchRandomApply(TorchBuiltInTransformer):

    def __init__(self, transforms_list: List[ImageProcessor], p=0.5):
        super().__init__(transforms.RandomApply(transforms_list, p))


class TorchRandomChoice(TorchBuiltInTransformer):

    def __init__(self, transforms_list: List[ImageProcessor], p=0.5):
        super().__init__(transforms.RandomChoice(transforms_list, p))


class TorchRandomOrder(TorchBuiltInTransformer):
    def __init__(self, transforms_list: List[ImageProcessor]):
        super().__init__(transforms.RandomOrder(transforms_list))


class TorchRandomCrop(TorchBuiltInTransformer):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(transforms.RandomCrop(size, padding, pad_if_needed, fill, padding_mode))


class TorchRandomHorizontalFlip(TorchBuiltInTransformer):
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomHorizontalFlip(p))


class TorchRandomVerticalFlip(TorchBuiltInTransformer):
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomVerticalFlip(p))


class TorchRandomResizedCrop(TorchBuiltInTransformer):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BILINEAR):
        super().__init__(transforms.RandomResizedCrop(size, scale, ratio, interpolation))


class TorchResize(TorchBuiltInTransformer):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(transforms.Resize(size, interpolation, max_size, antialias))


class TorchLinearTransformation(TorchBuiltInTransformer):
    def __init__(self, transformation_matrix, mean_vector):
        super().__init__(transforms.LinearTransformation(transformation_matrix, mean_vector))


class TorchColorJitter(TorchBuiltInTransformer):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(transforms.ColorJitter(brightness, contrast, saturation, hue))


class TorchRandomRotation(TorchBuiltInTransformer):
    def __init__(self, degrees, interpolation=InterpolationMode.NEAREST, expand=False,
                 center=None, fill=0, resample=None):
        super().__init__(transforms.RandomRotation(degrees, interpolation, expand, center, fill, resample))


class TorchRandomAffine(TorchBuiltInTransformer):
    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST,
                 fill=0, fillcolor=None, resample=None, center=None):

        super().__init__(transforms.RandomAffine(degrees, translate, scale, shear, interpolation, fill, fillcolor,
                                                 resample, center))


class TorchRandomGrayscale(TorchBuiltInTransformer):
    def __init__(self, p: float = 0.1):
        super().__init__(transforms.RandomGrayscale(p))


class TorchRandomPerspective(TorchBuiltInTransformer):
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__(transforms.RandomPerspective(distortion_scale, p, interpolation, fill))


class TorchRandomErasing(TorchBuiltInTransformer):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(transforms.RandomErasing(p, scale, ratio, value, inplace))


class TorchRandomInvert(TorchBuiltInTransformer):
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomInvert(p))


class TorchRandomPosterize(TorchBuiltInTransformer):
    def __init__(self, bits, p: float = 0.5):
        super().__init__(transforms.RandomPosterize(bits, p))


class TorchRandomSolarize(TorchBuiltInTransformer):
    def __init__(self, threshold, p: float = 0.5):
        super().__init__(transforms.RandomSolarize(threshold, p))


class TorchRandomAdjustSharpness(TorchBuiltInTransformer):
    def __init__(self, sharpness_factor, p: float = 0.5):
        super().__init__(transforms.RandomAdjustSharpness(sharpness_factor, p))


class TorchRandomAutocontrast(TorchBuiltInTransformer):
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomAutocontrast(p))


class TorchRandomEqualize(TorchBuiltInTransformer):
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomEqualize(p))
