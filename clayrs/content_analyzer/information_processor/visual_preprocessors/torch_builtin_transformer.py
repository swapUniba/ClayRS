from typing import Any, List, Optional, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from clayrs.content_analyzer.information_processor.information_processor_abstract import ImageProcessor

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
    """
    Abstract class for Transformers preprocessors from the torchvision library
    """

    def __init__(self, torch_builtin_transformer):
        super().__init__()

        self.transformer = torch_builtin_transformer

    def forward(self, field_data: torch.Tensor) -> torch.Tensor:
        """
        Method to apply the given transformer to the data contained in the field

        NOTE: the data is expected to be an image converted to a torch tensor, this technique will not work on
        textual data
        """
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
    """
    Class that implements the Compose Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Compose directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html)

    The only difference w.r.t. the TorchVision implementation is that while the original implementation expects a
    list of Transformer objects as parameter, this implementation expects a list of ImageProcessor (so other
    image pre-processors) as parameter
    """
    def __init__(self, transforms_list: List[ImageProcessor]):
        super().__init__(transforms.Compose(transforms_list))


class TorchConvertImageDtype(TorchBuiltInTransformer):
    """
    Class that implements the ConvertImageDtype Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer ConvertImageDtype directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.ConvertImageDtype.html)
    """
    def __init__(self, dtype: torch.dtype):
        super().__init__(transforms.ConvertImageDtype(dtype))


class TorchCenterCrop(TorchBuiltInTransformer):
    """
    Class that implements the CenterCrop Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer CenterCrop directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html)
    """

    def __init__(self, size: int):
        super().__init__(transforms.CenterCrop(size))


class TorchGrayscale(TorchBuiltInTransformer):
    """
    Class that implements the Grayscale Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Grayscale directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.Grayscale.html)
    """

    def __init__(self, num_output_channels: int = 1):
        super().__init__(transforms.Grayscale(num_output_channels))


class TorchGaussianBlur(TorchBuiltInTransformer):
    """
    Class that implements the GaussianBlur Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer GaussianBlur directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html)
    """

    def __init__(self, kernel_size: int, sigma: Any = (0.1, 2.0)):
        super().__init__(transforms.GaussianBlur(kernel_size, sigma))


class TorchNormalize(TorchBuiltInTransformer):
    """
    Class that implements the Normalize Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Normalize directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)
    """

    def __init__(self, mean: Any, std: Any):
        super().__init__(transforms.Normalize(mean, std, inplace=False))


class TorchLambda(TorchBuiltInTransformer):
    """
    Class that implements the Lambda Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Lambda directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.Lambda.html)
    """

    def __init__(self, lambd: callable):
        super().__init__(transforms.Lambda(lambd))


class TorchPad(TorchBuiltInTransformer):
    """
    Class that implements the Pad Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Pad directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.Pad.html)
    """

    def __init__(self, padding: int, fill: int = 0, padding_mode: str = "constant"):
        super().__init__(transforms.Pad(padding, fill, padding_mode))


class TorchRandomApply(TorchBuiltInTransformer):
    """
    Class that implements the RandomApply Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomApply directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomApply.html)

    The only difference w.r.t. the TorchVision implementation is that while the original implementation expects a
    list of Transformer objects as parameter, this implementation expects a list of ImageProcessor (so other
    image pre-processors) as parameter
    """

    def __init__(self, transforms_list: List[ImageProcessor], p: float = 0.5):
        super().__init__(transforms.RandomApply(transforms_list, p))


class TorchRandomChoice(TorchBuiltInTransformer):
    """
    Class that implements the RandomChoice Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomChoice directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomChoice.html)

    The only difference w.r.t. the TorchVision implementation is that while the original implementation expects a
    list of Transformer objects as parameter, this implementation expects a list of ImageProcessor (so other
    image pre-processors) as parameter
    """

    def __init__(self, transforms_list: List[ImageProcessor], p: Any = None):
        super().__init__(transforms.RandomChoice(transforms_list, p))


class TorchRandomOrder(TorchBuiltInTransformer):
    """
    Class that implements the RandomOrder Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomOrder directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomOrder.html)

    The only difference w.r.t. the TorchVision implementation is that while the original implementation expects a
    list of Transformer objects as parameter, this implementation expects a list of ImageProcessor (so other
    image pre-processors) as parameter
    """
    def __init__(self, transforms_list: List[ImageProcessor]):
        super().__init__(transforms.RandomOrder(transforms_list))


class TorchRandomCrop(TorchBuiltInTransformer):
    """
    Class that implements the RandomCrop Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomCrop directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html)
    """
    def __init__(self, size: int, padding: Any = None, pad_if_needed: bool = False,
                 fill: tuple = 0, padding_mode: str = "constant"):
        super().__init__(transforms.RandomCrop(size, padding, pad_if_needed, fill, padding_mode))


class TorchRandomHorizontalFlip(TorchBuiltInTransformer):
    """
    Class that implements the RandomHorizontalFlip Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomHorizontalFlip directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html)
    """
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomHorizontalFlip(p))


class TorchRandomVerticalFlip(TorchBuiltInTransformer):
    """
    Class that implements the RandomVerticalFlip Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomVerticalFlip directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomVerticalFlip.html)
    """
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomVerticalFlip(p))


class TorchRandomResizedCrop(TorchBuiltInTransformer):
    """
    Class that implements the RandomResizedCrop Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomResizedCrop directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)
    """
    def __init__(self, size: int, scale: Tuple[float] = (0.08, 1.0), ratio: Tuple[float] = (3.0 / 4.0, 4.0 / 3.0),
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR, antialias: Optional[bool] = None):
        super().__init__(transforms.RandomResizedCrop(size, scale, ratio, interpolation, antialias))


class TorchResize(TorchBuiltInTransformer):
    """
    Class that implements the Resize Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer Resize directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html)
    """
    def __init__(self, size: int, interpolation=InterpolationMode.BILINEAR, max_size: Any = None,
                 antialias: Any = None):
        super().__init__(transforms.Resize(size, interpolation, max_size, antialias))


class TorchLinearTransformation(TorchBuiltInTransformer):
    """
    Class that implements the LinearTransformation Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer LinearTransformation directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.LinearTransformation.html)
    """
    def __init__(self, transformation_matrix: torch.Tensor, mean_vector: torch.Tensor):
        super().__init__(transforms.LinearTransformation(transformation_matrix, mean_vector))


class TorchColorJitter(TorchBuiltInTransformer):
    """
    Class that implements the ColorJitter Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer ColorJitter directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)
    """
    def __init__(self, brightness: Any = 0, contrast: Any = 0, saturation: Any = 0, hue: Any = 0):
        super().__init__(transforms.ColorJitter(brightness, contrast, saturation, hue))


class TorchRandomRotation(TorchBuiltInTransformer):
    """
    Class that implements the RandomRotation Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomRotation directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomRotation.html)
    """
    def __init__(self, degrees: Any, interpolation: InterpolationMode = InterpolationMode.NEAREST, expand: Any = False,
                 center: Any = None, fill: Any = 0):
        super().__init__(transforms.RandomRotation(degrees, interpolation, expand, center, fill))


class TorchRandomAffine(TorchBuiltInTransformer):
    """
    Class that implements the RandomAffine Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomAffine directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAffine.html)
    """
    def __init__(self, degrees: Any, translate: Any = None, scale: Any = None,
                 shear: Any = None, interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Any = 0, center: Any = None):

        super().__init__(transforms.RandomAffine(degrees, translate, scale, shear, interpolation, fill, center))


class TorchRandomGrayscale(TorchBuiltInTransformer):
    """
    Class that implements the RandomGrayscale Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomGrayscale directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomGrayscale.html)
    """
    def __init__(self, p: float = 0.1):
        super().__init__(transforms.RandomGrayscale(p))


class TorchRandomPerspective(TorchBuiltInTransformer):
    """
    Class that implements the RandomPerspective Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomPerspective directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomPerspective.html)
    """
    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR, fill: Any = 0):
        super().__init__(transforms.RandomPerspective(distortion_scale, p, interpolation, fill))


class TorchRandomErasing(TorchBuiltInTransformer):
    """
    Class that implements the RandomErasing Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomErasing directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html)
    """
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: int = 0, inplace: bool = False):
        super().__init__(transforms.RandomErasing(p, scale, ratio, value, inplace))


class TorchRandomInvert(TorchBuiltInTransformer):
    """
    Class that implements the RandomInvert Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomInvert directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomInvert.html)
    """
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomInvert(p))


class TorchRandomPosterize(TorchBuiltInTransformer):
    """
    Class that implements the RandomPosterize Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomPosterize directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomPosterize.html)
    """
    def __init__(self, bits: int, p: float = 0.5):
        super().__init__(transforms.RandomPosterize(bits, p))


class TorchRandomSolarize(TorchBuiltInTransformer):
    """
    Class that implements the RandomSolarize Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomSolarize directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomSolarize.html)
    """
    def __init__(self, threshold: float, p: float = 0.5):
        super().__init__(transforms.RandomSolarize(threshold, p))


class TorchRandomAdjustSharpness(TorchBuiltInTransformer):
    """
    Class that implements the RandomAdjustSharpness Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomAdjustSharpness directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAdjustSharpness.html)
    """
    def __init__(self, sharpness_factor: float, p: float = 0.5):
        super().__init__(transforms.RandomAdjustSharpness(sharpness_factor, p))


class TorchRandomAutocontrast(TorchBuiltInTransformer):
    """
    Class that implements the RandomAutocontrast Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomAutocontrast directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAutocontrast.html)
    """
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomAutocontrast(p))


class TorchRandomEqualize(TorchBuiltInTransformer):
    """
    Class that implements the RandomEqualize Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandomEqualize directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomEqualize.html)
    """
    def __init__(self, p: float = 0.5):
        super().__init__(transforms.RandomEqualize(p))
