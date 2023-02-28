from typing import List, Optional

from torchvision import transforms
from torchvision.transforms import InterpolationMode, AutoAugmentPolicy

from clayrs.content_analyzer.information_processor.visual_preprocessors.torch_builtin_transformer import \
    TorchBuiltInTransformer

__all__ = [
    "AutoAugmentPolicy",
    "TorchAutoAugment",
    "TorchRandAugment",
    "TorchTrivialAugmentWide"
]


# AUGMENTERS

class TorchAutoAugment(TorchBuiltInTransformer):
    """
    Class that implements the AutoAugment Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer AutoAugment directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.AutoAugment.html)

    NOTE: the augmented result will SUBSTITUTE the original input
    """
    def __init__(self, policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None):

        super().__init__(transforms.AutoAugment(policy, interpolation, fill))


class TorchRandAugment(TorchBuiltInTransformer):
    """
    Class that implements the RandAugment Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer RandAugment directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html)

    NOTE: the augmented result will SUBSTITUTE the original input
    """
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__(transforms.RandAugment(num_ops, magnitude, num_magnitude_bins, interpolation, fill))


class TorchTrivialAugmentWide(TorchBuiltInTransformer):
    """
    Class that implements the TrivialAugmentWide Transformer from torchvision.
    The parameters one could pass are the same ones you would pass instantiating
    the transformer TrivialAugmentWide directly from torchvision.

    TorchVision documentation: [here](https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html)

    NOTE: the augmented result will SUBSTITUTE the original input
    """
    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__(transforms.TrivialAugmentWide(num_magnitude_bins, interpolation, fill))
