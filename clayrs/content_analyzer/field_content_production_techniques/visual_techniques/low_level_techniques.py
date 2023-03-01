from __future__ import annotations
from abc import abstractmethod
import inspect
from typing import List, TYPE_CHECKING, Tuple, Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import convolve
from skimage.feature import hog, canny, SIFT, local_binary_pattern
from sklearn.cluster import KMeans

from clayrs.content_analyzer.content_representation.content import EmbeddingField
from clayrs.content_analyzer.field_content_production_techniques.visual_techniques.visual_content_techniques import \
    VisualContentTechnique
from clayrs.utils.automatic_methods import autorepr
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.information_processor_abstract import ImageProcessor
    from clayrs.content_analyzer.information_processor.postprocessors.postprocessor import \
        EmbeddingInputPostProcessor


class LowLevelVisual(VisualContentTechnique):
    """
    Technique which encapsulates the logic for all visual techniques that work at a low level, that is instead of
    working on batches of images in an efficient way, these techniques require to process each image separately
    (because, for example, they need to analyze the single pixels of the images).
    """

    def __init__(self, imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):
        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)

    def produce_content(self, field_name: str, preprocessor_list: List[ImageProcessor],
                        postprocessor_list: List[EmbeddingInputPostProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:

        dl = self.get_data_loader(field_name, source)

        representation_list: [FieldRepresentation] = []

        with get_progbar(dl) as pbar:

            for data_batch in pbar:
                pbar.set_description(f"Processing and producing contents with {self}")

                for content_data in data_batch:
                    processed_data = self.process_data(content_data, preprocessor_list)
                    representation_list.append(self.produce_single_repr(processed_data))

        representation_list = self.postprocess_representations(representation_list, postprocessor_list)

        return representation_list

    @abstractmethod
    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:
        raise NotImplementedError


class SkImageHogDescriptor(LowLevelVisual):
    """
    Low level technique which implements the Hog Descriptor using the SkImage library

    Parameters are the same ones you would pass to the hog function in SkImage together with some framework specific
    parameters

    Arguments for [SkImage Hog](https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=hog#skimage.feature.hog)

    Args:
        flatten: whether the output of the technique should be flattened or not
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                 block_norm='L2-Hys', transform_sqrt=False, flatten: bool = False,
                 imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self.hog = lambda x, channel_axis: hog(x, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                               cells_per_block=cells_per_block, block_norm=block_norm,
                                               transform_sqrt=transform_sqrt, feature_vector=flatten,
                                               channel_axis=channel_axis)

        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor) -> EmbeddingField:

        # set channel axis for hog method
        # if image is grayscale but has the additional channel specified (for example, (1, 16, 16)) the additional
        # channel is removed and channel axis is set to None
        # otherwise, the RGB channel should always be the first one
        channel_axis = None
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            channel_axis = 0

        hog_features = self.hog(field_data.numpy(), channel_axis=channel_axis)
        return EmbeddingField(hog_features)

    def __str__(self):
        return "SkImageHogDescriptor"

    def __repr__(self):
        return self._repr_string


class SkImageCannyEdgeDetector(LowLevelVisual):
    """
    Low level technique which implements the Canny Edge Detector using the SkImage library

    Parameters are the same ones you would pass to the canny function in SkImage together with some framework specific
    parameters

    Arguments for [SkImage Canny](https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=hog#skimage.feature.canny)

    Args:
        flatten: whether the output of the technique should be flattened or not
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False,
                 mode='constant', cval=0.0, flatten: bool = False,
                 imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self.canny = lambda x: canny(image=x, sigma=sigma, low_threshold=low_threshold,
                                     high_threshold=high_threshold, mask=mask,
                                     use_quantiles=use_quantiles, mode=mode, cval=cval)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor) -> EmbeddingField:

        # canny edge detector requires grayscale images
        # if image is grayscale but has the additional channel specified (for example, (1, 16, 16)) the additional
        # channel is removed
        # otherwise, the rgb image is converted to grayscale
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        canny_features = self.canny(field_data.numpy())
        canny_features = canny_features.flatten() if self.flatten else canny_features
        return EmbeddingField(canny_features.astype(int))

    def __str__(self):
        return "SkImageCannyEdgeDetector"

    def __repr__(self):
        return self._repr_string


class CustomFilterConvolution(LowLevelVisual):
    """
    Low level technique which implements a custom filter for convolution over an image, using the convolve method from
    the scipy library

    Parameters are the same ones you would pass to the convolve function in scipy together with some framework specific
    parameters

    Arguments for [Scipy convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html)

    Args:
        flatten: whether the output of the technique should be flattened or not
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, weights, mode='reflect', cval=0.0, origin=0, flatten: bool = False,
                 imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self.convolve = lambda x: convolve(x, weights=weights, mode=mode, cval=cval, origin=origin)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor) -> EmbeddingField:

        # convolution requires grayscale images
        # if image is grayscale but has the additional channel specified (for example, (1, 16, 16)) the additional
        # channel is removed
        # otherwise, the rgb image is converted to grayscale
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        convolution_result = self.convolve(field_data.numpy())
        convolution_result = convolution_result.flatten() if self.flatten else convolution_result
        return EmbeddingField(convolution_result.astype(int))

    def __str__(self):
        return "CustomFilterConvolution"

    def __repr__(self):
        return self._repr_string


class ColorsHist(LowLevelVisual):
    """
    Low level technique which generates a color histogram for each channel of each RGB image

    The technique retrieves all the values for each one of the three RGB channels in the image, flattens them and
    returns an EmbeddingField representation containing a numpy two-dimensional array with three rows (one for each
    channel)

    Args:
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor) -> EmbeddingField:

        # color histogram requires RGB images
        # if image is grayscale but has the additional channel specified (for example, (1, 16, 16)) the additional
        # channel is removed
        # then, if the image is in grayscale, it is converted to RGB
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()

        if field_data.shape[0] != 3:
            field_data = field_data.repeat(3, 1, 1)
            logger.warning(f'Grayscale images were detected, {self} only accepts RGB images! '
                           'The images will be converted to RGB')

        numpy_field_data = field_data.numpy() * 255
        colors = np.array([numpy_field_data[0, :, :].flatten(),
                           numpy_field_data[1, :, :].flatten(),
                           numpy_field_data[2, :, :].flatten()])
        return EmbeddingField(colors.astype(int))

    def __str__(self):
        return "ColorHist"

    def __repr__(self):
        return self._repr_string


class ColorQuantization(LowLevelVisual):
    """
    Low level technique which returns the colors obtained from applying a clustering technique (in this case KMeans)

    Arguments for [SkLearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

    Args:
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, n_colors: Any = 3, init: Any = "k-means++", n_init: Any = 10, max_iter: Any = 300,
                 tol: Any = 1e-4, random_state: Any = None, copy_x: Any = True, algorithm: Any = "auto",
                 flatten: bool = False, imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self.k_means = KMeans(n_clusters=n_colors, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                              random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        # color quantization requires RGB images
        # if image is grayscale but has the additional channel specified (for example, (1, 16, 16)) the additional
        # channel is removed
        # then, if the image is in grayscale, it is converted to RGB
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()

        if field_data.shape[0] != 3:
            field_data = field_data.repeat(3, 1, 1)
            logger.warning(f'Grayscale images were detected, {self} only accepts RGB images! '
                           'The images will be converted to RGB')

        numpy_field_data = field_data.numpy() * 255
        numpy_field_data = numpy_field_data.reshape((numpy_field_data.shape[1] * numpy_field_data.shape[2], 3))
        self.k_means.fit(numpy_field_data)
        dominant_colors = self.k_means.cluster_centers_
        dominant_colors = dominant_colors.flatten() if self.flatten else dominant_colors
        return EmbeddingField(dominant_colors.astype(int))

    def __str__(self):
        return "ColorQuantization"

    def __repr__(self):
        return self._repr_string


class SkImageSIFT(LowLevelVisual):
    """
    Low level technique which allows for SIFT feature detection from SkImage

    Parameters are the same ones you would pass to the SIFT object in SkImage together with some framework specific
    parameters

    Arguments for [SkImage SIFT](https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=hog#skimage.feature.SIFT)

    Args:
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5, c_dog=0.013333333333333334,
                 c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8,
                 flatten: bool = False, imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self.sift = SIFT(upsampling=upsampling, n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min,
                         sigma_in=sigma_in, c_dog=c_dog, c_edge=c_edge, n_bins=n_bins, lambda_ori=lambda_ori,
                         c_max=c_max, lambda_descr=lambda_descr, n_hist=n_hist, n_ori=n_ori)
        self.flatten = flatten
        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        # the SIFT technique requires Grayscale images
        # if image is grayscale but has the additional channel specified (for example, (1, 16, 16)) the additional
        # channel is removed
        # otherwise, if the image is in RGB, it is converted to grayscale
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        self.sift.detect_and_extract(field_data.numpy())
        descriptors = self.sift.descriptors
        descriptors = descriptors.flatten() if self.flatten else descriptors
        return EmbeddingField(descriptors)

    def __str__(self):
        return "SkImageSIFT"

    def __repr__(self):
        return self._repr_string


class SkImageLBP(LowLevelVisual):
    """
    Low level technique which allows for LBP feature detection from SkImage

    Parameters are the same ones you would pass to the local_binary_pattern function in SkImage together with some
    framework specific parameters

    Furthermore, in this case, there is also an additional parameter, that is 'as_image'

    Arguments for [SkImage lbp](https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=hog#skimage.feature.draw_multiblock_lbp)

    Args:
        as_image: if True, the lbp image obtained from SkImage will be returned, otherwise the number of occurences
            of each binary pattern will be returned (as if it was a feature vector)
        imgs_dirs: directory where the images are stored (or will be stored in the case of fields containing links)
        max_timeout: maximum time to wait before considering a request failed (image from link)
        max_retries: maximum number of retries to retrieve an image from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the images dataloader
        resize_size: since the Tensorflow dataset requires all images to be of the same size, they will all be resized
            to the specified size. Note that if you were to specify a resize transformer in the preprocessing pipeline,
            the size specified in the latter will be the final resize size
    """

    def __init__(self, p: int, r: float, method='default', flatten: bool = False, as_image: bool = False,
                 imgs_dirs: str = "imgs_dirs", max_timeout: int = 2, max_retries: int = 5,
                 max_workers: int = 0, batch_size: int = 64, resize_size: Tuple[int, int] = (227, 227)):

        super().__init__(imgs_dirs, max_timeout, max_retries, max_workers, batch_size, resize_size)
        self.lbp = lambda x: local_binary_pattern(x, P=p, R=r, method=method)
        self.flatten = flatten
        self.as_image = as_image
        self._repr_string = autorepr(self, inspect.currentframe())

    def produce_single_repr(self, field_data: torch.Tensor) -> FieldRepresentation:

        # the LBP technique requires Grayscale images
        # if image is grayscale but has the additional channel specified (for example, (1, 16, 16)) the additional
        # channel is removed
        # otherwise, if the image is in RGB, it is converted to grayscale
        if field_data.shape[0] == 1:
            field_data = field_data.squeeze()
        else:
            field_data = TF.rgb_to_grayscale(field_data).squeeze()

        lbp_image = self.lbp(field_data.numpy())

        if self.as_image:
            return EmbeddingField(lbp_image.flatten()) if self.flatten else EmbeddingField(lbp_image)

        _, occurrences = np.unique(lbp_image, return_counts=True)
        return EmbeddingField(occurrences)

    def __str__(self):
        return "SkImageLBP"

    def __repr__(self):
        return self._repr_string