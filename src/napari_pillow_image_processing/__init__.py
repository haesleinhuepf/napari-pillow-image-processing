
import numpy as np
from toolz import curry
from functools import wraps
from typing import Callable
import inspect
from napari_plugin_engine import napari_hook_implementation
from napari_time_slicer import time_slicer
from napari_tools_menu import register_function
import napari

__version__ = "0.0.1"
__common_alias__ = "npil"


@napari_hook_implementation
def napari_experimental_provide_function():
    return [
        gaussian_blur,
        box_blur,
        unsharp_mask,
        median_filter,
        minimum_filter,
        maximum_filter,
        mode_filter,
        auto_contrast,
        equalize_histogram_bins,
        enhance_contrast,
        enhance_brightness,
        enhance_sharpness
]

def _pillow_to_numpy(pil_image):
    return np.asarray(pil_image)

def _numpy_to_pillow(np_image):
    from PIL import Image

    return Image.fromarray(np_image)


@curry
def plugin_function(
        function: Callable,
        convert_input_to_float: bool = False,
        convert_input_to_uint8: bool = False
) -> Callable:
    # copied from https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier0/_plugin_function.py
    @wraps(function)
    def worker_function(*args, **kwargs):
        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        input_shape = None

        # copy images to pyFAST-types, and create output array if necessary
        for key, value in bound.arguments.items():
            np_value = None
            if isinstance(value, np.ndarray):
                np_value = value
                input_shape = np_value.shape

            elif 'pyclesperanto_prototype._tier0._pycl.OCLArray' in str(type(value)) or \
                    'dask.array.core.Array' in str(type(value)):
                # compatibility with pyclesperanto and dask
                np_value = np.asarray(value)

            if convert_input_to_float and np_value is not None:
                np_value = np_value.astype(float)
            if convert_input_to_uint8 and np_value is not None:
                np_value = np_value.astype(np.uint8)

            if np_value is not None:
                if np_value.dtype == bool:
                    np_value = np_value * 1
                bound.arguments[key] = _numpy_to_pillow(np_value)

        # call the decorated function
        result = function(*bound.args, **bound.kwargs)
        import PIL

        if isinstance(result, PIL.Image.Image):
            result = _pillow_to_numpy(result)

        if input_shape is not None and result is not None:
            if len(input_shape) < len(result.shape):
                result = result[..., 0]

        return result

    worker_function.__module__ = "napari_pillow_image_processing"

    return worker_function


@curry
def slice_by_slice(function: Callable) -> Callable:
    @wraps(function)
    def worker_function(*args, **kwargs):
        args = list(args)
        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        # determine the largest z-stack in the parameter list
        max_z = 0
        for key, value in bound.arguments.items():
            if isinstance(value, np.ndarray) or str(type(value)) in ["<class 'cupy._core.core.ndarray'>",
                                                                     "<class 'dask.array.core.Array'>"]:
                if len(value.shape) == 3:
                    max_z = value.shape[0] - 1

        if max_z == 0:  # no stack in parameter list
            # just call the decorated function
            result = function(*bound.args, **bound.kwargs)
        else:
            # in case of 3D-data (stack) crop out the current 2D slice
            slices = []
            bound_arguments_copy = bound.arguments.copy()

            for z in range(max_z + 1):
                # replace 3D images with a given slice
                for key, value in bound_arguments_copy.items():
                    if isinstance(value, np.ndarray) or str(type(value)) in ["<class 'cupy._core.core.ndarray'>",
                                                                             "<class 'dask.array.core.Array'>"]:
                        if len(value.shape) == 3:
                            z_slice = min(z, value.shape[0] - 1)
                            image = value[z_slice]
                            bound.arguments[key] = image

                # call the decorated function
                slices.append(function(*bound.args, **bound.kwargs))
            result = np.asarray(slices)

        return result

    return worker_function

@register_function(menu="Filtering / noise removal > Gaussian blur (npil)")
@time_slicer
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def gaussian_blur(image: napari.types.ImageData, standard_deviation: float = 2) -> napari.types.ImageData:
    """
    Blurs the image with a sequence of extended box filters, which approximates a Gaussian kernel.
    For details on accuracy see [1].

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur
    """
    print(image.mode)

    from PIL import ImageFilter

    return image.filter(ImageFilter.GaussianBlur(standard_deviation))


@register_function(menu="Filtering / noise removal > Box blur (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def box_blur(image: napari.types.ImageData, radius: int = 2) -> napari.types.ImageData:
    """
    Blurs the image by setting each pixel to the average value of the pixels in a square box extending radius pixels in
    each direction. Supports float radius of arbitrary size. Uses an optimized implementation which runs in linear time
    relative to the size of the image for any radius value.

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.BoxBlur
    """
    from PIL import ImageFilter

    return image.filter(ImageFilter.BoxBlur(radius))


@register_function(menu="Filtering / edge enhancement > Unsharp mask (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def unsharp_mask(image: napari.types.ImageData, radius: int = 2, percent: int = 150, threshold: int = 3) -> napari.types.ImageData:
    """
    Unsharp mask filter.

    See Wikipedia’s entry on digital unsharp masking [1] for an explanation of the parameters.

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.UnsharpMask
    ..[1] https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking
    """
    from PIL import ImageFilter

    return image.filter(ImageFilter.UnsharpMask(radius, int(percent), int(threshold)))


@register_function(menu="Filtering / noise removal > Median filter (npil)")
@slice_by_slice
@plugin_function
def median_filter(image: napari.types.ImageData, radius: int = 1) -> napari.types.ImageData:
    """
    Picks the median pixel value in a window with the given size.

    Parameters
    ----------
    radius : int
        We pass `size = radius*2+1` as parameter to the pillow filter

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.MedianFilter
    """
    from PIL import ImageFilter

    return image.filter(ImageFilter.MedianFilter(radius * 2 + 1))


@register_function(menu="Filtering / noise removal > Minimum filter (npil)")
@slice_by_slice
@plugin_function
def minimum_filter(image: napari.types.ImageData, radius: int = 1) -> napari.types.ImageData:
    """
    Picks the minimum pixel value in a window with the given size.

    Parameters
    ----------
    radius : int
        We pass `size = radius*2+1` as parameter to the pillow filter

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.MinFilter
    """
    from PIL import ImageFilter

    return image.filter(ImageFilter.MinFilter(radius * 2 + 1))


@register_function(menu="Filtering / noise removal > Maximum filter (npil)")
@slice_by_slice
@plugin_function
def maximum_filter(image: napari.types.ImageData, radius: int = 1) -> napari.types.ImageData:
    """
    Picks the maximum pixel value in a window with the given size.

    Parameters
    ----------
    radius : int
        We pass `size = radius*2+1` as parameter to the pillow filter

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.MaxFilter
    """
    from PIL import ImageFilter

    return image.filter(ImageFilter.MaxFilter(radius * 2 + 1))


@register_function(menu="Filtering / noise removal > Mode filter (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def mode_filter(image: napari.types.ImageData, radius: int = 1) -> napari.types.ImageData:
    """
    Picks the most popular (mode) pixel value in a window with the given size.

    Parameters
    ----------
    radius : int
        We pass `size = radius*2+1` as parameter to the pillow filter

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.ModeFilter
    """
    from PIL import ImageFilter

    return image.filter(ImageFilter.ModeFilter(radius * 2 + 1))


@register_function(menu="Filtering > Auto contrast (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def auto_contrast(image: napari.types.ImageData, cutoff_percent: float = 0) -> napari.types.ImageData:
    """
    Maximize (normalize) image contrast. This function calculates a histogram of the input image (or mask region),
    removes cutoff percent of the lightest and darkest pixels from the histogram, and remaps the image so that the
    darkest pixel becomes black (0), and the lightest becomes white (255).

    Note: As operations are applied slice-by-slice in 3D image stacks, contrast can vary from slice to slice.
    It is recommended to apply this operation to 2D images only.

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.autocontrast
    """
    from PIL import ImageOps

    return ImageOps.autocontrast(image, cutoff=cutoff_percent, ignore=None, mask=None, preserve_tone=False)


@register_function(menu="Filtering > Equalize histogram bins (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def equalize_histogram_bins(image: napari.types.ImageData) -> napari.types.ImageData:
    """
    Equalize the image histogram. This function applies a non-linear mapping to the input image, in order to create a
    uniform distribution of grayscale values in the output image.

    This function is NOT comparable to ImageJ's Bleaching Correction with the option to equalize the histogram.
    Instead it makes all histogram bins similar to each other in a single image (slice).

    Note: As operations are applied slice-by-slice in 3D image stacks, contrast can vary from slice to slice.
    It is recommended to apply this operation to 2D images only.

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.equalize
    """
    from PIL import ImageOps

    return ImageOps.equalize(image)


@register_function(menu="Filtering > Enhance contrast (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def enhance_contrast(image: napari.types.ImageData, enhancement_factor: float = 1) -> napari.types.ImageData:
    """
    This function can be used to control the contrast of an image, similar to the contrast control on a TV set.
    An enhancement factor of 0.0 gives a solid grey image. A factor of 1.0 gives the original image.

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Contrast
    """
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(enhancement_factor)

@register_function(menu="Filtering > Enhance brightness (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def enhance_brightness(image: napari.types.ImageData, enhancement_factor: float = 1) -> napari.types.ImageData:
    """
    This function can be used to control the brightness of an image.
    An enhancement factor of 0.0 gives a black image. A factor of 1.0 gives the original image.

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Brightness
    """
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(enhancement_factor)


@register_function(menu="Filtering > Enhance sharpness (npil)")
@slice_by_slice
@plugin_function(convert_input_to_uint8=True)
def enhance_sharpness(image: napari.types.ImageData, enhancement_factor: float = 1) -> napari.types.ImageData:
    """
    This function can be used to adjust the sharpness of an image.
    An enhancement factor of 0.0 gives a blurred image, a factor of 1.0
    gives the original image, and a factor of 2.0 gives a sharpened image.

    See Also
    --------
    ..[0] https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Sharpness
    """
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(enhancement_factor)

