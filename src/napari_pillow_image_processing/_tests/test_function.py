import napari_pillow_image_processing as npil

import numpy as np

def test_all_operations():
    functions = [
        npil.gaussian_blur,
        npil.box_blur,
        npil.unsharp_mask,
        npil.median_filter,
        npil.minimum_filter,
        npil.maximum_filter,
        npil.mode_filter,
        npil.auto_contrast,
        npil.equalize_histogram_bins,
        npil.enhance_contrast,
        npil.enhance_brightness,
        npil.enhance_sharpness
    ]

    image = np.random.random((100, 100)) * 255

    for f in functions:
        f(image)

def test_unsharp_mask_with_different_types():
    image = np.random.random((100, 100)) * 255

    npil.unsharp_mask(image, int(1), int(1), int(1))
    npil.unsharp_mask(image, float(1), int(1), int(1))
    npil.unsharp_mask(image, int(1), float(1), int(1))
    npil.unsharp_mask(image, int(1), int(1), float(1))

