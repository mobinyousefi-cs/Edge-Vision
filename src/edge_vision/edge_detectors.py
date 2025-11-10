#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: edge_detectors.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Implementation of classic edge detection algorithms (Canny, Sobel, Laplacian)
using OpenCV. All functions accept either BGR or grayscale images and return
single-channel uint8 edge maps.

Usage:
from edge_vision.edge_detectors import canny_edges, sobel_edges, laplacian_edges

Notes:
- Input images are converted to grayscale internally if needed.
- Returned edge maps preserve the spatial resolution of the input.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from . import config


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if it is BGR; otherwise return as-is.

    Parameters
    ----------
    image:
        Input image, either grayscale (H, W) or BGR (H, W, 3).

    Returns
    -------
    numpy.ndarray
        Grayscale image with shape (H, W).
    """

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Unsupported image shape for grayscale conversion: {shape}".format(shape=image.shape))


def canny_edges(
    image: np.ndarray,
    low_threshold: int | float | None = None,
    high_threshold: int | float | None = None,
    aperture_size: int | None = None,
    l2gradient: bool | None = None,
) -> np.ndarray:
    """Compute edges using the Canny detector.

    Parameters
    ----------
    image:
        Input image (BGR or grayscale).
    low_threshold:
        Lower hysteresis threshold. Defaults to `config.CANNY_LOW_THRESHOLD`.
    high_threshold:
        Upper hysteresis threshold. Defaults to `config.CANNY_HIGH_THRESHOLD`.
    aperture_size:
        Aperture size for the Sobel operator (odd integer). Defaults to
        `config.CANNY_APERTURE_SIZE`.
    l2gradient:
        If True, use a more accurate L2 norm. Defaults to
        `config.CANNY_L2GRADIENT`.

    Returns
    -------
    numpy.ndarray
        Binary edge map (uint8) with values 0 or 255.
    """

    gray = _to_grayscale(image)

    low = config.CANNY_LOW_THRESHOLD if low_threshold is None else low_threshold
    high = config.CANNY_HIGH_THRESHOLD if high_threshold is None else high_threshold
    aperture = config.CANNY_APERTURE_SIZE if aperture_size is None else aperture_size
    l2 = config.CANNY_L2GRADIENT if l2gradient is None else l2gradient

    edges = cv2.Canny(gray, threshold1=low, threshold2=high, apertureSize=aperture, L2gradient=l2)
    return edges


def sobel_edges(image: np.ndarray, ksize: int | None = None) -> np.ndarray:
    """Compute edges using Sobel gradients (approximate gradient magnitude).

    Parameters
    ----------
    image:
        Input image (BGR or grayscale).
    ksize:
        Size of the extended Sobel kernel; must be 1, 3, 5, or 7.
        Defaults to `config.SOBEL_KSIZE`.

    Returns
    -------
    numpy.ndarray
        Gradient magnitude edge map (uint8).
    """

    gray = _to_grayscale(image)
    k = config.SOBEL_KSIZE if ksize is None else ksize

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)

    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude)) if np.max(magnitude) > 0 else np.zeros_like(gray, dtype=np.uint8)
    return magnitude


def laplacian_edges(
    image: np.ndarray,
    ksize: int | None = None,
    scale: int | float | None = None,
    delta: int | float | None = None,
) -> np.ndarray:
    """Compute edges using the Laplacian operator.

    Parameters
    ----------
    image:
        Input image (BGR or grayscale).
    ksize:
        Aperture size used to compute the second-derivative filters.
        Defaults to `config.LAPLACIAN_KSIZE`.
    scale:
        Optional scale factor for the computed Laplacian values.
        Defaults to `config.LAPLACIAN_SCALE`.
    delta:
        Optional delta value added to the results prior to storing them.
        Defaults to `config.LAPLACIAN_DELTA`.

    Returns
    -------
    numpy.ndarray
        Edge map (uint8) derived from the absolute Laplacian response.
    """

    gray = _to_grayscale(image)

    k = config.LAPLACIAN_KSIZE if ksize is None else ksize
    s = config.LAPLACIAN_SCALE if scale is None else scale
    d = config.LAPLACIAN_DELTA if delta is None else delta

    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k, scale=s, delta=d)
    abs_lap = cv2.convertScaleAbs(lap)
    return abs_lap


EDGE_METHODS: dict[str, callable] = {
    "canny": canny_edges,
    "sobel": sobel_edges,
    "laplacian": laplacian_edges,
}


def available_methods() -> Tuple[str, ...]:
    """Return the tuple of available edge detection method names."""

    return tuple(sorted(EDGE_METHODS.keys()))
