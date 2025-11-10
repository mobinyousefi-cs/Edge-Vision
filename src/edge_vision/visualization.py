#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: visualization.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Visualization helpers built on top of matplotlib. Provides convenience
functions to display original and processed images side by side.

Usage:
from edge_vision.visualization import show_side_by_side

Notes:
- Visualization is optional and mainly used for debugging and demos.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale image to RGB for matplotlib display.

    Parameters
    ----------
    image:
        Input image in BGR or grayscale format.

    Returns
    -------
    numpy.ndarray
        Image in RGB (H, W, 3) or grayscale retained.
    """

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        # OpenCV uses BGR; matplotlib expects RGB
        return image[:, :, ::-1]
    raise ValueError(f"Unsupported image shape for visualization: {image.shape}")


def show_side_by_side(
    original: np.ndarray,
    processed: np.ndarray,
    titles: Tuple[str, str] = ("Original", "Processed"),
) -> None:
    """Display original and processed images side by side.

    Parameters
    ----------
    original:
        Original input image (BGR or grayscale).
    processed:
        Processed image (often an edge map).
    titles:
        Tuple of titles for the two subplots.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    orig_rgb = _to_rgb(original)

    axes[0].imshow(orig_rgb, cmap=None if orig_rgb.ndim == 3 else "gray")
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    if processed.ndim == 2:
        axes[1].imshow(processed, cmap="gray")
    else:
        axes[1].imshow(_to_rgb(processed))

    axes[1].set_title(titles[1])
    axes[1].axis("off")

    fig.tight_layout()
    plt.show()


def show_grid(images: Iterable[np.ndarray], cols: int = 3, titles: Iterable[str] | None = None) -> None:
    """Display a list of images in a grid.

    Parameters
    ----------
    images:
        Iterable of images (BGR or grayscale).
    cols:
        Number of columns in the grid layout.
    titles:
        Optional iterable of titles for each image.
    """

    images = list(images)
    n = len(images)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    if titles is None:
        titles_list = [""] * n
    else:
        titles_list = list(titles)
        if len(titles_list) != n:
            raise ValueError("Number of titles must match number of images")

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for idx, (img, title) in enumerate(zip(images, titles_list)):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        img_rgb = _to_rgb(img)
        ax.imshow(img_rgb, cmap=None if img_rgb.ndim == 3 else "gray")
        ax.set_title(title)
        ax.axis("off")

    # Hide any unused axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.tight_layout()
    plt.show()
