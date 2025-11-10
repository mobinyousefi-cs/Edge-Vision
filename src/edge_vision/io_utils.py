#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: io_utils.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Utility functions for safe and convenient image I/O operations, including
recursive image discovery and robust read/write wrappers around OpenCV.

Usage:
from edge_vision.io_utils import list_images, load_image, save_image

Notes:
- All paths are handled using `pathlib.Path` for cross-platform compatibility.
- `load_image` can optionally return grayscale images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from .config import ALLOWED_EXTENSIONS


def list_images(directory: Path | str, recursive: bool = False) -> List[Path]:
    """Return a list of image file paths under the given directory.

    Parameters
    ----------
    directory:
        Directory to scan for images.
    recursive:
        If True, walk all subdirectories recursively.

    Returns
    -------
    list of Path
        Paths to image files with extensions listed in ALLOWED_EXTENSIONS.
    """

    base = Path(directory)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {base}")

    if recursive:
        candidates: Iterable[Path] = base.rglob("*")
    else:
        candidates = base.iterdir()

    images: List[Path] = [
        p for p in candidates if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    images.sort()
    return images


def load_image(path: Path | str, as_gray: bool = False) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Parameters
    ----------
    path:
        Path to the image file.
    as_gray:
        If True, return a single-channel grayscale image.

    Returns
    -------
    numpy.ndarray
        Loaded image in BGR (default) or grayscale format.
    """

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")

    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(p), flag)

    if img is None:
        raise ValueError(f"Failed to load image: {p}")

    return img


def save_image(image: np.ndarray, path: Path | str) -> None:
    """Save an image to disk, creating parent directories as needed.

    Parameters
    ----------
    image:
        Image array to save.
    path:
        Target path. Parent directories are created if they do not exist.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure image is a proper numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    success = cv2.imwrite(str(p), image)
    if not success:
        raise ValueError(f"Failed to save image to: {p}")
