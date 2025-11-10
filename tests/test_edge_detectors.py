#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: test_edge_detectors.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Basic unit tests for the edge detection functions. These tests focus on
shape / dtype invariants and simple sanity checks using synthetic images.

Usage:
pytest tests/test_edge_detectors.py

Notes:
- The goal is not to validate OpenCV internals but to ensure our wrappers
  behave as expected.
"""

from __future__ import annotations

import numpy as np

from edge_vision.edge_detectors import (
    EDGE_METHODS,
    available_methods,
    canny_edges,
    laplacian_edges,
    sobel_edges,
)


def _synthetic_image(width: int = 128, height: int = 128) -> np.ndarray:
    """Create a simple synthetic image with a gradient and a rectangle."""

    x = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(x, (height, 1))
    image = np.stack([gradient, gradient, gradient], axis=-1)
    image[32:96, 32:96, :] = 255
    return image


def test_available_methods_contains_expected_names() -> None:
    methods = available_methods()
    assert "canny" in methods
    assert "sobel" in methods
    assert "laplacian" in methods


def test_canny_edges_output_shape_and_dtype() -> None:
    img = _synthetic_image()
    edges = canny_edges(img)
    assert edges.shape[:2] == img.shape[:2]
    assert edges.dtype == np.uint8


def test_sobel_edges_output_shape_and_dtype() -> None:
    img = _synthetic_image()
    edges = sobel_edges(img)
    assert edges.shape[:2] == img.shape[:2]
    assert edges.dtype == np.uint8


def test_laplacian_edges_output_shape_and_dtype() -> None:
    img = _synthetic_image()
    edges = laplacian_edges(img)
    assert edges.shape[:2] == img.shape[:2]
    assert edges.dtype == np.uint8


def test_edge_methods_registry_matches_functions() -> None:
    img = _synthetic_image()
    for name, func in EDGE_METHODS.items():
        edges = func(img)
        assert edges.shape[:2] == img.shape[:2]
        assert edges.dtype == np.uint8
