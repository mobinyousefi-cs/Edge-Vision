#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: test_io_utils.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Basic tests for image I/O utilities. Ensures images can be saved and
reloaded correctly and that image discovery behaves as expected.

Usage:
pytest tests/test_io_utils.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edge_vision.io_utils import list_images, load_image, save_image


@pytest.fixture()
def tmp_img_dir(tmp_path: Path) -> Path:
    return tmp_path / "images"


def test_save_and_load_image_roundtrip(tmp_img_dir: Path) -> None:
    tmp_img_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # red channel

    img_path = tmp_img_dir / "test.png"
    save_image(img, img_path)

    loaded = load_image(img_path, as_gray=False)
    assert loaded.shape == img.shape
    assert loaded.dtype == img.dtype


def test_list_images_finds_saved_image(tmp_img_dir: Path) -> None:
    tmp_img_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img_path = tmp_img_dir / "a.png"
    save_image(img, img_path)

    images = list_images(tmp_img_dir, recursive=False)
    assert len(images) == 1
    assert images[0].name == "a.png"
