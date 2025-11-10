#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Central configuration module for the `edge_vision` package. Defines default
paths, supported file extensions, and sensible default parameters for
edge detection algorithms.

Usage:
from edge_vision.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR

Notes:
- You can override these defaults from your own scripts or CLI options.
- Keep this file free of heavy imports to avoid side-effects.
"""

from __future__ import annotations

from pathlib import Path

# Base project directory (resolved at runtime from this file location)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories (used by CLI and sample data generator)
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "edges"

# Allowed image extensions for discovery
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Default parameters for edge detection algorithms
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200
CANNY_APERTURE_SIZE = 3
CANNY_L2GRADIENT = True

SOBEL_KSIZE = 3

LAPLACIAN_KSIZE = 3
LAPLACIAN_SCALE = 1
LAPLACIAN_DELTA = 0
