#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: __init__.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Package initialization for the `edge_vision` module. Exposes high-level APIs
for edge detection, I/O utilities, and visualization.

Usage:
from edge_vision import edge_detectors, io_utils, visualization

Notes:
- This file also defines the package-level version string.
"""

from __future__ import annotations

from . import edge_detectors, io_utils, visualization  # noqa: F401

__all__ = ["edge_detectors", "io_utils", "visualization"]
__version__ = "0.1.0"
