#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: cli.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Command-line interface (CLI) entry point for batch edge detection. This module
wires together I/O utilities and edge detectors to process images from an
input directory and save edge maps to an output directory.

Usage:
python -m edge_vision.cli --input data/raw --output data/edges --method canny
# or, after installation
edge-vision --input data/raw --output data/edges --method sobel

Notes:
- The CLI is intentionally simple and dependency-free (only stdlib + package).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import cv2

from . import config
from .edge_detectors import EDGE_METHODS, available_methods
from .io_utils import list_images, load_image, save_image
from .visualization import show_side_by_side


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch edge detection using OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=config.DEFAULT_INPUT_DIR,
        help="Input directory containing images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Output directory to store edge maps.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=available_methods(),
        default="canny",
        help="Edge detection method to apply.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show original and edge images side by side (for debugging).",
    )
    parser.add_argument(
        "--canny-low",
        type=float,
        default=None,
        help="Override Canny lower threshold.",
    )
    parser.add_argument(
        "--canny-high",
        type=float,
        default=None,
        help="Override Canny upper threshold.",
    )

    return parser.parse_args()


def _process_image(
    img_path: Path,
    method: str,
    func: Callable,
    output_root: Path,
    visualize: bool,
    canny_low: float | None,
    canny_high: float | None,
) -> None:
    image = load_image(img_path, as_gray=False)

    if method == "canny":
        edges = func(image, low_threshold=canny_low, high_threshold=canny_high)
    else:
        edges = func(image)

    # Construct output path mirroring the input directory structure
    relative = img_path.name
    output_path = output_root / relative

    # Force grayscale single-channel images for saving when needed
    if edges.ndim == 2:
        save_image(edges, output_path)
    else:
        # If method returns multi-channel, convert to grayscale for saving
        gray_edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        save_image(gray_edges, output_path)

    if visualize:
        show_side_by_side(original=image, processed=edges, titles=("Original", f"{method.title()} Edges"))


def main() -> None:
    args = _parse_args()

    method = args.method.lower()
    if method not in EDGE_METHODS:
        raise SystemExit(f"Unsupported method: {method}. Available: {available_methods()}")

    edge_func = EDGE_METHODS[method]

    images = list_images(args.input, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No images found in: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        _process_image(
            img_path=img_path,
            method=method,
            func=edge_func,
            output_root=args.output,
            visualize=bool(args.visualize),
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
