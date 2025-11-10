#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Edge Detection using Computer Vision
File: generate_sample_data.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Generate a small synthetic dataset of simple geometric shapes for testing
edge detection algorithms. This script avoids external dependencies and
network access by programmatically creating images.

Usage:
python scripts/generate_sample_data.py

Notes:
- Images are saved under `data/raw/` by default (see config.DEFAULT_INPUT_DIR).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from edge_vision.config import DEFAULT_INPUT_DIR


def _blank_canvas(width: int = 512, height: int = 512) -> np.ndarray:
    """Create a blank white canvas."""

    return np.full((height, width, 3), 255, dtype=np.uint8)


def generate_shapes_dataset(output_dir: Path) -> None:
    """Generate synthetic images with basic geometric shapes.

    Parameters
    ----------
    output_dir:
        Directory where generated images will be stored.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rectangles
    img_rect = _blank_canvas()
    cv2.rectangle(img_rect, (50, 50), (450, 200), (0, 0, 0), thickness=3)
    cv2.rectangle(img_rect, (100, 250), (400, 450), (0, 0, 255), thickness=-1)
    cv2.imwrite(str(output_dir / "rectangles.png"), img_rect)

    # 2. Circles
    img_circ = _blank_canvas()
    cv2.circle(img_circ, (256, 256), 150, (0, 0, 0), thickness=3)
    cv2.circle(img_circ, (256, 256), 75, (255, 0, 0), thickness=-1)
    cv2.imwrite(str(output_dir / "circles.png"), img_circ)

    # 3. Lines and grid
    img_lines = _blank_canvas()
    for x in range(50, 512, 50):
        cv2.line(img_lines, (x, 50), (x, 462), (0, 0, 0), thickness=1)
    for y in range(50, 512, 50):
        cv2.line(img_lines, (50, y), (462, y), (0, 0, 0), thickness=1)
    cv2.imwrite(str(output_dir / "grid.png"), img_lines)

    # 4. Text
    img_text = _blank_canvas()
    cv2.putText(
        img_text,
        "Edge Vision",
        (40, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "text.png"), img_text)

    # 5. Mixed shapes
    img_mixed = _blank_canvas()
    cv2.rectangle(img_mixed, (30, 30), (200, 200), (255, 0, 0), thickness=4)
    cv2.circle(img_mixed, (350, 150), 80, (0, 255, 0), thickness=-1)
    pts = np.array([[100, 300], [200, 450], [50, 450]], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_mixed, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.imwrite(str(output_dir / "mixed.png"), img_mixed)


def main() -> None:
    print(f"Generating synthetic dataset under: {DEFAULT_INPUT_DIR}")
    generate_shapes_dataset(DEFAULT_INPUT_DIR)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
