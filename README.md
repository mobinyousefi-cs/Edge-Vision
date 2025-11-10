# Edge Detection using Computer Vision

> "Edges are where the information lives." – Computer Vision proverb

## Overview

This project implements a clean, extensible **edge detection pipeline** using **Python** and **OpenCV**. It provides:

- Reusable edge detector functions (Canny, Sobel, Laplacian)
- A simple **command‑line interface (CLI)** for batch processing images
- Utility functions for safe file I/O and visualization
- A small **synthetic dataset generator** so you can experiment without hunting for a dataset
- A standard, production‑ready Python project structure (PEP 621 / `pyproject.toml`)

The code is written to be:

- **Readable and well‑documented** (master’s level code quality)
- **Tested** with `pytest`
- **Ready for GitHub CI** (Ruff + Black + pytest)

Repository layout follows a standard `src/` pattern and is suitable for direct publishing to GitHub:

```text
edge-vision/
├── .editorconfig
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── scripts/
│   └── generate_sample_data.py
├── src/
│   └── edge_vision/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── edge_detectors.py
│       ├── io_utils.py
│       └── visualization.py
└── tests/
    ├── test_edge_detectors.py
    └── test_io_utils.py
```

---

## 1. Motivation: Why Edge Detection?

**Edge detection** is one of the foundational tasks in Computer Vision. Edges correspond to points in an image where the intensity changes sharply. They are crucial for:

- Object boundary detection
- Image segmentation
- Shape analysis and recognition
- Feature extraction for more complex CV tasks

In this project, we focus on classic gradient‑based edge detectors:

- **Sobel** – Estimates image gradients in x/y directions
- **Laplacian** – Second‑order derivatives, highlighting regions of rapid intensity change
- **Canny** – Multi‑stage, robust edge detector (noise reduction, gradient estimation, non‑max suppression, hysteresis thresholding)

---

## 2. Features

- 🔹 **Multiple edge detection algorithms** implemented in a unified way
- 🔹 **Batch processing** of entire folders (recursive or non‑recursive)
- 🔹 **Configurable thresholds and parameters** via CLI options
- 🔹 **Optional visualization** for quick inspection of results
- 🔹 **Synthetic dataset generator** (`scripts/generate_sample_data.py`) to produce simple geometric test images
- 🔹 **Unit tests** to validate core behavior

---

## 3. Installation

### 3.1. Clone the repository

```bash
git clone https://github.com/mobinyousefi-cs/edge-vision.git
cd edge-vision
```

### 3.2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# or
.venv\Scripts\activate        # Windows
```

### 3.3. Install the package

This project uses **`pyproject.toml`** (PEP 621).

```bash
pip install -e .
```

Alternatively, you can just install the dependencies directly:

```bash
pip install opencv-python numpy matplotlib
```

For development (linting, formatting, tests):

```bash
pip install -e .[dev]
```

---

## 4. Quick Start

### 4.1. Generate synthetic sample data (optional)

You don’t need an external dataset to start. Run:

```bash
python scripts/generate_sample_data.py
```

This will create images like circles, rectangles and lines in `data/raw/`. They are ideal for visualizing how each edge detector behaves.

### 4.2. Run edge detection from the CLI

The package exposes a console script named `edge-vision` via `pyproject.toml`.

Basic usage:

```bash
edge-vision \
  --input data/raw \
  --output data/edges \
  --method canny
```

Options:

```bash
edge-vision --help
```

Example commands:

```bash
# Canny edge detection with default thresholds
edge-vision --input data/raw --output data/edges --method canny

# Sobel edge detection (combined magnitude)
edge-vision --input data/raw --output data/edges --method sobel

# Laplacian edge detection
edge-vision --input data/raw --output data/edges --method laplacian

# Recursive traversal of subdirectories
edge-vision --input /path/to/images --output /path/to/edges --recursive

# Visualize the result for each image (blocks, for debugging / demos)
edge-vision --input data/raw --output data/edges --method canny --visualize
```

---

## 5. API Overview

All core functionality lives under the `edge_vision` package.

### 5.1. Edge detectors (`edge_vision.edge_detectors`)

```python
from edge_vision import edge_detectors
import cv2

image_bgr = cv2.imread("path/to/image.png")
edges_canny = edge_detectors.canny_edges(image_bgr, low_threshold=100, high_threshold=200)
edges_sobel = edge_detectors.sobel_edges(image_bgr)
edges_laplacian = edge_detectors.laplacian_edges(image_bgr)
```

Each function:

- Accepts a BGR or grayscale image (`numpy.ndarray`)
- Internally converts to grayscale if needed
- Returns a single‑channel uint8 edge map with the same spatial resolution as the input

### 5.2. I/O utilities (`edge_vision.io_utils`)

```python
from edge_vision.io_utils import list_images, load_image, save_image

paths = list_images("data/raw")
img = load_image(paths[0], as_gray=False)
save_image(img, "data/processed/example.png")
```

### 5.3. Visualization helpers (`edge_vision.visualization`)

```python
from edge_vision.visualization import show_side_by_side

show_side_by_side(original=image_bgr, processed=edges_canny,
                  titles=("Original", "Canny Edges"))
```

---

## 6. Configuration

`edge_vision.config` centralizes common configuration options:

- Default input / output directories
- Allowed image extensions
- Default Canny / Sobel / Laplacian parameters

You can either use these defaults or override them in your own scripts.

---

## 7. Testing

Run all tests with:

```bash
pytest
```

The test suite focuses on:

- Ensuring that edge detection functions preserve image shape and type
- Verifying correct I/O behavior (saving and loading images)

---

## 8. Continuous Integration (optional)

A minimal GitHub Actions workflow is included in `.github/workflows/ci.yml` (you can create it by following the same pattern as in your other projects). It typically:

1. Sets up Python
2. Installs the package and dev dependencies
3. Runs Ruff (lint), Black (format check) and pytest

You can adapt that workflow to match your global GitHub template.

---

## 9. Extending the Project

Here are some ideas if you want to push this further:

- Add **Scharr** and **Prewitt** filters
- Implement **automatic threshold selection** for Canny (e.g. using median of gradients)
- Integrate **non‑max suppression** and **hysteresis** for custom edge detectors
- Build a small **GUI** (e.g. with `tkinter` or `PyQt5`) to adjust thresholds interactively
- Combine edge maps with segmentation / object detection pipelines

Because the project uses a clean `src/` layout and a simple API, you can safely extend it without breaking the CLI or tests.

---

## 10. License

This project is released under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

---

## 11. Author

**Mobin Yousefi**  
GitHub: <https://github.com/mobinyousefi-cs>

Feel free to open issues or pull requests if you extend this project or find any problems.

