# MED-GRIP MScan Annotator

A lightweight, callable desktop tool for annotating retinal layer boundaries in **OCT M-scan** (`.npy`) data. Users place sparse seed points on the image, fit a smooth B-spline through them, and optionally snap the curve to the dominant vertical gradient to follow the true tissue boundary.

---

## Features

- **PyQt6 GUI** — native High-DPI rendering on Windows/macOS/Linux
- **Automatic spline updates** — a smooth interpolating spline is redrawn after every seed edit; no separate fit step required
- **Spline-guided fine-tuning** — refinement favors the first plausible dark-to-bright edge at the retinal top rim and stays within a hard +/-5 pixel band around the spline
- **Smoother retinal boundary detection** — gradient maps and final boundaries are smoothed across neighboring A-scans to suppress single-column outliers and jitter
- **Black-background aware refinement** — if the top layer fades out near scan ends, fine-tuning keeps the spline instead of snapping down to a deeper layer
- **Multi-class regional classification** — press `1`, `2`, `3`, `4` to classify seed regions; each seed stores its class, and boundaries are colored per-region (dark green, light green, orange, red), with class labels exported to a combined `.npy` file
- **Per-class PNG overlay** — saved PNG shows colored boundary regions matching the assigned classes
- **Experiment-friendly output layout** — saved annotations, class labels, and overlay PNGs go to an `annotated/` subfolder
- **Fast review workflow** — saving or flagging a scan automatically advances to the next source file and can continue into the next sibling experiment folder
- **Clean file list** — only source `*.npy` scans appear in the dropdown; generated `*_annotations.npy` files are excluded
- **Improved seed editing** — right-click removes the latest seed, right-clicking a seed removes that exact seed, and duplicate X-columns are ignored
- **NaN window tools** — create multiple exclusion windows with automatic color cycling; excluded columns are exported with a sentinel value
- **Too-hard triage** — one action marks the OCT as all-NaN (too hard to label) and saves it to the regular `annotated/` folder, then advances to the next file
- **Keyboard shortcuts** — `1/2/3/4` classify seed classes (dark green/light green/orange/red), `A` fine-tunes, `D` saves, `F` marks too hard
- **Scroll-to-zoom** — inspect individual A-scans at any magnification
- **Pan and reset controls** — hold left + right mouse buttons to pan; middle-click resets zoom to fit
- **CLI entry point** — `oct-annotate [<folder>]` launches the app directly from the terminal
- **CI-ready** — Azure Pipelines YAML included for automated testing and wheel builds

---

## Project Structure

```
oct_annotator/
├── src/
│   └── oct_annotator/
│       ├── __init__.py
│       ├── main.py        # QMainWindow entry point
│       ├── engine.py      # Spline fitting & gradient refinement
│       └── viewer.py      # Custom QGraphicsView canvas
├── tests/
│   └── test_engine.py     # Unit, integration & performance tests
├── pyproject.toml
├── requirements.txt
└── .azure-pipelines.yml
```

---

## Getting Started

### Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.9 |
| PyQt6 | ≥ 6.5, < 6.11 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |
| opencv-python | ≥ 4.8 |

### Installation

0. **Verify Python is available**:

   ```bash
   python --version
   ```

   If this command fails on Windows, install Python 3.10+ from python.org and make sure the installer option to add Python to PATH is enabled.

1. **Create a virtual environment using Python 3.9+** (Python 3.12 recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

   > If your network intercepts TLS/SSL traffic and pip fails with certificate errors, use your organization's Python package mirror, or as a last resort add  
   > `--trusted-host pypi.org --trusted-host files.pythonhosted.org` to pip commands.

2. **Install dependencies**:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install the package** (editable mode for development):

   ```bash
   pip install -e .
   ```

---

## Usage

### Launch from the command line

```bash
oct-annotate                      # opens a folder-picker dialog
oct-annotate "C:/path/to/scans"   # opens directly with the given folder
```

Alternative launch options:

```bash
# Run as module from the project root
python -m oct_annotator.main "C:/path/to/scans"
```

On Windows, you can also double-click `start_oct_annotate.bat` from the project root.

### Annotation workflow

1. **Open folder** — select a directory containing source `*.npy` M-scan files  
   *(expected shape: `rows × columns`, float64, values in [0, 1])*
2. **Review the file list** — the dropdown shows only source scans and hides generated `*_annotations.npy` files
3. **Place seeds and classify** — left-click on the image to mark boundary control points (red dots); press `1` (dark green), `2` (light green), `3` (orange), or `4` (red) to set the classification for subsequently placed seeds; the spline updates automatically once at least two seeds exist
4. **Edit seeds quickly** — right-click removes the most recent seed, or right-click directly on an existing seed to remove that exact point
5. **Fine-tune** — press `A` or click **Fine-tune** to refine the spline into a smooth boundary that follows the earliest plausible top-layer rim near the spline while staying within +/-5 pixels of it (darker green curve for refined vs. bright green for original spline)
6. **Mark excluded regions** — add one or more NaN windows for columns that should export as the NaN sentinel value `65535`
7. **Save** — press `D` or click **Save** to write:
   - `annotated/<source_stem>_annotations.npy` — shape `(columns, 2)` where column 0 = boundary indices (float32) and column 1 = class labels (1.0, 2.0, 3.0, 4.0, or NaN)
   - `annotated/<source_stem>_annotations.png` — overlay with colored boundaries (dark green, light green, orange, red) matching assigned classes
   
   The next file is loaded automatically after saving.
8. **Flag difficult scans** — press `F` or click **Too Hard** to mark the entire OCT as all-NaN (indicating difficulty in labeling), save it to `annotated/`, and continue to the next file

If the current folder is exhausted, the app attempts to open the next sibling experiment folder that contains source scans.

### Fine-tuning behavior

- Fine-tuning searches only inside a local +/-5 pixel window around the current spline, so refinement stays tied to the user-guided boundary instead of drifting across the scan.
- Only positive vertical gradients are considered, which biases the result toward the dark-to-bright transition expected at the top retinal layer.
- Candidates must also have visible post-edge brightness, which helps reject faint early edges that remain almost black.
- When multiple nearby candidates are plausible, the algorithm prefers the earliest sufficiently strong candidate at or above the spline before considering deeper fallback edges.
- The final boundary is median-filtered and then smoothed across columns to reduce isolated jumps while preserving the overall layer shape.
- If no plausible bright-tissue candidate exists in a column, the spline position is kept there; this is important when the retinal surface disappears into black background at the scan edges.

### Interaction details

- **Left-click** — place a seed point
- **Right-click** — remove the last seed, or remove the clicked seed directly
- **1 / 2 / 3** — set the classification for subsequently placed seeds (green / yellow / red)
- **A** — fine-tune the boundary
- **D** — save the annotation
- **F** — mark as too hard (all-NaN)
- **Left + right mouse buttons** — pan the image
- **Middle-click** — reset zoom to fit
- **Duplicate seed X positions** — ignored silently to keep the spline well-formed

### Scroll / zoom

- **Scroll wheel** — zoom in/out centred on the cursor
- The image auto-fits the window on load and resize
- NaN windows can be repositioned freely, including all the way to the left edge

---

## Build and Test

```bash
# Run the full test suite
pytest tests/ -v

# Build a distributable wheel
pip install build
python -m build
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'PyQt6'`

- Confirm your virtual environment is active.
- Reinstall project dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

### `ImportError: DLL load failed while importing QtCore` (Windows)

- Use the pinned PyQt6 range from this project (`PyQt6>=6.5,<6.11`).
- Reinstall to ensure matching wheels are present:

```bash
pip install --force-reinstall "PyQt6>=6.5,<6.11"
```

### `ModuleNotFoundError: No module named 'cv2'`

- Install OpenCV in the same environment that runs the app:

```bash
pip install opencv-python
```

### Test coverage

| Class | Tests |
|---|---|
| `TestFitSpline` | correct width, integer dtype, linear 2-point case, single-point error, in-range values |
| `TestRefineBoundary` | edge snapping, first-layer preference, stronger-deeper-edge rejection, outlier smoothing, dim-edge rejection, black-background fallback, +/-5 spline bound, output dtype, output shape |
| `TestIntegrationSaveLoad` | full annotate→save→reload round-trip, filename convention |
| `TestPerformance` | refinement < 100 ms on a 1024 × 1000 scan |

---

## CI / CD (Azure Pipelines)

The included `.azure-pipelines.yml` triggers on pushes to `main` and:

1. Provisions Python 3.10 on `ubuntu-latest`
2. Installs all dependencies and `pytest`
3. Runs the test suite
4. Builds the wheel with `python -m build`

---

## Contributing

1. Branch from `main` using the convention `feature/<short-description>` or `fix/<short-description>`
2. Add or update tests for any logic changes in `engine.py`
3. Ensure `pytest tests/ -v` passes locally before opening a pull request
4. Open a pull request against `main` and request at least one review before merge
