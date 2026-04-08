# MED-GRIP MScan Annotator

A lightweight, callable desktop tool for annotating retinal layer boundaries in **OCT M-scan** (`.npy`) data. Users place sparse seed points on the image, fit a smooth B-spline through them, and optionally snap the curve to the dominant vertical gradient to follow the true tissue boundary.

---

## Features

- **PyQt6 GUI** — native High-DPI rendering on Windows/macOS/Linux
- **Automatic spline updates** — a smooth interpolating spline is redrawn after every seed edit; no separate fit step required
- **Spline-guided fine-tuning** — refinement favors the first strong dark-to-bright edge near the spline instead of blindly taking the strongest deeper peak
- **Smoother retinal boundary detection** — gradient maps and final boundaries are smoothed across neighboring A-scans to suppress single-column outliers and jitter
- **Black-background aware refinement** — if the top layer fades out near scan ends, fine-tuning keeps the spline instead of snapping down to a deeper layer
- **Experiment-friendly output layout** — saved annotations and overlay PNGs go to an `annotated/` subfolder, while flagged scans go to `2hard2label/`
- **Fast review workflow** — saving or flagging a scan automatically advances to the next source file and can continue into the next sibling experiment folder
- **Clean file list** — only source `*.npy` scans appear in the dropdown; generated `*_annotations.npy` files are excluded
- **Improved seed editing** — right-click removes the latest seed, right-clicking a seed removes that exact seed, and duplicate X-columns are ignored
- **NaN window tools** — create multiple exclusion windows with automatic color cycling; excluded columns are exported with a sentinel value
- **Too-hard triage** — one action copies the source scan to `2hard2label/`, writes a PNG preview, and advances to the next file
- **Keyboard shortcuts** — `A` fine-tunes, `D` saves, `F` marks the scan as too hard
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

1. **Create a virtual environment using Python 3.9+** (Python 3.12 recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

   > On machines where access to PyPI requires bypassing a corporate SSL proxy, add  
   > `--trusted-host pypi.org --trusted-host files.pythonhosted.org` to every `pip` call.

2. **Install dependencies**:

   ```bash
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

### Annotation workflow

1. **Open folder** — select a directory containing source `*.npy` M-scan files  
   *(expected shape: `rows × columns`, float64, values in [0, 1])*
2. **Review the file list** — the dropdown shows only source scans and hides generated `*_annotations.npy` files
3. **Place seeds** — left-click on the image to mark boundary control points (red dots); the spline updates automatically once at least two seeds exist
4. **Edit seeds quickly** — right-click removes the most recent seed, or right-click directly on an existing seed to remove that exact point
5. **Fine-tune** — press `A` or click **Fine-tune** to refine the spline into a smooth boundary that follows the earliest plausible top-layer edge near the spline (blue curve)
6. **Mark excluded regions** — add one or more NaN windows for columns that should export as the NaN sentinel value `65535`
7. **Save** — press `D` or click **Save** to write:
   - `annotated/<source_stem>_annotations.npy`
   - `annotated/<source_stem>_annotations.png`
   The next file is loaded automatically after saving.
8. **Flag difficult scans** — press `F` or click **Too Hard** to copy the original scan into `2hard2label/`, generate a PNG preview, and continue to the next file

If the current folder is exhausted, the app attempts to open the next sibling experiment folder that contains source scans.

### Fine-tuning behavior

- Fine-tuning searches in a local window around the current spline, so refinement stays tied to the user-guided boundary instead of drifting across the scan.
- Only positive vertical gradients are considered, which biases the result toward the dark-to-bright transition expected at the top retinal layer.
- Candidates must also have visible post-edge brightness, which helps reject faint early edges that remain almost black.
- When multiple nearby candidates are plausible, the algorithm prefers the earliest sufficiently strong one, which improves first-layer detection.
- The final boundary is median-filtered and then smoothed across columns to reduce isolated jumps while preserving the overall layer shape.
- If no plausible bright-tissue candidate exists in a column, the spline position is kept there; this is important when the retinal surface disappears into black background at the scan edges.

### Interaction details

- **Left-click** — place a seed point
- **Right-click** — remove the last seed, or remove the clicked seed directly
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

### Test coverage

| Class | Tests |
|---|---|
| `TestFitSpline` | correct width, integer dtype, linear 2-point case, single-point error, in-range values |
| `TestRefineBoundary` | edge snapping, first-layer preference, outlier smoothing, dim-edge rejection, black-background fallback, output dtype, output shape |
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
4. Open a PR against `main` in Azure DevOps — at least one reviewer approval is required before merge
