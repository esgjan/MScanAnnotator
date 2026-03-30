# MED-GRIP MScan Annotator

A lightweight, callable desktop tool for annotating retinal layer boundaries in **OCT M-scan** (`.npy`) data. Users place sparse seed points on the image, fit a smooth B-spline through them, and optionally snap the curve to the dominant vertical gradient to follow the true tissue boundary.

---

## Features

- **PyQt6 GUI** — native High-DPI rendering on Windows/macOS/Linux
- **B-spline interpolation** — `scipy.interpolate.splprep` gives a smooth curve through any number of seed clicks
- **Gradient fine-tuning** — each A-scan column snaps to the local intensity-gradient peak within a configurable ±δ window
- **Scroll-to-zoom** — inspect individual A-scans at any magnification
- **One-click save** — exports a `(width × 1) uint16` annotation array alongside the source file
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
| PyQt6 | ≥ 6.5 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |

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

1. **Open folder** — select a directory containing `*.npy` M-scan files  
   *(expected shape: `rows × columns`, float64, values in [0, 1])*
2. **Place seeds** — left-click on the image to mark boundary control points (red dots)
3. **Fit Spline** — fits a cubic B-spline through the seeds (green curve)
4. **Fine-tune** — snaps each A-scan point to the nearest gradient peak within ±δ pixels (blue curve); adjust δ with the spin-box
5. **Save** — writes `<source_stem>_annotations.npy` (`shape: columns × 1, dtype: uint16`) next to the source file

### Scroll / zoom

- **Scroll wheel** — zoom in/out centred on the cursor
- The image auto-fits the window on load and resize

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
| `TestRefineBoundary` | edge snapping, output dtype, output shape |
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
