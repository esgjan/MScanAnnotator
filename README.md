# MED-GRIP MScan Annotator

A lightweight, callable desktop tool for annotating retinal layer boundaries in **OCT M-scan** (`.npy`) data. Users place sparse seed points on the image, fit a smooth B-spline through them, and optionally snap the curve to the dominant vertical gradient to follow the true tissue boundary.

---

## Features

- **PyQt6 GUI** — native High-DPI rendering on Windows/macOS/Linux
- **Automatic spline updates** — a smooth interpolating spline is redrawn after every seed edit; no separate fit step required
- **Gradient fine-tuning** — each A-scan column snaps to the local intensity-gradient peak within a configurable ±δ window
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

On Windows, the included launcher scripts start the app with the default scan directory at `D:\iiOCT_data\npy`.

### Windows batch launcher

`start_oct_annotate.bat` is a simple Windows launcher for users who want to start the app by double-clicking a file instead of opening a terminal.

Before another user runs it, update these lines inside the batch file:

- `PROJECT_DIR=...` should point to that user's local `MScanAnnotator` folder
- `.venv\Scripts\python.exe` must exist inside that project folder
- `src\oct_annotator\main.py` must exist inside that project folder
- `DEFAULT_SCAN_DIR=...` should point to the default scan folder that should open at startup

Example:

```bat
@echo off
set "PROJECT_DIR=C:\Users\Alice\git\MScanAnnotator"
set "VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "APP_ENTRY=%PROJECT_DIR%\src\oct_annotator\main.py"
cd /d "%PROJECT_DIR%"
"%VENV_PYTHON%" "%APP_ENTRY%" "D:\iiOCT_data\npy"
```

For the shared project setup, this is the recommended Windows entry point. Users only need the project folder, the project `.venv`, and the batch file above.

If a user does not want to edit the batch file, they can skip it and run the app directly from a terminal instead:

```bash
oct-annotate
oct-annotate "D:/iiOCT_data/npy"
```

### Annotation workflow

1. **Open folder** — select a directory containing source `*.npy` M-scan files  
   *(expected shape: `rows × columns`, float64, values in [0, 1])*
2. **Review the file list** — the dropdown shows only source scans and hides generated `*_annotations.npy` files
3. **Place seeds** — left-click on the image to mark boundary control points (red dots); the spline updates automatically once at least two seeds exist
4. **Edit seeds quickly** — right-click removes the most recent seed, or right-click directly on an existing seed to remove that exact point
5. **Fine-tune** — press `A` or click **Fine-tune** to snap each A-scan point to the nearest gradient peak (blue curve)
6. **Mark excluded regions** — add one or more NaN windows for columns that should export as the NaN sentinel value `65535`
7. **Save** — press `D` or click **Save** to write:
   - `annotated/<source_stem>_annotations.npy`
   - `annotated/<source_stem>_annotations.png`
   The next file is loaded automatically after saving.
8. **Flag difficult scans** — press `F` or click **Too Hard** to copy the original scan into `2hard2label/`, generate a PNG preview, and continue to the next file

If the current folder is exhausted, the app attempts to open the next sibling experiment folder that contains source scans.

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
