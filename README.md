# ARAP Pelt UV Unwrap

A standalone Python tool for UV unwrapping 3D `.obj` meshes using **As-Rigid-As-Possible (ARAP)** parameterization. Built with Tkinter, it provides a full GUI with a real-time 3D OpenGL viewport and a 2D UV layout preview - no DCC application required.

Optimized for **rocks, cliffs, and natural environment assets** where uniform texel density across organic, high-curvature surfaces matters most.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

![Screenshot](screenshot.png)

## Features

- **ARAP Pelt Unwrap** - single-island unwrap with free-boundary ARAP iterations, the gold standard for minimizing stretch distortion
- **Automatic seam placement** - curvature-weighted geodesic seams that follow crevices and sharp edges (ideal for rocks/cliffs)
- **Degenerate face filtering** - quality threshold parameter to exclude broken/sliver triangles from UV computation (great for photogrammetry meshes with noisy geometry)
- **Robust SVD handling** - per-triangle fallback for failed SVD decompositions; NaN/Inf sanitization after ARAP; automatic PCA fallback for collapsed UVs
- **Auto-oriented PCA projection** - when ARAP falls back to planar projection, face normals are used to project from the correct side of the mesh
- **Flip controls** - Flip U, Flip V buttons and a Flip Projection checkbox for manual orientation correction
- **Real-time 3D viewport** - OpenGL-powered orbit/pan/zoom with checker texture overlay to visualize UV density
- **2D UV layout** - color-coded island view with grid overlay, pan and zoom
- **Log panel** - built-in log window with color-coded INFO/WARNING/ERROR messages showing every pipeline step
- **Preserves original normals** - export only adds UV data, never overwrites your vertex normals
- **UV image export** - export the UV wireframe layout as a PNG (2048×2048)
- **Pure Python** - no compiled extensions, runs anywhere Python + pip works

## Use Cases

- **Game environment art** - unwrap rock, cliff, and terrain meshes for tiling textures
- **Photogrammetry cleanup** - re-unwrap scanned meshes that have poor or no UVs; filter degenerate geometry with the quality threshold
- **Batch pipeline integration** - `pelt_unwrap()` can be called directly from scripts without the GUI
- **Learning tool** - study ARAP parameterization with a readable, well-commented implementation

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy
- scipy
- PyOpenGL
- pyopengltk
- Pillow

## Usage

### GUI

```bash
python main.py
```

1. **Open** an `.obj` mesh (`Ctrl+O`)
2. Adjust parameters:
   - **ARAP Iterations** (default 50 - higher = more even UVs)
   - **Island Margin** (padding in UV space)
   - **Quality Threshold** (0 = keep all faces; increase to 0.01–0.1 to filter degenerate triangles)
3. Check **Flip Projection** if the mesh is being projected from the wrong side
4. Click **Unwrap**
5. Use **Flip U** / **Flip V** to mirror the island if needed
6. Inspect the result with the **checker texture** toggle and adjust the checker **Size**
7. Check the **Log** panel at the bottom for warnings about degenerate geometry
8. **Save** the unwrapped `.obj` (`Ctrl+S`) or **Export UV Image** as PNG

### Script / Pipeline

```python
from mesh import load_obj, save_obj
from uv_algorithms import pelt_unwrap

mesh = load_obj("rock.obj")
pelt_unwrap(mesh, arap_iterations=50, island_margin=0.01)
save_obj("rock_uv.obj", mesh)

# For meshes with degenerate geometry (e.g. photogrammetry):
pelt_unwrap(mesh, arap_iterations=50, quality_threshold=0.05)

# Project from the opposite side:
pelt_unwrap(mesh, flip_projection=True)
```

## Algorithm Pipeline

1. **Quality filter** *(optional)* - compute per-face quality scores (area vs. equilateral ideal) and exclude faces below the threshold
2. **Build local mesh** - re-index vertices for the unwrap solver
3. **Auto-seam** (closed meshes) - find a geodesic cycle using curvature-weighted Dijkstra so seams follow natural crevices rather than cutting across flat surfaces
4. **Initial flattening** - cotangent-Laplacian harmonic map with boundary vertices pinned to a circle
5. **ARAP iterations** - local/global alternation with:
   - Vectorized per-triangle SVD for best-fit rotations (local step)
   - Per-triangle SVD fallback for singular covariance matrices
   - Pre-factored sparse Cholesky solve (global step)
   - Free boundary - the boundary is not pinned, allowing the solver to find the lowest-distortion layout
6. **Sanitize** - replace NaN/Inf values; detect collapsed UVs and fall back to auto-oriented PCA projection
7. **Normalize** - fit UVs into `[0, 1]` space with configurable margin
8. **Restore filtered faces** - re-add excluded faces with collapsed UVs at island centre

## Parameters

| Parameter | Range | Default | Description |
|---|---|---|---|
| `arap_iterations` | 1–200 | 50 | Number of ARAP local/global iterations. More = lower distortion, diminishing returns past ~80. |
| `island_margin` | 0.0–0.05 | 0.01 | Padding around the UV island in normalized UV space. |
| `quality_threshold` | 0.0–0.5 | 0.0 | Faces with quality below this value are excluded from UV computation. 0 = keep all. Increase to 0.01–0.1 for meshes with degenerate triangles (common in photogrammetry). |
| `flip_projection` | - | off | Project from the opposite side of the mesh (checkbox in GUI). |

## Project Structure

```
├── main.py             # Tkinter GUI application (log panel, controls, 3D/UV views)
├── mesh.py             # Mesh data structure and OBJ I/O
├── uv_algorithms.py    # ARAP Pelt unwrap algorithm + quality filtering
├── viewer3d.py         # OpenGL 3D viewport widget
└── requirements.txt    # Python dependencies
```

## Academic References

The algorithm is an original implementation based on these published methods:

- **Olga Sorkine & Marc Alexa**, *"As-Rigid-As-Possible Surface Modeling"*, Symposium on Geometry Processing (SGP), 2007.
  - The core ARAP formulation: local/global alternation with cotangent weights and per-triangle SVD rotations.

- **Takeo Igarashi, Tomer Moscovich, John F. Hughes**, *"As-Rigid-As-Possible Shape Manipulation"*, ACM Transactions on Graphics, 2005.
  - Earlier formulation of rigid-as-possible deformation.

- **Bruno Lévy, Sylvain Petitjean, Nicolas Ray & Jérome Maillot**, *"Least Squares Conformal Maps for Automatic Texture Atlas Generation"*, ACM SIGGRAPH, 2002.
  - Cotangent-Laplacian formulation used for the initial harmonic map.

## License

MIT License - see [LICENSE](LICENSE) for details.
