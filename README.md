# cardio

`cardio` is a simple web-based viewer for 3D and 4D ('cine') medical imaging data,
built primarily on [trame](https://github.com/kitware/trame),
[vtk](https://github.com/kitware/vtk), and
[itk](https://github.com/insightsoftwareconsortium/itk).  `cardio` is able to
render sequences of mesh files (e.g., `\*.obj` files), segmentation files (e.g.
`\*nii.gz` files with discrete labels) and volume renderings of grayscale images
(e.g. \*.nii.gz files with continuous values).  `cardio` is launched from the
commandline and may be configured either directly from the commandline, via a static
TOML configuration file, or a combination of the two.

## Quickstart

### Installation

```bash
$ cd /path/to/your/project
$ uv init
$ uv add cardio
$ . ./.venv/bin/activate
(project) cardio --version
cardio 2025.8.0
```

### Developing

Ensuring you have all required dependencies:

```bash
$ uv sync --all-extras
```

Pre-commit checklist:

```bash
$ isort .
$ ruff format
$ pytest -v
```
