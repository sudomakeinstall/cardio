# Cardio

`cardio` is a simple web-based viewer for 4D ('cine') medical imaging data.  `cardio` is able to render sequences of mesh files (e.g., `\*.obj` files) and volume renderings of image files (e.g. \*.nii.gz files).  `cardio` is launched from the commandline and configured using a TOML file.

## Quickstart

```bash
$ python -m venv .venv
$ . ./.venv/bin/activate
(.venv) pip install cardio
(.venv) cardio --config ./examples/cfg-example.toml

App running at:
 - Local:   http://localhost:8080/
 - Network: http://127.0.0.1:8080/
```
