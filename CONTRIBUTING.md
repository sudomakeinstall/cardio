# Contributing to Cardio

Contributions are welcome!  Here are the instructions for setting up a development build, as well as some useful links.

## Setting Up a Development Environment

```bash
git clone git@github.com:DVigneault/cardio.git
cd cardio
python -m venv .venv
. ./.venv/bin/activate
pip install --upgrade pip
pip install -e .
cardio --config ./examples/cfg-example.toml
```

## Useful Links for Developers

### Python Packaging

- https://packaging.python.org/en/latest/specifications/declaring-project-metadata/
- https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
- https://realpython.com/pypi-publish-python-package/

### Trame

- https://materialdesignicons.com/
- https://vimeo.com/761096621/af2287747f
- https://kitware.github.io/trame/docs/index.html
- https://trame.readthedocs.io/en/latest/index.html