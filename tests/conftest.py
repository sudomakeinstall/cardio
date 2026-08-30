"""Fixtures shared across the suite."""

# System
import pathlib as pl

# Third Party
import pytest
import tomlkit as tk

ASSETS = pl.Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def asset():
    """Read one of the TOML fixtures in tests/assets, by file name."""

    def read(name: str):
        with (ASSETS / name).open("rt", encoding="utf-8") as fp:
            return tk.load(fp)

    return read
