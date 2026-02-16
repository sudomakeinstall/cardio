#!/usr/bin/env python

# System
import os

# Third Party
import pydantic_settings as ps
import trame as tm
import trame.decorators
import vtk

from . import __version__
from .logic import Logic
from .scene import Scene
from .ui import UI


@tm.decorators.TrameApp()
class CardioApp:
    def __init__(self, server=None):
        self.server = tm.app.get_server(server, client_type="vue3")

        self.server.cli.add_argument(
            "--config", help="TOML configuration file.", dest="cfg_file", required=False
        )

        self.server.cli.add_argument(
            "--version", action="version", version=f"cardio {__version__}"
        )

        cli_settings = ps.CliSettingsSource(
            Scene, root_parser=self.server.cli, cli_parse_args=True
        )

        args, unknown = self.server.cli.parse_known_args()
        config_file = getattr(args, "cfg_file", None)

        Scene._cli_source = cli_settings
        Scene._config_file = config_file

        try:
            scene = Scene()
        finally:
            if hasattr(Scene, "_cli_source"):
                delattr(Scene, "_cli_source")
            if hasattr(Scene, "_config_file"):
                delattr(Scene, "_config_file")

        Logic(self.server, scene)
        UI(self.server, scene)


def main():
    if hasattr(vtk, "vtkEGLRenderWindow"):
        os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    app = CardioApp()
    app.server.start(open_browser=False)


if __name__ == "__main__":
    main()
