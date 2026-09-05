#!/usr/bin/env python

# System
import os

# Third Party
import pydantic_settings as ps
import trame as tm
import trame.decorators
import vtk

from . import __version__
from .scene import Scene
from .session import Session
from .ui import UI


def scene_from_command_line(cli) -> Scene:
    """The scene the command line asks for, config file included."""
    cli.add_argument(
        "--config", help="TOML configuration file.", dest="cfg_file", required=False
    )
    cli.add_argument("--version", action="version", version=f"{__version__}")

    cli_source = ps.CliSettingsSource(Scene, root_parser=cli, cli_parse_args=True)
    args, _unknown = cli.parse_known_args()

    return Scene.load(
        config_file=getattr(args, "cfg_file", None), cli_source=cli_source
    )


@tm.decorators.TrameApp()
class CardioApp:
    """A session, and a page to drive it from."""

    def __init__(self, server=None):
        self.server = tm.app.get_server(server, client_type="vue3")
        self.session = Session(
            scene_from_command_line(self.server.cli), server=self.server
        )
        UI(self.server, self.session.scene, self.session.logic)


def main():
    if hasattr(vtk, "vtkEGLRenderWindow"):
        os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    app = CardioApp()
    app.server.start(open_browser=False)


if __name__ == "__main__":
    main()
