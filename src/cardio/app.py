#!/usr/bin/env python

# Third Party
import pydantic_settings as ps
import tomlkit as tk
import trame as tm

from . import __version__
from .logic import Logic

# Internal
from .scene import Scene
from .ui import UI


class CardioApp(tm.app.TrameApp):
    def __init__(self, name=None):
        super().__init__(server=name, client_type="vue2")

        # Add config file argument to Trame's parser
        self.server.cli.add_argument(
            "--config", help="TOML configuration file.", dest="cfg_file", required=False
        )

        # Add version argument
        self.server.cli.add_argument(
            "--version", action="version", version=f"cardio {__version__}"
        )

        # Create CLI settings source with Trame's parser - enable argument parsing
        cli_settings = ps.CliSettingsSource(
            Scene, root_parser=self.server.cli, cli_parse_args=True
        )

        # Parse arguments to get config file path (use parse_known_args to avoid conflicts)
        args, unknown = self.server.cli.parse_known_args()
        config_file = getattr(args, "cfg_file", None)

        # Set the CLI source and config file on the Scene class temporarily
        Scene._cli_source = cli_settings
        Scene._config_file = config_file

        try:
            # Create Scene with CLI and config file support
            scene = Scene()
        finally:
            # Clean up class attributes
            if hasattr(Scene, "_cli_source"):
                delattr(Scene, "_cli_source")
            if hasattr(Scene, "_config_file"):
                delattr(Scene, "_config_file")

        Logic(self.server, scene)
        UI(self.server, scene)


def main():
    app = CardioApp()
    app.server.start(open_browser=False)


if __name__ == "__main__":
    main()
