#!/usr/bin/env python

# Third Party
import trame as tm
import tomlkit as tk

# Internal
from .scene import Scene
from .logic import Logic
from .ui import UI


class CardioApp(tm.app.TrameApp):
    def __init__(self, name=None):
        super().__init__(server=name, client_type="vue2")

        self.server.cli.add_argument(
            "--config", help="TOML configutation file.", dest="cfg_file", required=True
        )
        args = self.server.cli.parse_args()

        with open(args.cfg_file, mode="rt", encoding="utf-8") as fp:
            cfg = tk.load(fp)

        scene = Scene(**cfg)
        Logic(self.server, scene)
        UI(self.server, scene)


def main():
    app = CardioApp()
    app.server.start(open_browser=False)


if __name__ == "__main__":
    main()
