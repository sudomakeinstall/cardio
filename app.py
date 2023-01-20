#!/usr/bin/env python

from trame.app import get_server, Server

import cyclone as cy

def main(server = None, **kwargs):

    if server is None:
        server = get_server()

    if isinstance(server, str):
        server = get_server(server)

    server.cli.add_argument("--config", help="TOML configutation file.", dest="cfg_file", required=True)
    args = server.cli.parse_args()

    scene = cy.Scene(args.cfg_file)

    cy.Logic(server, scene)
    cy.UI(server, scene)
    
    server.start(open_browser=False)

if __name__ == "__main__":
    main()