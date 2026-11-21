# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The `gf` command's entry point.

The commands of section 2 of the surface spec are built on the Python API
in their own change; until then `gf` prints its help and version and says
so for any command.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

HELP = """usage: gf <resource> <verb> [NAME] [flags]

General Forecast from the command line. The commands arrive with the next
release; until then, use the Python API:

  import generalforecast as gf
  gf.login()
  gf.contexts.create("adult", data="adult.csv")

options:
  -h, --help     show this help and exit
  -V, --version  show the version and exit
"""


def main(argv: Sequence[str] | None = None) -> None:
    """Prints help or the version; any command exits 2 with a pointer to the Python API."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        sys.stdout.write(HELP)
        return
    if args[0] in ("-V", "--version"):
        from generalforecast import __version__

        sys.stdout.write(f"gf {__version__}\n")
        return
    sys.stderr.write(
        f"error: `gf {args[0]}` is not available in this release\n"
        "help: use the Python API, `import generalforecast as gf`\n"
    )
    raise SystemExit(2)
