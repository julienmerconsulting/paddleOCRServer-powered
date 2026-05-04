"""Entry point for `python -m paddleocrserver_powered`."""

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
