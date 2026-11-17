#!/usr/bin/env python3
"""analyse_part.py with --shape round as the default (a flange/hub whose outline is not clearly round)."""
import sys

from partvision.cli import main

if __name__ == "__main__":
    sys.exit(main(default_shape="round"))
