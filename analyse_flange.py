#!/usr/bin/env python3
"""Flange/hub analysis: the 'round' profile of analyse_part.py. Same options; see --help."""
import sys

from partvision.cli import main

if __name__ == "__main__":
    sys.exit(main(default_profile="round"))
