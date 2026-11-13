#!/usr/bin/env python3
"""Live camera capture with an A4 reference frame (automatic capture + calibration + inspection). Run with --help."""
import sys

from partvision.camera_ui import main

if __name__ == "__main__":
    sys.exit(main())
