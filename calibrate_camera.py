#!/usr/bin/env python3
"""Lens calibration (intrinsics + distortion) from chessboard photos, for capture.py --intrinsics.

Print a chessboard, tape it flat, and take 15-25 photos with the SAME camera, resolution, zoom and focus you use
for inspection, covering the whole field of view at different tilts. Then:

    python3 calibrate_camera.py "calib/*.jpg" --board 9x6 --square-mm 25 --out camera_intrinsics.json

--board counts INNER corners (a 10x7-square board has 9x6). Repeat whenever the resolution, zoom or focus changes.
"""
import argparse
import glob
import json
import sys

import cv2
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="glob of chessboard photos, e.g. 'calib/*.jpg'")
    ap.add_argument("--board", default="9x6", help="inner corners COLSxROWS (default 9x6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square size in mm (only scales the extrinsics)")
    ap.add_argument("--out", default="camera_intrinsics.json")
    a = ap.parse_args(argv)
    cols, rows = (int(v) for v in a.board.lower().split("x"))
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * a.square_mm
    objs, imgs, size = [], [], None
    for p in sorted(glob.glob(a.images)):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if size and (g.shape[1], g.shape[0]) != size:
            print(f"skip {p}: {g.shape[1]}x{g.shape[0]} differs from {size[0]}x{size[1]}")
            continue
        size = (g.shape[1], g.shape[0])
        ok, c = cv2.findChessboardCornersSB(g, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
        print(f"{'ok  ' if ok else 'miss'} {p}")
        if ok:
            objs.append(obj)
            imgs.append(c.astype(np.float32))
    if len(objs) < 8:
        raise SystemExit(f"only {len(objs)} usable photos; need at least 8 (15-25 recommended)")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    with open(a.out, "w") as f:
        json.dump(dict(camera_matrix=K.tolist(), dist_coeffs=dist.ravel().tolist(), image_size=list(size), rms=rms,
                       photos=len(objs), board=a.board, square_mm=a.square_mm), f, indent=1)
    print(f"RMS reprojection error {rms:.3f} px from {len(objs)} photos -> {a.out}")
    if rms > 1.0:
        print("warning: RMS above 1 px; retake blurred photos or check the --board size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
