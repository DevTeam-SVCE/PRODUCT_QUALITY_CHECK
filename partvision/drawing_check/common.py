"""Shared helpers of the drawing check: settings access, colours, geometry utilities, feature classes."""
import cv2
import numpy as np

from ..config import C


def _k(section):
    """Settings of one stage of the drawing check (config drawing_check.<section>)."""
    return C["drawing_check"][section]

CYAN, GREEN, RED, ORANGE, MAG, WHITE = ((80, 220, 255), (40, 220, 40), (255, 60, 60), (255, 160, 30),
                                        (255, 0, 230), (255, 255, 255))
STATUS_COL = {"OK": GREEN, "DETECTED": (170, 235, 40), "DEVIATES": RED, "MISSING": MAG, "CAN'T VERIFY": ORANGE, "UNCERTAIN": ORANGE,
              "NOT MEASURED": WHITE}


def _resample(poly, n):
    p = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    u = np.linspace(0, cum[-1], n, endpoint=False)
    return np.c_[np.interp(u, cum, p[:, 0]), np.interp(u, cum, p[:, 1])]


def _nearest(a, b):
    """For each point in a, the index of and distance to the nearest point in b (chunked brute force)."""
    idx = np.empty(len(a), int)
    dist = np.empty(len(a))
    for i in range(0, len(a), 512):
        d = np.hypot(a[i:i + 512, None, 0] - b[None, :, 0], a[i:i + 512, None, 1] - b[None, :, 1])
        idx[i:i + 512] = d.argmin(1)
        dist[i:i + 512] = d.min(1)
    return idx, dist


def _encloses(f, model):
    """True when the contour contains other drawing features (a recess or step edge, not a plain opening)."""
    poly = f["pts"].astype(np.float32).reshape(-1, 1, 2)
    return any(g is not f and cv2.pointPolygonTest(poly, (float(g["cx"]), float(g["cy"])), False) > 0
               for g in model["features"])


def _plain_openings(model):
    return [f for f in model["features"] if f["kind"] == "opening" and not _encloses(f, model)]


def _runs(flags):
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out
