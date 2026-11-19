"""Edge measurement along the normals of an expected contour (outline, recess edge)."""
import cv2
import numpy as np

from ..core import odd
from .common import _k


def _normals(poly):
    """Unit normals pointing out of the closed polygon."""
    p = np.asarray(poly, float)
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    n = np.c_[t[:, 1], -t[:, 0]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    c = p.astype(np.float32).reshape(-1, 1, 2)
    ke = _k("edges")
    k = max(1, len(p) // ke["normal_probes"])
    probe = p[::k] + n[::k] * ke["normal_probe_px"]
    outside = np.mean([cv2.pointPolygonTest(c, (float(x), float(y)), False) < 0 for x, y in probe])
    return n if outside >= 0.5 else -n


def inner_limits(poly_px, circles_px, margin=None):
    """Per outline point: how far inward the edge search may go before reaching a drawing hole
    (hole centre distance - margin x radius). Keeps the outline from snapping onto nearby hole rims."""
    ke = _k("edges")
    margin = ke["hole_margin_radii"] if margin is None else margin
    lim = np.full(len(poly_px), np.inf)
    for cx, cy, r in circles_px:
        lim = np.minimum(lim, np.hypot(poly_px[:, 0] - cx, poly_px[:, 1] - cy) - margin * r)
    return np.maximum(lim, ke["hole_margin_min_px"])


def edge_offsets(ctx, poly_px, band, sign=None, inner_lim=None):
    """Strongest edge along the outward normal at each point of the expected contour.
    Returns offset in px (+ = edge lies outside the expected line) and edge strength (gray levels / px).
    sign=+1 when the image gets lighter going outward (dark part on light background), -1 the reverse,
    None for either direction."""
    ke = _k("edges")
    n = _normals(poly_px)
    step = ke["profile_step_px"]
    rs = np.arange(-band, band + 0.01, step, dtype=np.float32)
    P = np.asarray(poly_px, np.float32)[:, None, :] + rs[None, :, None] * n[:, None, :].astype(np.float32)
    prof = cv2.remap(ctx.gs, np.ascontiguousarray(P[..., 0]), np.ascontiguousarray(P[..., 1]), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    prof = cv2.GaussianBlur(prof, (odd(ke["profile_blur_px"]), 1), 0)
    g = np.gradient(prof, axis=1) / step
    if inner_lim is not None:  # positions further inward than allowed cannot be the outline edge
        g = np.where(rs[None, :] < -np.asarray(inner_lim, np.float32)[:, None], 0.0, g)
    a = np.arange(len(g))
    if sign is None:
        i = np.abs(g).argmax(1)
        return rs[i], np.abs(g)[a, i]
    i = (sign * g).argmax(1)
    return rs[i], (sign * g)[a, i]
