"""Precise measurement of one round hole, shared by the bolt-circle and generic hole finders."""
import math

import cv2
import numpy as np

from .core import fit_circle


def hole_wall_fit(ctx, cx, cy, level, hole_pol, rmin, rmax, n=72, passes=3, floor=0.6):
    """Walk outward from the hole core to the first crossing of `level` on each ray, then fit a circle.
    hole_pol is -1 for holes darker than the surface, +1 for brighter. Specks inside the hole give
    too-short rays, which the robust fit rejects."""
    rs = np.arange(rmin, rmax, 0.25, dtype=np.float32)
    for _ in range(passes):
        pts = []
        for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
            p = ctx.ray_profile(cx, cy, th, rs)
            out = np.where((p - level) * hole_pol < 0)[0]
            if len(out) and out[0] > 0:
                i = out[0]
                r_ = rs[i - 1] + 0.25 * (level - p[i - 1]) / (p[i] - p[i - 1] if p[i] != p[i - 1] else 1e-3)
                pts.append((cx + r_ * np.cos(th), cy + r_ * np.sin(th)))
        if len(pts) < 8:
            return None
        cx, cy, r, std, used, _ = fit_circle(pts, iters=6, floor=floor)
    return dict(x=cx, y=cy, r=r, d=2 * r, std=std, rays=used, of=n)


def measure_round_hole(ctx, x0, y0, rm, hole_pol, g5):
    """Recentre on the hole, fit its wall, cross-check with a blob area and look for a ring/counterbore."""
    H, W = ctx.H, ctx.W
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(rm)) or rm <= 0 or \
            not (-rm < x0 < W + rm and -rm < y0 < H + rm):
        return None  # expected outside the photo (e.g. a wrong drawing being tried)
    core = ctx.disc_stat(x0, y0, 0, max(2, 0.33 * rm))
    surf = ctx.disc_stat(x0, y0, 1.6 * rm, 2.1 * rm)
    level = (core + surf) / 2
    half = int(round(1.8 * rm))
    xa, ya = int(max(x0 - half, 0)), int(max(y0 - half, 0))
    roi = g5[ya:int(min(y0 + half + 1, H)), xa:int(min(x0 + half + 1, W))].astype(np.float32)
    b = (((roi - level) * hole_pol) > 0).astype(np.uint8)
    n, lab, st, ce = cv2.connectedComponentsWithStats(b)
    sx, sy = int(round(x0 - xa)), int(round(y0 - ya))
    L = lab[sy, sx] if 0 <= sy < lab.shape[0] and 0 <= sx < lab.shape[1] else 0
    if L == 0 and n > 1:
        dd = np.hypot(ce[1:, 0] - sx, ce[1:, 1] - sy)
        L = 1 + int(dd.argmin()) if dd.min() < rm else 0
    if L:
        x0, y0 = ce[L, 0] + xa, ce[L, 1] + ya
    f = hole_wall_fit(ctx, x0, y0, level, hole_pol, 0.25 * rm, 1.7 * rm, floor=0.05 * rm)
    if f is None:
        return None
    h = dict(x=f["x"], y=f["y"], r=f["r"], d=f["d"], fit_std=f["std"], rays=f["rays"], of=f["of"])
    h["core"] = ctx.disc_stat(h["x"], h["y"], 0, max(2, 0.33 * rm))
    h["surface"] = ctx.disc_stat(h["x"], h["y"], h["r"] + 0.33 * rm, h["r"] + 0.75 * rm)
    t = (h["core"] + h["surface"]) / 2
    half = int(round(2.5 * rm))
    xa, ya = int(max(h["x"] - half, 0)), int(max(h["y"] - half, 0))
    roi = g5[ya:int(min(h["y"] + half + 1, H)), xa:int(min(h["x"] + half + 1, W))].astype(np.float32)
    b = (((roi - t) * hole_pol) > 0).astype(np.uint8)
    n, lab, st, ce = cv2.connectedComponentsWithStats(b)
    sx, sy = int(round(h["x"] - xa)), int(round(h["y"] - ya))
    L = lab[sy, sx] if 0 <= sy < lab.shape[0] and 0 <= sx < lab.shape[1] else 0
    h["blob_d"] = None
    if L:
        x_, y_, w_, hh_ = st[L, :4]
        if x_ > 0 and y_ > 0 and x_ + w_ < roi.shape[1] and y_ + hh_ < roi.shape[0]:
            h["blob_d"] = float(2 * math.sqrt(st[L, 4] / math.pi))
    rr = np.arange(h["r"] + 0.12 * rm, h["r"] + 1.1 * rm, 0.5, dtype=np.float32)
    prof = np.median([ctx.ray_profile(h["x"], h["y"], t_, rr) for t_ in np.linspace(0, 2 * np.pi, 72, endpoint=False)],
                     axis=0)
    outer = float(np.median(prof[-max(3, len(rr) // 4):]))
    first = prof[:max(4, int(len(rr) * 0.75))]
    im = int((first * hole_pol).argmax())  # most hole-coloured point near the wall (a groove)
    h["ring_dip"] = float((outer - prof[im]) * -hole_pol)
    h["ring_dip_at_r"] = float(rr[im])
    h["local_contrast"] = float((h["surface"] - h["core"]) * -hole_pol)
    rec = np.where(((prof[im:] - (outer + prof[im]) / 2) * -hole_pol) > 0)[0]
    h["ring_outer_r"] = float(rr[im + rec[0]]) if len(rec) else None
    h["quality_ok"] = bool(f["std"] <= 0.1 * rm and f["rays"] >= 0.75 * f["of"] and
                           (h["blob_d"] is None or 0.8 <= h["blob_d"] / h["d"] <= 1.2))
    return h
