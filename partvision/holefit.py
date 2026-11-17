"""Precise measurement of one round hole, shared by the bolt-circle, generic hole and drawing-check steps.

Sizes are multiples of the expected hole radius rm; settings in config analysis.hole_fit."""
import math

import cv2
import numpy as np

from .config import C
from .core import fit_circle


def hole_wall_fit(ctx, cx, cy, level, hole_pol, rmin, rmax, n=None, passes=3, floor=0.6):
    """Walk outward from the hole core to the first crossing of `level` on each ray, then fit a circle.
    hole_pol is -1 for holes darker than the surface, +1 for brighter. Specks inside the hole give
    too-short rays, which the robust fit rejects."""
    k = C["analysis"]["hole_fit"]
    n = n or k["rays"]
    rs = np.arange(rmin, rmax, k["radial_step_px"], dtype=np.float32)
    for _ in range(passes):
        pts = []
        for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
            p = ctx.ray_profile(cx, cy, th, rs)
            out = np.where((p - level) * hole_pol < 0)[0]
            if len(out) and out[0] > 0:
                i = out[0]
                r_ = rs[i - 1] + k["radial_step_px"] * (level - p[i - 1]) / (p[i] - p[i - 1] if p[i] != p[i - 1] else 1e-3)
                pts.append((cx + r_ * np.cos(th), cy + r_ * np.sin(th)))
        if len(pts) < k["min_rays"]:
            return None
        cx, cy, r, std, used, _ = fit_circle(pts, iters=6, floor=floor)
    return dict(x=cx, y=cy, r=r, d=2 * r, std=std, rays=used, of=n)


def measure_round_hole(ctx, x0, y0, rm, hole_pol, g5):
    """Recentre on the hole, fit its wall, cross-check with a blob area and look for a ring/counterbore."""
    k = C["analysis"]["hole_fit"]
    H, W = ctx.H, ctx.W
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(rm)) or rm <= 0 or \
            not (-rm < x0 < W + rm and -rm < y0 < H + rm):
        return None  # expected outside the photo (e.g. a wrong drawing being tried)
    core = ctx.disc_stat(x0, y0, 0, max(2, k["core_radii"] * rm))
    surf = ctx.disc_stat(x0, y0, k["surface_ring_radii"][0] * rm, k["surface_ring_radii"][1] * rm)
    level = (core + surf) / 2
    half = int(round(k["recentre_window_radii"] * rm))
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
    f = hole_wall_fit(ctx, x0, y0, level, hole_pol, k["wall_search_radii"][0] * rm, k["wall_search_radii"][1] * rm,
                      floor=k["fit_floor_radii"] * rm)
    if f is None:
        return None
    h = dict(x=f["x"], y=f["y"], r=f["r"], d=f["d"], fit_std=f["std"], rays=f["rays"], of=f["of"])
    h["core"] = ctx.disc_stat(h["x"], h["y"], 0, max(2, k["core_radii"] * rm))
    s0, s1 = k["surface_beyond_wall_radii"]
    h["surface"] = ctx.disc_stat(h["x"], h["y"], h["r"] + s0 * rm, h["r"] + s1 * rm)
    t = (h["core"] + h["surface"]) / 2
    half = int(round(k["blob_window_radii"] * rm))
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
    r0, r1 = k["ring_profile_radii"]
    rr = np.arange(h["r"] + r0 * rm, h["r"] + r1 * rm, k["ring_profile_step_px"], dtype=np.float32)
    prof = np.median([ctx.ray_profile(h["x"], h["y"], t_, rr)
                      for t_ in np.linspace(0, 2 * np.pi, k["rays"], endpoint=False)], axis=0)
    outer = float(np.median(prof[-max(3, len(rr) // 4):]))  # outermost quarter of the profile: plain surface
    first = prof[:max(4, int(len(rr) * k["ring_search_frac"]))]
    im = int((first * hole_pol).argmax())  # most hole-coloured point near the wall (a groove)
    h["ring_dip"] = float((outer - prof[im]) * -hole_pol)
    h["ring_dip_at_r"] = float(rr[im])
    h["local_contrast"] = float((h["surface"] - h["core"]) * -hole_pol)
    rec = np.where(((prof[im:] - (outer + prof[im]) / 2) * -hole_pol) > 0)[0]
    h["ring_outer_r"] = float(rr[im + rec[0]]) if len(rec) else None
    lo, hi = k["blob_size_ratio"]
    h["quality_ok"] = bool(f["std"] <= k["max_scatter_radii"] * rm and f["rays"] >= k["min_ray_frac"] * f["of"] and
                           (h["blob_d"] is None or lo <= h["blob_d"] / h["d"] <= hi))
    return h


def rect_edge_fit(ctx, rect):
    """Precise rectangle (window, slot end, pocket) from its four edges: on each side of the rough rectangle,
    profiles across the side give the strongest-gradient position; a robust line through those points per side,
    and the corners from the intersections of adjacent lines. rect is a cv2.minAreaRect. Settings: config
    analysis.rect_fit (sizes relative to the rectangle's shorter side). Returns None when a side has no edge."""
    k = C["analysis"]["rect_fit"]
    box = cv2.boxPoints(rect).astype(np.float64)
    ctr = box.mean(0)
    short = max(min(rect[1]), 1.0)
    band = max(k["band_min_px"], k["band_frac"] * short)
    offs = np.arange(-band, band + 1e-9, k["profile_step_px"])
    lines, scatter = [], []
    for i in range(4):
        p0, p1 = box[i], box[(i + 1) % 4]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1:
            return None
        d /= L
        nrm = np.array([-d[1], d[0]])
        if np.dot((p0 + p1) / 2 - ctr, nrm) < 0:
            nrm = -nrm  # outward
        ts = np.linspace(k["end_margin"], 1 - k["end_margin"], k["samples_per_side"])
        pts = []
        for t in ts:
            base = p0 + t * L * d
            xy = base[None, :] + offs[:, None] * nrm[None, :]
            prof = cv2.remap(ctx.gs, xy[:, 0].astype(np.float32).reshape(-1, 1), xy[:, 1].astype(np.float32).reshape(-1, 1),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            gr = np.abs(np.gradient(cv2.GaussianBlur(prof.reshape(-1, 1), (1, k["profile_blur"]), 0).ravel()))
            j = int(np.argmax(gr))
            pts.append(base + offs[j] * nrm)
        pts = np.array(pts, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        res = (pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx
        med = float(np.median(np.abs(res)))
        keep = np.abs(res) <= max(k["inlier_px"], 3 * 1.4826 * med)
        if keep.sum() >= 2:
            vx, vy, x0, y0 = cv2.fitLine(pts[keep], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            res = (pts[keep, 0] - x0) * vy - (pts[keep, 1] - y0) * vx
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
        scatter.append((float(np.std(res)), int(keep.sum())))
    corners = []
    for i in range(4):
        (a, u), (b, v) = lines[i - 1], lines[i]
        M = np.array([u, -v]).T
        if abs(np.linalg.det(M)) < 1e-9:
            return None
        s = np.linalg.solve(M, b - a)
        corners.append(a + s[0] * u)
    corners = np.array(corners)
    sides = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    a_, b_ = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    e = corners[1] - corners[0] if a_ >= b_ else corners[2] - corners[1]
    std = max(s for s, _ in scatter)
    used = sum(n for _, n in scatter)
    return dict(corners=corners.tolist(), x=float(corners[:, 0].mean()), y=float(corners[:, 1].mean()),
                length=max(a_, b_), width=min(a_, b_), axis_deg=float(math.degrees(math.atan2(e[1], e[0])) % 180),
                std=std, rays=used, of=4 * k["samples_per_side"],
                ok=bool(std <= max(k["max_scatter_px"], k["max_scatter_frac"] * short)
                        and used >= k["min_point_frac"] * 4 * k["samples_per_side"]))
