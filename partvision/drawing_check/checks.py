"""Measurement of single drawing features at their expected place in the photo.

  check_hole(ctx, f, sim, g5)       round hole (or pin/boss seen as a round edge)
  check_opening(ctx, f, sim)        window / slot through the part
  check_contour(ctx, f, sim, band)  inner contour (recess or step edge) along its normals

Each returns a dict with the expected geometry (exp_*) and, when found, the measured one (meas_*), or a status
with a note. They are used both while aligning the drawing (as evidence) and for the final verdicts.
"""
import math

import cv2
import numpy as np

from ..config import C
from ..core import odd
from ..holefit import measure_round_hole, rect_edge_fit
from .common import _k, _resample
from .edges import _normals, edge_offsets


def _surrounded(ctx, x, y, rq, dark=True, margin=None):
    """Fraction of 8 sectors of the ring around a spot that differ from its core in the hole sense (lighter
    around a dark hole, darker around a light one). A hole is enclosed on all sides; a shadow patch is not."""
    k = _k("surround")
    margin = k["margin_gray"] if margin is None else margin
    r0, r1 = k["ring_radii"]
    m = int(math.ceil(r1 * rq)) + 1
    x0, y0 = int(max(x - m, 0)), int(max(y - m, 0))
    x1, y1 = int(min(x + m + 1, ctx.W)), int(min(y + m + 1, ctx.H))
    patch = ctx.gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    core = float(np.mean(patch[d < max(k["core_min_px"], k["core_radii"] * rq)]))
    ang = np.degrees(np.arctan2(yy - y, xx - x)) % 360
    ring = (d >= r0 * rq) & (d < r1 * rq)
    hits = 0
    step = 360 // k["sectors"]
    for a in range(0, 360, step):
        sel = patch[ring & (ang >= a) & (ang < a + step)]
        if sel.size and ((np.median(sel) > core + margin) if dark else (np.median(sel) < core - margin)):
            hits += 1
    return hits / float(k["sectors"])


def _hole_candidates(ctx, c, r, reach):
    """Compact round spots of about hole size (r) within `reach` px of c, nearest first. Dark spots (blind or
    shadowed holes) and light spots (background seen through) both count; crescent-shaped shadows of pocket
    walls and irregular marks do not."""
    k = _k("hole_search")
    m = int(reach + k["window_margin_radii"] * r) + 2
    x0, y0 = int(max(c[0] - m, 0)), int(max(c[1] - m, 0))
    x1, y1 = int(min(c[0] + m, ctx.W)), int(min(c[1] + m, ctx.H))
    g = cv2.GaussianBlur(ctx.gray[y0:y1, x0:x1], (3, 3), 0)
    if g.size == 0 or min(g.shape) < 5:
        return []
    kw = odd(min(min(g.shape) - 1 if min(g.shape) % 2 == 0 else min(g.shape),
                 max(k["median_window_min_px"], k["median_window_radii"] * r)))
    med = cv2.medianBlur(g, max(3, kw)).astype(np.float32)
    out = []
    for dark, D in ((True, med - g.astype(np.float32)), (False, g.astype(np.float32) - med)):
        peak = float(np.percentile(D, k["peak_percentile"]))
        if peak < C["image"]["min_contrast"]:
            continue
        b = (D > max(float(C["image"]["min_contrast"]), k["peak_frac"] * peak)).astype(np.uint8)
        n, lab, st, ce = cv2.connectedComponentsWithStats(b)
        for i in range(1, n):
            a = st[i, 4]
            req = math.sqrt(a / math.pi)
            w, h = st[i, 2], st[i, 3]
            if not k["size_ratio"][0] * r <= req <= k["size_ratio"][1] * r or max(w, h) > k["max_aspect"] * min(w, h):
                continue
            if a < k["min_compact"] * math.pi * (max(w, h) / 2) ** 2:
                continue  # not compact: crescent shadow, streak or irregular mark
            cx, cy = ce[i, 0] + x0, ce[i, 1] + y0
            d = math.hypot(cx - c[0], cy - c[1])
            if d <= reach and _surrounded(ctx, cx, cy, req, dark=dark) >= k["min_surrounded"]:
                out.append((d, cx, cy, req))
    # Threshold sweep for low-contrast holes (e.g. inside a shadowed pocket): stepping up from the darkest
    # level, a hole shows up as a small compact disc before the surrounding shadow joins it.
    gf = g.astype(np.float32)
    lo_, hi_ = float(np.percentile(gf, k["sweep_percentiles"][0])), float(np.percentile(gf, k["sweep_percentiles"][1]))
    seen = []
    for T in np.arange(lo_ + k["sweep_step_gray"], hi_, k["sweep_step_gray"]):
        n, lab, st, ce = cv2.connectedComponentsWithStats((gf < T).astype(np.uint8))
        for i in range(1, n):
            a = st[i, 4]
            req = math.sqrt(a / math.pi)
            w, h = st[i, 2], st[i, 3]
            if not k["sweep_size_ratio"][0] * r <= req <= k["sweep_size_ratio"][1] * r \
                    or max(w, h) > k["sweep_max_aspect"] * min(w, h) or \
                    a < k["sweep_min_compact"] * math.pi * (max(w, h) / 2) ** 2:
                continue
            cx, cy = ce[i, 0] + x0, ce[i, 1] + y0
            d = math.hypot(cx - c[0], cy - c[1])
            if d <= reach and all(math.hypot(cx - q[1], cy - q[2]) > k["same_spot_radii"] * r for q in out + seen) and \
                    _surrounded(ctx, cx, cy, req, dark=True) >= k["min_surrounded"]:
                seen.append((d, cx, cy, req))
    out += seen
    out.sort()
    return out


def check_hole(ctx, f, sim, g5, reach_k=None, exclude=()):
    k, ks = _k("hole_check"), _k("hole_search")
    reach_k = ks["reach_radii"] if reach_k is None else reach_k
    c = sim(np.array([[f["cx"], f["cy"]]]))[0]
    r = f["d"] / 2 * sim.local_scale(f["cx"], f["cy"])
    out = dict(exp_x=c[0], exp_y=c[1], exp_r=r)
    if not (np.all(np.isfinite(c)) and np.isfinite(r)) or not (0 <= c[0] < ctx.W and 0 <= c[1] < ctx.H):
        out.update(status="MISSING", note="the drawing puts this feature outside the photo")
        return out
    if r < C["image"]["min_hole_radius_px"]:
        out.update(status="CAN'T VERIFY", note=f"only {2 * r:.1f} px across in the photo")
        return out
    if r >= C["image"]["big_hole_radius_px"]:
        # large hole / bore: the inside wall can show in the photo, so take the strongest edge near the
        # expected radius (same method as the bore step) instead of tracing the dark core outward
        fe = ctx.ray_edge_fit(c[0], c[1], k["bore_edge_search"][0] * r, k["bore_edge_search"][1] * r,
                              n=k["bore_rays"], floor=max(k["bore_floor_px"], k["bore_floor_radii"] * r))
        ok = fe["rays"] >= k["bore_min_ray_frac"] * k["bore_rays"] \
            and fe["std"] <= max(k["bore_max_scatter_px"], k["bore_max_scatter_radii"] * r)  # inner wall adds scatter
        core0 = ctx.disc_stat(c[0], c[1], 0, k["bore_core_radii"] * r)
        sur0 = ctx.disc_stat(c[0], c[1], k["bore_surround_radii"][0] * r, k["bore_surround_radii"][1] * r)
        if not ok and abs(core0 - sur0) < C["image"]["min_contrast"]:
            out.update(status="MISSING", note=f"no edge and no contrast at the expected place "
                                              f"({abs(core0 - sur0):.0f} gray levels)")
            return out
        out.update(meas_x=fe["x"], meas_y=fe["y"], meas_r=fe["r"], fit_std=fe["std"], quality_ok=bool(ok))
        return out
    core_r, (s0_, s1_) = max(k["core_min_px"], k["core_radii"] * r), k["surround_radii"]
    core = ctx.disc_stat(c[0], c[1], 0, core_r)
    sur = ctx.disc_stat(c[0], c[1], s0_ * r, s1_ * r)
    searched = abs(core - sur) < C["image"]["min_contrast"]
    sh = k["shift_radii"]
    lo_s, hi_s = k["measured_size_ratio"]
    c_exp = c.copy()

    def round_edge(not_dark_inside=False):
        """No dark hole: is there a round edge of about the right size (pin, boss, plug, spot face)? With
        not_dark_inside, only when the inside of that edge is not darker than just outside it (a pin's top, not a
        hole's dark interior)."""
        e0, e1 = k["edge_search"]
        fe = ctx.ray_edge_fit(c_exp[0], c_exp[1], e0 * r, e1 * r, n=k["edge_rays"], floor=k["edge_floor_px"])
        if fe["rays"] >= k["edge_min_ray_frac"] * k["edge_rays"] \
                and fe["std"] <= max(k["edge_max_scatter_px"], (k["pin_max_scatter_radii"] if not_dark_inside
                                                                else k["edge_max_scatter_radii"]) * r) and \
                math.hypot(fe["x"] - c_exp[0], fe["y"] - c_exp[1]) <= k["edge_max_offset_radii"] * r \
                and e0 <= fe["r"] / r <= e1:
            if not_dark_inside:
                pi, (po0, po1) = k["pin_inside_radii"], k["pin_around_radii"]
                inside = ctx.disc_stat(fe["x"], fe["y"], 0, pi * fe["r"])
                around = ctx.disc_stat(fe["x"], fe["y"], po0 * fe["r"], po1 * fe["r"])
                if inside < around - C["image"]["min_contrast"]:
                    return False  # dark inside: a hole (its counterbore edge), not a pin
            out.update(meas_x=fe["x"], meas_y=fe["y"], meas_r=fe["r"], fit_std=fe["std"], quality_ok=True,
                       seen="round edge", note="round edge found but not a dark hole: pin, boss, plug or spot face")
            return True
        return False

    # Prefer a real hole-shaped spot: the nearest compact round spot of about this size within 1.25 hole
    # diameters (far enough to catch a hole shifted by parallax, short of the neighbouring holes).
    cands = _hole_candidates(ctx, c_exp, r, reach_k * r)
    # skip spots that are another drawing hole's place; rank by core darkness with a mild distance penalty
    cands = [q for q in cands if all(math.hypot(q[1] - ex, q[2] - ey) > ks["exclude_radii"] * er
                                     for ex, ey, er in exclude)]
    if reach_k > ks["reach_radii"] and len(cands) > 1:
        # A feature lower than the fitted face (recess floor) is further from the camera, so it appears
        # shifted toward the image centre (principal point). Prefer the spot whose shift points that way.
        toc = np.array([ctx.W / 2 - c_exp[0], ctx.H / 2 - c_exp[1]])
        toc /= max(np.linalg.norm(toc), 1e-9)

        def off_axis(q):
            v = np.array([q[1] - c_exp[0], q[2] - c_exp[1]])
            n = np.linalg.norm(v)
            return 0.0 if n <= sh * r else math.degrees(math.acos(float(np.clip(v @ toc / n, -1, 1))))
        cands.sort(key=lambda q: (off_axis(q) > ks["off_axis_deg"], off_axis(q) if q[0] > sh * r else 0.0, q[0]))
    if cands and cands[0][0] > sh * r and k["prefer_centred_round_edge"] and round_edge(not_dark_inside=True):
        # nothing hole-dark at the expected place, but a clean round edge of the right size exactly there (a pin
        # pressed into the hole, a boss): better evidence than a dark spot beside it, usually the pin's own shadow
        return out
    if cands and (searched or cands[0][0] > sh * r):
        d0, cx_, cy_, _ = cands[0]
        c = np.array([cx_, cy_])
        core = ctx.disc_stat(c[0], c[1], 0, core_r)
        sur = ctx.disc_stat(c[0], c[1], s0_ * r, s1_ * r)
        h = measure_round_hole(ctx, c[0], c[1], r, -1.0 if core < sur else 1.0, g5)
        if h is not None and lo_s <= h["r"] / r <= hi_s \
                and math.hypot(h["x"] - c[0], h["y"] - c[1]) <= k["max_recentre_radii"] * r:
            out.update(meas_x=h["x"], meas_y=h["y"], meas_r=h["r"], fit_std=h["fit_std"], quality_ok=h["quality_ok"])
            if d0 > sh * r:
                out["note"] = f"hole found {d0 / r * 0.5:.1f} diameters from the drawing position"
            return out
        if d0 > sh * r:
            # a round, enclosed spot but its wall is too faint to trace (e.g. deep in a shadowed pocket):
            # report the spot itself rather than whatever is at the expected place
            out.update(meas_x=float(cx_), meas_y=float(cy_), meas_r=float(cands[0][3]), fit_std=float("nan"),
                       quality_ok=False, note="hole located from its dark spot (wall too faint to trace precisely)")
            return out
    if searched and not cands:
        if round_edge():
            return out
        out.update(status="MISSING", note=f"no round hole within 1.25 diameters of the expected place "
                                          f"({abs(core - sur):.0f} gray levels contrast there)")
        return out
    if searched:
        # the hole may be shifted: look for the strongest hole-like spot within one diameter
        best = None
        for dx in np.linspace(-r, r, k["grid_steps"]):
            for dy in np.linspace(-r, r, k["grid_steps"]):
                cc = ctx.disc_stat(c[0] + dx, c[1] + dy, 0, core_r)
                ss = ctx.disc_stat(c[0] + dx, c[1] + dy, s0_ * r, s1_ * r)
                if best is None or abs(cc - ss) > best[0]:
                    best = (abs(cc - ss), dx, dy, cc, ss)
        if best[0] < C["image"]["min_contrast"]:
            if round_edge():
                return out
            out.update(status="MISSING", note=f"no hole contrast at the expected place ({abs(core - sur):.0f} gray levels)")
            return out
        c = c + [best[1], best[2]]
        core, sur = best[3], best[4]
    pol = -1.0 if core < sur else 1.0
    h = measure_round_hole(ctx, c[0], c[1], r, pol, g5)
    bad = h is None or not lo_s <= h["r"] / r <= hi_s
    if searched and (bad or not h["quality_ok"]):
        if round_edge():
            return out
        out.update(status="MISSING", note="no hole at the expected place; the nearest "
                                          "mark within one diameter is not a round hole")
        return out
    if bad:
        if round_edge():
            return out
        out.update(status="MISSING", note="only an irregular mark at the expected place, no round hole or round edge")
        return out
    out.update(meas_x=h["x"], meas_y=h["y"], meas_r=h["r"], fit_std=h["fit_std"], quality_ok=h["quality_ok"])
    return out


def check_opening(ctx, f, sim):
    poly = sim(f["pts"])
    x, y, w, h = cv2.boundingRect(poly.astype(np.float32).reshape(-1, 1, 2))
    k = _k("opening_check")
    size = max(w, h)
    m = int(k["window_margin_frac"] * size) + k["window_margin_px"]
    xa, ya, xb, yb = max(x - m, 0), max(y - m, 0), min(x + w + m, ctx.W), min(y + h + m, ctx.H)
    exp = np.zeros((yb - ya, xb - xa), np.uint8)
    cv2.fillPoly(exp, [(poly - [xa, ya]).astype(np.int32)], 1)
    out = dict(exp_poly=poly)
    if min(w, h) < k["min_px"]:
        out.update(status="CAN'T VERIFY", note=f"only {min(w, h)} px wide in the photo")
        return out
    if f.get("shape") == "rectangular" and C["matching"]["opening_rect_fit"]:
        # its four edges, from the expected rectangle (as the drawing-free analysis measures a window): robust to a
        # backdrop that is shaded or partly lit, which breaks a brightness segmentation of the inside
        r = rect_edge_fit(ctx, cv2.minAreaRect(poly.astype(np.float32).reshape(-1, 1, 2)))
        if r and r["ok"]:
            fit = np.zeros_like(exp)
            corners = np.array(r["corners"])
            cv2.fillPoly(fit, [(corners - [xa, ya]).astype(np.int32)], 1)
            iou = float((fit & exp).sum() / max((fit | exp).sum(), 1))
            out.update(meas_x=r["x"], meas_y=r["y"], meas_L=r["length"], meas_W=r["width"], iou=iou,
                       meas_contour=corners.tolist(), method="4-edge line fit", fit_std_px=r["std"])
            return out
    g = ctx.gray[ya:yb, xa:xb].astype(np.float32)
    kb = max(1, int(k["band_frac"] * min(w, h)))
    inner = cv2.erode(exp, np.ones((2 * kb + 1,) * 2, np.uint8))
    ring = cv2.dilate(exp, np.ones((4 * kb + 1,) * 2, np.uint8)) - cv2.dilate(exp, np.ones((2 * kb + 1,) * 2, np.uint8))
    if inner.sum() < k["min_samples"] or ring.sum() < k["min_samples"]:
        out.update(status="CAN'T VERIFY", note="too small to sample")
        return out
    gi, go = float(np.median(g[inner > 0])), float(np.median(g[ring > 0]))
    if abs(gi - go) < C["image"]["min_contrast"]:
        out.update(status="MISSING", note=f"no contrast between inside and around ({abs(gi - go):.0f} gray levels)")
        return out
    b = (((g - (gi + go) / 2) * (1 if gi > go else -1)) > 0).astype(np.uint8)
    n, lab, st, ce = cv2.connectedComponentsWithStats(b)
    if n < 2:
        out.update(status="MISSING", note="nothing segmented")
        return out
    ov = [(int(((lab == i) & (exp > 0)).sum()), i) for i in range(1, n)]
    _, i = max(ov)
    meas = (lab == i).astype(np.uint8)
    iou = float((meas & exp).sum() / max((meas | exp).sum(), 1))
    cc, _ = cv2.findContours(meas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cc, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    out.update(meas_x=float(ce[i, 0] + xa), meas_y=float(ce[i, 1] + ya), meas_L=max(rw, rh), meas_W=min(rw, rh),
               iou=iou, meas_contour=(c[:, 0, :] + [xa, ya]).tolist(), method="brightness region")
    return out


def check_contour(ctx, f, sim, band):
    """Edge-based check of an inner contour (recess/step/opening edge) along its normals."""
    ke = _k("edges")
    poly = sim(_resample(f["pts"], ke["contour_samples"]))
    off, st = edge_offsets(ctx, poly, band, None)
    found = st >= max(ke["contour_strong_min"], ke["strong_frac"] * float(np.median(st)))
    return dict(exp_poly=poly, off=off, found=found, meas=(poly + off[:, None] * _normals(poly)))
