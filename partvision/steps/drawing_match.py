"""Check a part against its drawing: register drawing -> photo, then verify every feature where it should be."""
import itertools
import math

import cv2
import numpy as np

from .. import drawing as dwg
from ..core import odd
from ..holefit import measure_round_hole
from ..config import C
from ..registry import step

CYAN, GREEN, RED, ORANGE, MAG, WHITE = ((80, 220, 255), (40, 220, 40), (255, 60, 60), (255, 160, 30),
                                        (255, 0, 230), (255, 255, 255))
STATUS_COL = {"OK": GREEN, "DETECTED": (170, 235, 40), "DEVIATES": RED, "MISSING": MAG, "CAN'T VERIFY": ORANGE, "UNCERTAIN": ORANGE,
              "NOT MEASURED": WHITE}


class Sim:
    """x_img = s * R(theta) * M * x_dwg + t, M = diag(1, -1) when the photo shows the mirror image."""

    def __init__(self, s, theta, t, mirror):
        self.s, self.theta, self.t, self.mirror = s, theta, np.asarray(t, float), mirror

    @property
    def A(self):
        c, sn = math.cos(self.theta), math.sin(self.theta)
        M = np.diag([1.0, -1.0]) if self.mirror else np.eye(2)
        return self.s * np.array([[c, -sn], [sn, c]]) @ M

    def __call__(self, p):
        return np.asarray(p, float) @ self.A.T + self.t

    def inverse(self, p):
        return (np.asarray(p, float) - self.t) @ np.linalg.inv(self.A).T

    def local_scale(self, x, y):
        return self.s


class Proj:
    """Perspective (homography) mapping drawing mm -> image px, for a flat face photographed at an angle.
    Offers the same interface as Sim; s/A/t/theta describe the local linear behaviour at `ref` (mm)."""

    def __init__(self, H, ref):
        self.H = np.asarray(H, float)
        self.Hi = np.linalg.inv(self.H)
        self.ref = np.asarray(ref, float)
        J = self.jacobian(*self.ref)
        self.mirror = bool(np.linalg.det(J) < 0)
        self.s = math.sqrt(abs(np.linalg.det(J)))
        self.A = J
        self.t = self(self.ref[None])[0] - J @ self.ref
        B = J @ np.diag([1.0, -1.0]) if self.mirror else J
        self.theta = math.atan2(B[1, 0], B[0, 0])

    def __call__(self, p):
        p = np.asarray(p, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def inverse(self, p):
        p = np.asarray(p, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.Hi).reshape(-1, 2)

    def jacobian(self, x, y, h=0.01):
        c = self(np.array([[x, y], [x + h, y], [x, y + h]]))
        return np.c_[(c[1] - c[0]) / h, (c[2] - c[0]) / h]

    def local_scale(self, x, y):
        return math.sqrt(abs(np.linalg.det(self.jacobian(x, y))))


def fit_similarity(src, dst, mirror, w=None, fixed_scale=None):
    """Weighted Umeyama similarity (no reflection beyond the fixed mirror)."""
    src = np.asarray(src, float) @ (np.diag([1.0, -1.0]) if mirror else np.eye(2)).T
    dst = np.asarray(dst, float)
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    W = w / w.sum()
    ms, md = (W[:, None] * src).sum(0), (W[:, None] * dst).sum(0)
    S, D = src - ms, dst - md
    cov = (W[:, None] * D).T @ S
    U, sig, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ np.diag([1.0, d]) @ Vt
    s = fixed_scale or float((sig * np.array([1.0, d])).sum() / (W * (S ** 2).sum(1)).sum())
    t = md - s * R @ ms
    return Sim(s, math.atan2(R[1, 0], R[0, 0]), t, mirror)


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


def register(ctx, model, fixed_scale, mirror_opt):
    p = ctx.part
    cont = p["contour"][:, 0, :].astype(float)
    opts = model["outline"]["pts"]
    holes = [f for f in model["features"] if f["kind"] == "hole"]
    dA = abs(cv2.contourArea(opts.astype(np.float32).reshape(-1, 1, 2)))
    s0 = fixed_scale or math.sqrt(p["area"] / max(dA, 1e-9))
    cd = np.array(dwg._centroid(opts))
    ci = np.array([p["cx"], p["cy"]])

    # coarse search on a small raster: outline overlap + hole evidence
    bx, by, bw, bh = p["bbox"]
    m = int(0.15 * max(bw, bh))
    x0, y0 = bx - m, by - m
    f = 220.0 / (max(bw, bh) + 2 * m)
    Wl, Hl = int((bw + 2 * m) * f) + 1, int((bh + 2 * m) * f) + 1
    img_mask = np.zeros((Hl, Wl), np.uint8)
    cv2.fillPoly(img_mask, [((cont - [x0, y0]) * f).astype(np.int32)], 1)
    g = cv2.resize(ctx.gray[max(y0, 0):by + bh + m, max(x0, 0):bx + bw + m], None, fx=f, fy=f)
    g = cv2.copyMakeBorder(g, max(0, -y0), 0, max(0, -x0), 0, cv2.BORDER_REPLICATE)[:Hl, :Wl]
    g = cv2.copyMakeBorder(g, 0, Hl - g.shape[0], 0, Wl - g.shape[1], cv2.BORDER_REPLICATE)
    D = np.abs(g.astype(np.float32) - cv2.medianBlur(g, odd(0.15 * 220)).astype(np.float32))
    hole_c = np.array([[h["cx"], h["cy"]] for h in holes]) if holes else np.zeros((0, 2))
    best = None
    mirrors = {"auto": (False, True), "no": (False,), "yes": (True,)}[mirror_opt]

    def score(sim):
        poly = ((sim(opts) - [x0, y0]) * f).astype(np.int32)
        dm = np.zeros_like(img_mask)
        cv2.fillPoly(dm, [poly], 1)
        inter = (dm & img_mask).sum()
        iou = inter / max((dm | img_mask).sum(), 1)
        hs = 0.0
        if len(hole_c):
            q = ((sim(hole_c) - [x0, y0]) * f)
            ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < Wl - 1) & (q[:, 1] < Hl - 1)
            if ok.any():
                hs = float(np.mean(np.clip(D[q[ok, 1].astype(int), q[ok, 0].astype(int)] / 40.0, 0, 1)))
        return iou + 0.3 * hs, iou

    def make(theta, mir):
        M = np.diag([1.0, -1.0]) if mir else np.eye(2)
        c, sn = math.cos(theta), math.sin(theta)
        A = s0 * np.array([[c, -sn], [sn, c]]) @ M
        return Sim(s0, theta, ci - A @ cd, mir)

    for mir in mirrors:
        for a in np.arange(0, 360, 2.0):
            sc = score(make(math.radians(a), mir))
            if best is None or sc[0] > best[0]:
                best = (sc[0], sc[1], a, mir)
    _, _, a_best, mir = best
    for a in np.arange(a_best - 2, a_best + 2.01, 0.25):
        sc = score(make(math.radians(a), mir))
        if sc[0] > best[0]:
            best = (sc[0], sc[1], a, mir)
    sim = make(math.radians(best[2]), best[3])
    coarse_iou = best[1]

    # refine on the outline edge (trimmed nearest-point ICP)
    dpts = _resample(opts, 800)
    size_px = s0 * model["outline"]["size"]
    for it in range(12):
        q = sim(dpts)
        idx, dist = _nearest(q, cont)
        thr = max(2.0, size_px * (0.05 if it < 4 else 0.02))
        keep = dist < thr
        if keep.sum() > 20:
            keep &= dist <= max(2.0, 2.5 * np.median(dist[keep]))
        if keep.sum() < 20:
            break
        sim = fit_similarity(dpts[keep], cont[idx[keep]], sim.mirror, fixed_scale=fixed_scale)
    q = sim(dpts)
    _, dist = _nearest(q, cont)
    return sim, dict(coarse_iou=coarse_iou, mirror=sim.mirror, angle_deg=math.degrees(sim.theta) % 360,
                     px_per_mm=sim.s, outline_rms_px=float(np.sqrt(np.mean(np.minimum(dist, 3 * np.median(dist)) ** 2))),
                     outline_within_2px=float(np.mean(dist <= 2.0))), dpts


def _surrounded(ctx, x, y, rq, dark=True, margin=12):
    """Fraction of 8 sectors of the ring around a spot that differ from its core in the hole sense (lighter
    around a dark hole, darker around a light one). A hole is enclosed on all sides; a shadow patch is not."""
    m = int(math.ceil(2.0 * rq)) + 1
    x0, y0 = int(max(x - m, 0)), int(max(y - m, 0))
    x1, y1 = int(min(x + m + 1, ctx.W)), int(min(y + m + 1, ctx.H))
    patch = ctx.gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    core = float(np.mean(patch[d < max(1.5, 0.5 * rq)]))
    ang = np.degrees(np.arctan2(yy - y, xx - x)) % 360
    ring = (d >= 1.3 * rq) & (d < 2.0 * rq)
    hits = 0
    for a in range(0, 360, 45):
        sel = patch[ring & (ang >= a) & (ang < a + 45)]
        if sel.size and ((np.median(sel) > core + margin) if dark else (np.median(sel) < core - margin)):
            hits += 1
    return hits / 8.0


def _hole_candidates(ctx, c, r, reach):
    """Compact round spots of about hole size (r) within `reach` px of c, nearest first. Dark spots (blind or
    shadowed holes) and light spots (background seen through) both count; crescent-shaped shadows of pocket
    walls and irregular marks do not."""
    m = int(reach + 2.5 * r) + 2
    x0, y0 = int(max(c[0] - m, 0)), int(max(c[1] - m, 0))
    x1, y1 = int(min(c[0] + m, ctx.W)), int(min(c[1] + m, ctx.H))
    g = cv2.GaussianBlur(ctx.gray[y0:y1, x0:x1], (3, 3), 0)
    if g.size == 0 or min(g.shape) < 5:
        return []
    k = odd(min(min(g.shape) - 1 if min(g.shape) % 2 == 0 else min(g.shape), max(9, 6 * r)))
    med = cv2.medianBlur(g, max(3, k)).astype(np.float32)
    out = []
    for dark, D in ((True, med - g.astype(np.float32)), (False, g.astype(np.float32) - med)):
        peak = float(np.percentile(D, 99.5))
        if peak < C["image"]["min_contrast"]:
            continue
        b = (D > max(float(C["image"]["min_contrast"]), 0.45 * peak)).astype(np.uint8)
        n, lab, st, ce = cv2.connectedComponentsWithStats(b)
        for i in range(1, n):
            a = st[i, 4]
            req = math.sqrt(a / math.pi)
            w, h = st[i, 2], st[i, 3]
            if not 0.5 * r <= req <= 1.6 * r or max(w, h) > 1.5 * min(w, h):
                continue
            if a < 0.6 * math.pi * (max(w, h) / 2) ** 2:
                continue  # not compact: crescent shadow, streak or irregular mark
            cx, cy = ce[i, 0] + x0, ce[i, 1] + y0
            d = math.hypot(cx - c[0], cy - c[1])
            if d <= reach and _surrounded(ctx, cx, cy, req, dark=dark) >= 0.75:
                out.append((d, cx, cy, req))
    # Threshold sweep for low-contrast holes (e.g. inside a shadowed pocket): stepping up from the darkest
    # level, a hole shows up as a small compact disc before the surrounding shadow joins it.
    gf = g.astype(np.float32)
    lo_, hi_ = float(np.percentile(gf, 0.5)), float(np.percentile(gf, 60))
    seen = []
    for T in np.arange(lo_ + 3, hi_, 3.0):
        n, lab, st, ce = cv2.connectedComponentsWithStats((gf < T).astype(np.uint8))
        for i in range(1, n):
            a = st[i, 4]
            req = math.sqrt(a / math.pi)
            w, h = st[i, 2], st[i, 3]
            if not 0.45 * r <= req <= 1.6 * r or max(w, h) > 1.4 * min(w, h) or \
                    a < 0.65 * math.pi * (max(w, h) / 2) ** 2:
                continue
            cx, cy = ce[i, 0] + x0, ce[i, 1] + y0
            d = math.hypot(cx - c[0], cy - c[1])
            if d <= reach and all(math.hypot(cx - q[1], cy - q[2]) > 0.5 * r for q in out + seen) and \
                    _surrounded(ctx, cx, cy, req, dark=True) >= 0.75:
                seen.append((d, cx, cy, req))
    out += seen
    out.sort()
    return out


def check_hole(ctx, f, sim, g5, reach_k=2.5, exclude=()):
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
        fe = ctx.ray_edge_fit(c[0], c[1], 0.8 * r, 1.2 * r, n=360, floor=max(1.0, 0.004 * r))
        ok = fe["rays"] >= 0.6 * 360 and fe["std"] <= max(1.5, 0.03 * r)  # a bore's inner wall adds edge scatter
        core0 = ctx.disc_stat(c[0], c[1], 0, 0.5 * r)
        sur0 = ctx.disc_stat(c[0], c[1], 1.15 * r, 1.4 * r)
        if not ok and abs(core0 - sur0) < C["image"]["min_contrast"]:
            out.update(status="MISSING", note=f"no edge and no contrast at the expected place "
                                              f"({abs(core0 - sur0):.0f} gray levels)")
            return out
        out.update(meas_x=fe["x"], meas_y=fe["y"], meas_r=fe["r"], fit_std=fe["std"], quality_ok=bool(ok))
        return out
    core = ctx.disc_stat(c[0], c[1], 0, max(1.5, 0.5 * r))
    sur = ctx.disc_stat(c[0], c[1], 1.4 * r, 1.9 * r)
    searched = abs(core - sur) < C["image"]["min_contrast"]
    c_exp = c.copy()

    def round_edge():
        """No dark hole: is there a round edge of about the right size (pin, boss, plug, spot face)?"""
        fe = ctx.ray_edge_fit(c_exp[0], c_exp[1], 0.75 * r, 1.25 * r, n=120, floor=0.7)
        if fe["rays"] >= 0.6 * 120 and fe["std"] <= max(0.7, 0.06 * r) and \
                math.hypot(fe["x"] - c_exp[0], fe["y"] - c_exp[1]) <= 0.5 * r and 0.75 <= fe["r"] / r <= 1.25:
            out.update(meas_x=fe["x"], meas_y=fe["y"], meas_r=fe["r"], fit_std=fe["std"], quality_ok=True,
                       seen="round edge", note="round edge found but not a dark hole: pin, boss, plug or spot face")
            return True
        return False

    # Prefer a real hole-shaped spot: the nearest compact round spot of about this size within 1.25 hole
    # diameters (far enough to catch a hole shifted by parallax, short of the neighbouring holes).
    cands = _hole_candidates(ctx, c_exp, r, reach_k * r)
    # skip spots that are another drawing hole's place; rank by core darkness with a mild distance penalty
    cands = [q for q in cands if all(math.hypot(q[1] - ex, q[2] - ey) > 0.8 * er for ex, ey, er in exclude)]
    if reach_k > 2.5 and len(cands) > 1:
        # A feature lower than the fitted face (recess floor) is further from the camera, so it appears
        # shifted toward the image centre (principal point). Prefer the spot whose shift points that way.
        toc = np.array([ctx.W / 2 - c_exp[0], ctx.H / 2 - c_exp[1]])
        toc /= max(np.linalg.norm(toc), 1e-9)

        def off_axis(q):
            v = np.array([q[1] - c_exp[0], q[2] - c_exp[1]])
            n = np.linalg.norm(v)
            return 0.0 if n <= 0.5 * r else math.degrees(math.acos(float(np.clip(v @ toc / n, -1, 1))))
        cands.sort(key=lambda q: (off_axis(q) > 45, off_axis(q) if q[0] > 0.5 * r else 0.0, q[0]))
    if cands and (searched or cands[0][0] > 0.5 * r):
        d0, cx_, cy_, _ = cands[0]
        c = np.array([cx_, cy_])
        core = ctx.disc_stat(c[0], c[1], 0, max(1.5, 0.5 * r))
        sur = ctx.disc_stat(c[0], c[1], 1.4 * r, 1.9 * r)
        h = measure_round_hole(ctx, c[0], c[1], r, -1.0 if core < sur else 1.0, g5)
        if h is not None and 0.5 <= h["r"] / r <= 1.6 and math.hypot(h["x"] - c[0], h["y"] - c[1]) <= 0.6 * r:
            out.update(meas_x=h["x"], meas_y=h["y"], meas_r=h["r"], fit_std=h["fit_std"], quality_ok=h["quality_ok"])
            if d0 > 0.5 * r:
                out["note"] = f"hole found {d0 / r * 0.5:.1f} diameters from the drawing position"
            return out
        if d0 > 0.5 * r:
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
        for dx in np.linspace(-r, r, 7):
            for dy in np.linspace(-r, r, 7):
                cc = ctx.disc_stat(c[0] + dx, c[1] + dy, 0, max(1.5, 0.5 * r))
                ss = ctx.disc_stat(c[0] + dx, c[1] + dy, 1.4 * r, 1.9 * r)
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
    bad = h is None or not 0.5 <= h["r"] / r <= 1.6
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
    size = max(w, h)
    m = int(0.4 * size) + 4
    xa, ya, xb, yb = max(x - m, 0), max(y - m, 0), min(x + w + m, ctx.W), min(y + h + m, ctx.H)
    exp = np.zeros((yb - ya, xb - xa), np.uint8)
    cv2.fillPoly(exp, [(poly - [xa, ya]).astype(np.int32)], 1)
    out = dict(exp_poly=poly)
    if min(w, h) < 4:
        out.update(status="CAN'T VERIFY", note=f"only {min(w, h)} px wide in the photo")
        return out
    g = ctx.gray[ya:yb, xa:xb].astype(np.float32)
    k = max(1, int(0.12 * min(w, h)))
    inner = cv2.erode(exp, np.ones((2 * k + 1,) * 2, np.uint8))
    ring = cv2.dilate(exp, np.ones((4 * k + 1,) * 2, np.uint8)) - cv2.dilate(exp, np.ones((2 * k + 1,) * 2, np.uint8))
    if inner.sum() < 5 or ring.sum() < 5:
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
               iou=iou, meas_contour=(c[:, 0, :] + [xa, ya]).tolist())
    return out


def _normals(poly):
    """Unit normals pointing out of the closed polygon."""
    p = np.asarray(poly, float)
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    n = np.c_[t[:, 1], -t[:, 0]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    c = p.astype(np.float32).reshape(-1, 1, 2)
    k = max(1, len(p) // 60)
    probe = p[::k] + n[::k] * 1.5
    outside = np.mean([cv2.pointPolygonTest(c, (float(x), float(y)), False) < 0 for x, y in probe])
    return n if outside >= 0.5 else -n


def inner_limits(poly_px, circles_px, margin=1.3):
    """Per outline point: how far inward the edge search may go before reaching a drawing hole
    (hole centre distance - margin x radius). Keeps the outline from snapping onto nearby hole rims."""
    lim = np.full(len(poly_px), np.inf)
    for cx, cy, r in circles_px:
        lim = np.minimum(lim, np.hypot(poly_px[:, 0] - cx, poly_px[:, 1] - cy) - margin * r)
    return np.maximum(lim, 2.0)


def edge_offsets(ctx, poly_px, band, sign=None, inner_lim=None):
    """Strongest edge along the outward normal at each point of the expected contour.
    Returns offset in px (+ = edge lies outside the expected line) and edge strength (gray levels / px).
    sign=+1 when the image gets lighter going outward (dark part on light background), -1 the reverse,
    None for either direction."""
    n = _normals(poly_px)
    rs = np.arange(-band, band + 0.01, 0.5, dtype=np.float32)
    P = np.asarray(poly_px, np.float32)[:, None, :] + rs[None, :, None] * n[:, None, :].astype(np.float32)
    prof = cv2.remap(ctx.gs, np.ascontiguousarray(P[..., 0]), np.ascontiguousarray(P[..., 1]), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    prof = cv2.GaussianBlur(prof, (5, 1), 0)
    g = np.gradient(prof, axis=1) / 0.5
    if inner_lim is not None:  # positions further inward than allowed cannot be the outline edge
        g = np.where(rs[None, :] < -np.asarray(inner_lim, np.float32)[:, None], 0.0, g)
    a = np.arange(len(g))
    if sign is None:
        i = np.abs(g).argmax(1)
        return rs[i], np.abs(g)[a, i]
    i = (sign * g).argmax(1)
    return rs[i], (sign * g)[a, i]


def _blobs(ctx, rmin, rmax, box, k_ref=None):
    """Round dark (blind/through, dark inside) and light (background seen through) spots: hole candidates.
    The local background is a median over ~6 hole radii; sized from the typical hole (k_ref), not the
    largest, so shadows and dark recesses nearby do not swamp small holes."""
    x0, y0, x1, y1 = box
    g = ctx.gray[y0:y1, x0:x1]
    k = odd(min(201, max(15, 6 * (k_ref or rmax))))
    med = cv2.medianBlur(g, k).astype(np.float32)
    out = []
    for dark, D in ((1.0, med - g.astype(np.float32)), (0.0, g.astype(np.float32) - med)):
        b = (D > C["image"]["spot_contrast"]).astype(np.uint8)
        n, lab, st, ce = cv2.connectedComponentsWithStats(b)
        for i in range(1, n):
            a = st[i, 4]
            r = math.sqrt(a / math.pi)
            w, h = st[i, 2], st[i, 3]
            if not (0.6 * rmin <= r <= 1.4 * rmax) or max(w, h) > 1.6 * min(w, h) or a < 0.5 * w * h:
                continue
            out.append((ce[i, 0] + x0, ce[i, 1] + y0, r, float(D[lab == i].mean()), dark))
    out.sort(key=lambda b: -b[3])
    return np.array(out[:40], float).reshape(-1, 5)  # x, y, r, strength, dark(1)/light(0)


def hole_hypotheses(ctx, model, s0, mirrors, fixed_scale, k_best=3):
    """Alignments from pairs of drawing holes matched to pairs of round spots in the photo. Independent of
    the outline, so it still works when reflections or shadows spoil the part's silhouette."""
    allh = sorted([f for f in model["features"] if f["kind"] == "hole"], key=lambda f: -f["d"])
    holes = allh[:8] + [f for f in allh[8:] if all(abs(f["d"] - g["d"]) > 0.3 for g in allh[:8])][:4]
    if len(holes) < 2:
        return []
    Hc = np.array([[f["cx"], f["cy"]] for f in holes])
    Hr = np.array([f["d"] / 2 for f in holes])
    smin, smax = (0.97 * fixed_scale, 1.03 * fixed_scale) if fixed_scale else (0.6 * s0, 1.5 * s0)
    bx, by, bw, bh = ctx.part["bbox"]
    m = int(0.3 * max(bw, bh))
    box = (max(bx - m, 0), max(by - m, 0), min(bx + bw + m, ctx.W), min(by + bh + m, ctx.H))
    small = Hr[Hr < 0.1 * model["outline"]["size"]]  # bolt/fixing holes, not the bore
    k_ref = float(1.5 * small.max() * s0) if len(small) else None  # largest fixing hole (+50%), never the bore
    B = _blobs(ctx, max(2.0, smin * Hr.min()), smax * Hr.max(), box, k_ref)
    if len(B) < 2:
        return []
    gb = cv2.GaussianBlur(ctx.gray, (5, 5), 0).astype(np.float32)
    G = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    gref = float(np.percentile(G[box[1]:box[3], box[0]:box[2]], 95)) + 1e-6
    opts = _resample(model["outline"]["pts"], 160)
    hyps = []
    for i, j in itertools.combinations(range(len(holes)), 2):
        dd = np.linalg.norm(Hc[j] - Hc[i])
        if dd < 1e-6:
            continue
        for a, b in itertools.permutations(range(len(B)), 2):
            sc = np.linalg.norm(B[b, :2] - B[a, :2]) / dd
            if not smin <= sc <= smax:
                continue
            if abs(B[a, 2] - sc * Hr[i]) > 0.5 * sc * Hr[i] + 2 or abs(B[b, 2] - sc * Hr[j]) > 0.5 * sc * Hr[j] + 2:
                continue
            for mir in mirrors:
                M = np.diag([1.0, -1.0]) if mir else np.eye(2)
                vd, vi = M @ (Hc[j] - Hc[i]), B[b, :2] - B[a, :2]
                th = math.atan2(vi[1], vi[0]) - math.atan2(vd[1], vd[0])
                A = sc * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) @ M
                sim = Sim(sc, th, B[a, :2] - A @ Hc[i], mir)
                P = sim(Hc)
                dist = np.hypot(P[:, None, 0] - B[None, :, 0], P[:, None, 1] - B[None, :, 1])
                nb = dist.argmin(1)
                # loose on purpose (ranking only): a tilted photo can put holes ~1 radius off a flat fit
                match = (dist[np.arange(len(P)), nb] <= np.maximum(3.0, 1.0 * sc * Hr)) & \
                        (np.abs(B[nb, 2] - sc * Hr) <= 0.5 * sc * Hr + 2)
                q = sim(opts)
                ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < ctx.W - 1) & (q[:, 1] < ctx.H - 1)
                edge = float(np.mean(G[q[ok, 1].astype(int), q[ok, 0].astype(int)])) / gref if ok.any() else 0.0
                hyps.append((int(match.sum()) + min(edge, 1.0), sim))
    hyps.sort(key=lambda h: -h[0])
    # upgrade: a hypothesis from two holes is a flat (similarity) guess; on a tilted photo distant holes sit
    # off it. Fit a homography to the holes it already matches, re-match tightly, rank by that.
    allH = sorted([f for f in model["features"] if f["kind"] == "hole"], key=lambda f: -f["d"])
    AHc = np.array([[f["cx"], f["cy"]] for f in allH])
    AHr = np.array([f["d"] / 2 for f in allH])
    ref = np.array(dwg._centroid(model["outline"]["pts"]))
    up = []
    for sc_, sim in hyps[:300]:
        T = sim
        for tol_k in (1.0, 0.6, 0.45):
            P = T(AHc)
            ls = T.s
            dist = np.hypot(P[:, None, 0] - B[None, :, 0], P[:, None, 1] - B[None, :, 1])
            nb = dist.argmin(1)
            m = (dist[np.arange(len(P)), nb] <= np.maximum(3.0, tol_k * ls * AHr)) & \
                (np.abs(B[nb, 2] - ls * AHr) <= 0.5 * ls * AHr + 2)
            if m.sum() < 6 or len(set(nb[m])) < 6:
                break
            H, inl = cv2.findHomography(AHc[m].astype(np.float32), B[nb[m], :2].astype(np.float32), cv2.RANSAC,
                                        max(3.0, 0.5 * ls * float(np.median(AHr))))
            if H is None:
                break
            try:
                Tn = Proj(H, ref)
            except np.linalg.LinAlgError:
                break
            if Tn.mirror != sim.mirror or not 0.7 < Tn.s / sim.s < 1.4:
                break
            T = Tn
        P = T(AHc)
        dist = np.hypot(P[:, None, 0] - B[None, :, 0], P[:, None, 1] - B[None, :, 1])
        nb = dist.argmin(1)
        ls = T.s
        n_ok = int(((dist[np.arange(len(P)), nb] <= np.maximum(2.0, 0.45 * ls * AHr)) &
                    (np.abs(B[nb, 2] - ls * AHr) <= 0.5 * ls * AHr + 2)).sum())
        # perspective-upgraded guesses rank first (by tight matches); the rest keep their original score
        up.append(((1, n_ok + (sc_ - int(sc_))) if isinstance(T, Proj) else (0, sc_), T))
    hyps = [(k[1], T) for k, T in sorted(up, key=lambda h: (-h[0][0], -h[0][1]))]
    out = []
    for mir in mirrors:  # the best few for EACH orientation, so one cannot crowd out the other
        kept = []
        for sc_, sim in hyps:
            if sim.mirror != mir:
                continue
            if all(abs((sim.theta - o.theta + math.pi) % (2 * math.pi) - math.pi) > math.radians(2) or
                   np.linalg.norm(sim(ref[None])[0] - o(ref[None])[0]) > 5 for o in kept):
                kept.append(sim)
            if len(kept) >= k_best:
                break
        out += kept
    return out


def _iou(ctx, sim, opts):
    bx, by, bw, bh = ctx.part["bbox"]
    m = int(0.3 * max(bw, bh))
    x0, y0 = max(bx - m, 0), max(by - m, 0)
    x1, y1 = min(bx + bw + m, ctx.W), min(by + bh + m, ctx.H)
    dm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(dm, [(sim(opts) - [x0, y0]).astype(np.int32)], 1)
    im = (ctx.part["mask_outer"][y0:y1, x0:x1] > 0).astype(np.uint8)
    return float((dm & im).sum() / max((dm | im).sum(), 1))


def refine(ctx, model, sim, dpts, fixed_scale, holes, g5, sign):
    """Fit on measured edge points along the outline plus measured hole centres; score hole agreement."""
    size_px = sim.s * model["outline"]["size"]
    band = max(4.0, 0.03 * size_px)
    ok = np.zeros(len(dpts), bool)
    openings = [f for f in model["features"] if f["kind"] == "opening" and not _encloses(f, model)]
    info = {}
    for _ in range(6):
        q = sim(dpts)
        off, st = edge_offsets(ctx, q, band, sign)
        ok = st >= max(4.0, 0.3 * float(np.median(st)))
        if ok.sum() > 20:
            med = np.median(off[ok])
            mad = np.median(np.abs(off[ok] - med))
            ok &= np.abs(off - med) <= 3 * 1.4826 * mad + 1.0
        tgt = q + off[:, None] * _normals(q)
        res = {f["id"]: check_hole(ctx, f, sim, g5) for f in holes}
        # datum features: measured hole centres + centres of plain openings (window, slot) found clearly
        src, dst = [], []
        for f in holes:
            r = res[f["id"]]
            if "meas_x" in r and r["quality_ok"]:
                src.append([f["cx"], f["cy"]])
                dst.append([r["meas_x"], r["meas_y"]])
        for f in openings:
            r = check_opening(ctx, f, sim)
            if "meas_x" in r and r["iou"] >= 0.7:
                src.append([f["cx"], f["cy"]])
                dst.append([r["meas_x"], r["meas_y"]])
        src, dst = np.array(src).reshape(-1, 2), np.array(dst).reshape(-1, 2)
        keep = np.ones(len(src), bool)
        if len(src) >= 3:
            e = np.hypot(*(sim(src) - dst).T)
            keep = e <= max(2.0, 3 * np.median(e))
        # inverse-variance weighting: each group counts by its own measured precision. Outline samples are
        # strongly correlated along the edge, so all of them together count as ~40 independent points.
        sig_o, sig_d = 1.0, 1.0
        if ok.sum() >= 20:
            so = fit_similarity(dpts[ok], tgt[ok], sim.mirror, fixed_scale=fixed_scale)
            ro = np.hypot(*(so(dpts[ok]) - tgt[ok]).T)
            sig_o = max(0.5, 1.4826 * float(np.median(ro)))
            info["scale_outline"] = so.s
        if keep.sum() >= 3:
            sd = fit_similarity(src[keep], dst[keep], sim.mirror, fixed_scale=fixed_scale)
            rd = np.hypot(*(sd(src[keep]) - dst[keep]).T)
            sig_d = max(0.5, float(np.sqrt(np.mean(rd ** 2))))
            info["scale_features"] = sd.s
        S = np.vstack([dpts[ok], src[keep]])
        D = np.vstack([tgt[ok], dst[keep]])
        if len(S) < 10 and keep.sum() < 3:
            break
        w = np.r_[np.full(ok.sum(), 40.0 / max(ok.sum(), 1) / sig_o ** 2), np.full(keep.sum(), 1.0 / sig_d ** 2)]
        info.update(outline_sigma_px=sig_o, feature_sigma_px=sig_d, n_datum=int(keep.sum()))
        sim = fit_similarity(S, D, sim.mirror, w=w, fixed_scale=fixed_scale)
        band = max(3.0, band * 0.6)
    res = {f["id"]: check_hole(ctx, f, sim, g5) for f in holes}
    hole_score = _agreement(holes, res)
    info.update(score=hole_score + 0.3 * float(ok.mean()), hole_agreement=hole_score, edge_found=float(ok.mean()),
                feature_agreement=_feature_agreement(ctx, model, sim, holes, res))
    return sim, res, info


def _hole_agree(r):
    """Agreement of one hole result in [0, 1]: closeness of the measured centre to the expected one (1 = on
    target, 1/e at matching.agreement_falloff_radii hole radii), times the evidence weight. A hole found with a
    faint or doubled wall (counterbore, chamfer, shadowed bore) counts matching.weak_hole_weight; a round edge
    found by searching at the expected place counts matching.round_edge_weight, because that search centres on
    the expected place by construction, so its position is no independent evidence."""
    m = C["matching"]
    if "meas_x" not in r or not np.isfinite(r["meas_x"]):
        return 0.0
    e = math.hypot(r["meas_x"] - r["exp_x"], r["meas_y"] - r["exp_y"]) / max(r["exp_r"], 1.0)
    w = float(m["round_edge_weight"]) if r.get("seen") == "round edge" else \
        1.0 if r.get("quality_ok") else float(m["weak_hole_weight"])
    return w * math.exp(-(e / m["agreement_falloff_radii"]) ** 2)


def _agreement(holes, res):
    """Mean hole agreement over the drawing's holes."""
    return float(np.mean([_hole_agree(res[f["id"]]) for f in holes])) if holes else 0.0


def _plain_openings(model):
    return [f for f in model["features"] if f["kind"] == "opening" and not _encloses(f, model)]


def _feature_agreement(ctx, model, sim, holes, res):
    """Alignment evidence from the inner features: hole agreement plus, for each plain opening (window, slot),
    its overlap with the opening segmented at the aligned place. Openings tell the orientation of a part whose
    silhouette is ambiguous (thick or stepped parts, where the outline is the union of both faces)."""
    vals = [_hole_agree(res[f["id"]]) for f in holes]
    for f in _plain_openings(model):
        r = check_opening(ctx, f, sim)
        vals.append(float(r.get("iou", 0.0)) if "meas_x" in r else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def perspective_refine(ctx, model, T, res, holes, g5, rounds=4):
    """A flat face photographed at an angle maps to the drawing by a homography. With 6+ measured holes,
    fit one (RANSAC) and keep it only when it brings clearly more holes onto their measured centres."""
    ref = np.array(dwg._centroid(model["outline"]["pts"]))
    best = (T, res, _agreement(holes, res))
    for _ in range(rounds):
        T_, res_, ag_ = best
        good = [f for f in holes if "meas_x" in res_[f["id"]] and res_[f["id"]].get("quality_ok")]
        if len(good) < 6:
            break
        src = np.array([[f["cx"], f["cy"]] for f in good], np.float32)
        dst = np.array([[res_[f["id"]]["meas_x"], res_[f["id"]]["meas_y"]] for f in good], np.float32)
        thr = max(2.0, 0.4 * float(np.median([res_[f["id"]]["exp_r"] for f in good])))
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, thr)
        if H is None or inl is None or inl.sum() < 6:
            break
        P = Proj(H, ref)
        if P.mirror != T.mirror or not 0.7 < P.s / T.s < 1.4:
            break
        res2 = {f["id"]: check_hole(ctx, f, P, g5) for f in holes}
        ag2 = _agreement(holes, res2)
        if ag2 > ag_ + 0.02:
            best = (P, res2, ag2)
        else:
            break
    return best


def check_contour(ctx, f, sim, band):
    """Edge-based check of an inner contour (recess/step/opening edge) along its normals."""
    poly = sim(_resample(f["pts"], 400))
    off, st = edge_offsets(ctx, poly, band, None)
    found = st >= max(6.0, 0.3 * float(np.median(st)))
    return dict(exp_poly=poly, off=off, found=found, meas=(poly + off[:, None] * _normals(poly)))


@step("drawing_match", needs=("part",))
def drawing_match(ctx):
    """Register the drawing (--drawing DXF/JSON) to the photo and check every drawing feature."""
    path = ctx.opt_get("drawing")
    if not path:
        raise SystemExit("drawing_match needs --drawing FILE.dxf or FILE.json")
    tol = float(ctx.opt_get("tol") or C["tolerance"]["default_mm"])
    model, notes, info = dwg.load(path, tol, ctx.opt_get("pos_tol") or C["tolerance"]["position_mm"])
    for n in notes:
        ctx.note(n)
    fixed = ctx.opt_get("mm_per_px")
    fixed_scale = 1.0 / float(fixed) if fixed else None
    g5 = cv2.GaussianBlur(ctx.gray, (odd(max(3, 5 / ctx.sN)),) * 2, 0)
    holes = [f for f in model["features"] if f["kind"] == "hole"]
    sign = 1.0 if ctx.part["polarity"].startswith("dark") else -1.0
    mopt = ctx.opt_get("mirror", "auto") or "auto"
    cands = []
    mirrors = {"auto": ("no", "yes"), "no": ("no",), "yes": ("yes",)}[mopt]
    starts = []
    for m in mirrors:  # (a) outline-overlap search
        sim0, reg0, dpts = register(ctx, model, fixed_scale, m)
        starts.append((sim0, dict(reg0, start="outline overlap search")))
    # (b) hole-pair matching: independent of the silhouette
    for sim0 in hole_hypotheses(ctx, model, starts[0][0].s, [m == "yes" for m in mirrors], fixed_scale):
        starts.append((sim0, dict(starts[0][1], start="hole-pair match")))
    # With few holes, "holes agree" is weak evidence (2 holes always fit some rotation + scale): shift the
    # weight to the outline overlap. Full hole weight from 6 holes up.
    n_feat = len(holes) + len(_plain_openings(model))
    w_h = min(1.0, max(0.25, (n_feat - 2) / 4))

    def cand_score(ag, ef, iou):
        return w_h * ag + 0.3 * ef + (0.2 + 0.6 * (1 - w_h)) * iou

    for sim0, reg0 in starts:
        if isinstance(sim0, Proj):
            if fixed_scale:
                continue
            res0 = {f["id"]: check_hole(ctx, f, sim0, g5) for f in holes}
            T, res_p, ag = perspective_refine(ctx, model, sim0, res0, holes, g5)
            q0 = T(dpts)
            off0, st0 = edge_offsets(ctx, q0, max(4.0, 0.02 * T.s * model["outline"]["size"]), sign)
            ef = float((st0 >= max(4.0, 0.3 * float(np.median(st0)))).mean())
            reg0 = dict(reg0, coarse_iou=_iou(ctx, T, model["outline"]["pts"]), start="hole-pair match + perspective")
            fa = _feature_agreement(ctx, model, T, holes, res_p)
            cands.append((cand_score(fa, ef, reg0["coarse_iou"]), T, reg0, res_p,
                          dict(score=ag + 0.3 * ef, hole_agreement=ag, feature_agreement=fa, edge_found=ef,
                               perspective=True)))
            continue
        simr, res_h, rinfo = refine(ctx, model, sim0, dpts, fixed_scale, holes, g5, sign)
        if len(holes) >= 6 and not fixed_scale:
            T, res_p, ag = perspective_refine(ctx, model, simr, res_h, holes, g5)
            if isinstance(T, Proj):
                simr, res_h = T, res_p
                rinfo = dict(rinfo, hole_agreement=ag, score=ag + 0.3 * rinfo["edge_found"], perspective=True,
                             feature_agreement=_feature_agreement(ctx, model, T, holes, res_p))
        reg0 = dict(reg0, coarse_iou=_iou(ctx, simr, model["outline"]["pts"]))
        cands.append((cand_score(rinfo["feature_agreement"], rinfo["edge_found"], reg0["coarse_iou"]), simr, reg0,
                      res_h, rinfo))
    # identical solutions found from different starts: keep one of each (they do not count as ambiguity)
    uniq = []
    for c in sorted(cands, key=lambda c: -c[0]):
        rp = np.array(dwg._centroid(model["outline"]["pts"]))[None]
        if all(c[1].mirror != u[1].mirror or abs((c[1].theta - u[1].theta + math.pi) % (2 * math.pi) - math.pi) >
               math.radians(1.5) or np.linalg.norm(c[1](rp)[0] - u[1](rp)[0]) > 3 for u in uniq):
            uniq.append(c)
    cands = uniq
    cands.sort(key=lambda c: -c[0])
    if len(cands) >= 2 and cands[0][1].mirror and not cands[1][1].mirror and cands[1][0] >= 0.98 * cands[0][0]:
        cands[0], cands[1] = cands[1], cands[0]  # a symmetric part fits both ways: report it as drawn
    _, sim, reg, res, rinfo = cands[0]
    reg.update(rinfo)
    if rinfo.get("scale_outline") and rinfo.get("scale_features"):
        ratio = rinfo["scale_features"] / rinfo["scale_outline"]
        reg["feature_to_outline_scale_ratio"] = ratio
        if abs(ratio - 1) > 0.03:
            ctx.warn(f"The holes/openings and the outline imply scales {100 * (ratio - 1):+.1f}% apart: the features "
                     "and the outline are probably at different heights (stepped part or stacked plates) seen in "
                     "perspective from a close camera, or the drawing does not match the part. The fit follows the "
                     "more precise group; deviations of the other group partly reflect this. A photo from further "
                     "away (zoomed in) reduces the effect.")
    reg["candidates"] = [dict(start=c[2].get("start"), mirrored=c[1].mirror, angle_deg=round(math.degrees(c[1].theta) % 360, 2),
                              px_per_mm=round(c[1].s, 4), score=round(c[0], 3), hole_agreement=round(c[4]["hole_agreement"], 3),
                              feature_agreement=round(c[4]["feature_agreement"], 3))
                         for c in cands]
    if len(cands) >= 2:
        if 0.9 * cands[0][0] < cands[1][0] < 0.98 * cands[0][0]:
            ctx.warn("Two different alignments fit almost equally well (see registration.candidates in the "
                     "report); check the overlay, or force the orientation with --mirror yes/no.")

    s = sim.s  # px per mm
    mm = 1.0 / s
    ctx.mm = mm
    q = sim(dpts)
    if isinstance(sim, Proj):
        reg["transform_px_from_mm"] = dict(H=sim.H.tolist(), usage="homography: [u,v,w] = H @ [x_mm, y_mm, 1], "
                                           "x_px = u/w, y_px = v/w  (drawing y down)")
        ctx.note("The photo was taken at an angle: the drawing was mapped with a perspective (homography) fit "
                 "on the measured holes instead of a flat rotation + scale.")
    else:
        reg["transform_px_from_mm"] = dict(A=sim.A.tolist(), t=sim.t.tolist(),
                                           usage="x_px = A @ [x_mm, y_mm] + t  (drawing y down)")
    reg.update(px_per_mm=s, mm_per_px=mm, angle_deg=math.degrees(sim.theta) % 360, mirror=sim.mirror,
               scale_source="fixed (--mm-per-px)" if fixed else "fitted to the drawing")
    ctx.data["drawing"] = dict(file=path, info=info, registration=reg,
                               outline_size_mm=[float(np.ptp(model["outline"]["pts"][:, 0])),
                                                float(np.ptp(model["outline"]["pts"][:, 1]))])
    # Outline at a different height from the holes/openings (stepped or stacked part, close camera): one
    # 2-D mapping cannot fit both. If aligning the outline on its own fits its edge clearly better, check the
    # outline's SHAPE with that alignment; holes and openings keep the joint alignment.
    band0 = max(4.0, 0.03 * s * model["outline"]["size"])
    hole_px = [(*sim(np.array([[g_["cx"], g_["cy"]]]))[0], g_["d"] / 2 * sim.local_scale(g_["cx"], g_["cy"]))
               for g_ in model["features"] if g_["kind"] == "hole"]

    def _p95(qq):
        o_, s_ = edge_offsets(ctx, qq, band0, sign, inner_limits(qq, hole_px))
        k_ = s_ >= max(4.0, 0.3 * float(np.median(s_)))
        return (float(np.percentile(np.abs(o_[k_]), 95)) if k_.sum() > 20 else 99.0), o_, s_, k_
    p_joint, o_j, s_j, k_j = _p95(q)
    own = None
    if k_j.sum() > 20 and not fixed:
        cur = fit_similarity(dpts[k_j], (q + o_j[:, None] * _normals(q))[k_j], sim.mirror)
        for _ in range(6):
            qq = cur(dpts)
            o_, s_ = edge_offsets(ctx, qq, max(3.0, 0.5 * band0), sign, inner_limits(qq, hole_px))
            k_ = s_ >= max(4.0, 0.3 * float(np.median(s_)))
            if k_.sum() < 20:
                break
            cur = fit_similarity(dpts[k_], (qq + o_[:, None] * _normals(qq))[k_], sim.mirror)
        p_own = _p95(cur(dpts))[0]
        # second candidate: fit to the segmented silhouette (robust where a raised plate's edges distract)
        cont_ = ctx.part["contour"][:, 0, :].astype(float)
        sil = fit_similarity(dpts, q, sim.mirror)
        for _ in range(15):
            qq = sil(dpts)
            idx_, dist_ = _nearest(qq, cont_)
            kk = dist_ < max(2.0, 2.5 * np.median(dist_))
            if kk.sum() < 20:
                break
            sil = fit_similarity(dpts[kk], cont_[idx_[kk]], sim.mirror)
        p_sil = _p95(sil(dpts))[0]
        if p_sil < p_own:
            cur, p_own = sil, p_sil
        reg.update(outline_p95_px_joint=p_joint, outline_p95_px_own=p_own, outline_own_scale=cur.s)
        if p_own < 0.85 * p_joint - 0.5:
            own = cur
            q = cur(dpts)
            ctx.warn(f"The outline and the holes/openings do not fit one flat mapping (outline-only fit: scale "
                     f"{100 * (cur.s / s - 1):+.1f}%, edge error {p_own / cur.s:.2f} mm vs {p_joint / s:.2f} mm with the "
                     "holes): they are at different heights (stepped or stacked part) or the camera is close and "
                     "angled. The outline SHAPE is checked with its own alignment; its position relative to the "
                     "holes is not checked from this photo.")
    reg["outline_alignment"] = "own (shape check)" if own else "joint with holes/openings"
    # outline: measured edge along each normal of the drawing outline (+ = part larger than drawing)
    off_o, st_o_ = edge_offsets(ctx, q, max(4.0, 0.03 * s * model["outline"]["size"]), sign, inner_limits(q, hole_px))
    found_o = st_o_ >= max(4.0, 0.3 * float(np.median(st_o_)))
    sd = off_o * mm
    fo = off_o[found_o]
    rms_px = float(np.sqrt(np.mean((fo - np.median(fo)) ** 2))) if len(fo) else 99.0
    reg.update(outline_rms_px=rms_px, outline_rms_mm=rms_px * mm, outline_edge_found=float(found_o.mean()))
    unc_edge = math.sqrt(1.0 + rms_px ** 2) * mm  # edges/contours: 1 px + scatter of the outline edge
    # holes: size is a direct 1 px edge measurement; position adds the scatter of hole centres about the fit
    e_h = [math.hypot(r["meas_x"] - r["exp_x"], r["meas_y"] - r["exp_y"]) for r in res.values()
           if "meas_x" in r and r.get("quality_ok")]
    # robust: 2-D scatter from the median distance (rms ~ 1.2 x median), so a misplaced hole cannot hide itself
    hole_rms_px = float(1.2 * np.median(e_h)) if len(e_h) >= 3 else rms_px
    unc_pos = math.sqrt(1.0 + hole_rms_px ** 2) * mm
    unc_size = 1.0 * mm
    unc = unc_edge
    reg.update(uncertainty_edge_mm=unc_edge, uncertainty_hole_position_mm=unc_pos, uncertainty_hole_size_mm=unc_size,
               hole_centre_rms_px=hole_rms_px)

    rows, results = [], []

    def verdict(dev_pos, dev_size, f, extra_bad=False, u_pos=None, u_size=None):
        """Guard band: OK only if clearly inside tolerance, DEVIATES if clearly outside,
        CAN'T VERIFY when the deviation is within +-uncertainty of the limit."""
        u_pos = unc_edge if u_pos is None else u_pos
        u_size = unc_edge if u_size is None else u_size
        worst, close, near = False, False, []
        for dev, lim, u in ((dev_pos, f["pos_tol"], u_pos),
                            (None if dev_size is None else abs(dev_size), f["tol"], u_size)):
            if dev is None:
                continue
            if dev > lim + u:
                worst = True
            elif dev > lim - u:
                close = True
                near.append(u)
        if worst or extra_bad:
            return "DEVIATES", ""
        if close:
            return "CAN'T VERIFY", f"within +-{max(near):.2f} mm (photo uncertainty) of the tolerance limit"
        return "OK", ""

    det_pos = float(ctx.opt_get("detect_pos_tol") or C["tolerance"]["detect_position_mm"])
    det_size = float(ctx.opt_get("detect_size_tol") or C["tolerance"]["detect_size_mm"])
    reg.update(detection_tolerance=dict(position_mm=f"max({det_pos}, 0.5 x nominal size)",
                                        size_mm=f"max({det_size}, 0.5 x nominal size)"))

    # Features inside a recess contour sit at a different height from the face the drawing was fitted on;
    # in an angled photo they shift sideways (parallax), so allow them a larger detection distance.
    recess = [g for g in model["features"] if g["kind"] == "opening" and _encloses(g, model)]
    on_recess = set()
    for g in recess:
        poly = g["pts"].astype(np.float32).reshape(-1, 1, 2)
        for f_ in model["features"]:
            if f_ is not g and f_["kind"] == "hole" and cv2.pointPolygonTest(poly, (float(f_["cx"]), float(f_["cy"])), False) > 0:
                on_recess.add(f_["id"])
    reg["features_inside_recess"] = sorted(on_recess)
    if on_recess and isinstance(sim, Proj):
        others = {g["id"]: (*sim(np.array([[g["cx"], g["cy"]]]))[0], g["d"] / 2 * sim.local_scale(g["cx"], g["cy"]))
                  for g in holes}
        for g in holes:
            if g["id"] in on_recess and g["d"] / 2 * sim.local_scale(g["cx"], g["cy"]) < 20:
                ex = [v for k_, v in others.items() if k_ != g["id"] and k_ not in on_recess]
                res[g["id"]] = check_hole(ctx, g, sim, g5, reach_k=4.5, exclude=ex)

    def detected(dpos_v, dsize_v, size, fid_=None):
        """Present near where the drawing puts it, at roughly the right size (detection tolerance)."""
        lim = max(det_pos, 0.5 * size) if fid_ not in on_recess else max(3 * det_pos, 1.5 * size)
        return (dpos_v is None or dpos_v <= lim) and \
               (dsize_v is None or abs(dsize_v) <= max(det_size, 0.5 * size))

    def combine(st_dim, det):
        """Main status: OK = detected and within the dimensional tolerance; DETECTED = present but outside
        (or not verifiable against) the dimensional tolerance; DEVIATES = outside the detection tolerance."""
        if st_dim == "OK":
            return "OK"
        if det:
            return "DETECTED"
        if st_dim in ("MISSING", "NOT MEASURED") or (st_dim == "UNCERTAIN" and det is None):
            return st_dim
        return "DEVIATES"

    # outline
    otol = model["outline"]["tol"]
    out_bad = found_o & (np.abs(sd) > otol + C["tolerance"]["outline_guard_px"] * mm)  # edges come in 0.5 px steps
    if found_o.mean() < 0.6:
        st_o, note_o = "UNCERTAIN", f"edge visible on only {found_o.mean():.0%} of the outline"
    else:
        st_o, note_o = verdict(float(np.percentile(np.abs(sd[found_o]), 95)), None, dict(tol=otol, pos_tol=otol))
    dim_o = st_o
    det_o = bool(found_o.mean() >= 0.6 and np.median(np.abs(sd[found_o])) <= max(det_pos, 1.0)) if found_o.any() else False
    st_o = combine(dim_o, det_o)
    sd = np.where(found_o, sd, 0.0)
    rows.append(["outline", "outline", 1, "", "", f"{np.ptp(model['outline']['pts'][:, 0]):.2f} x "
                 f"{np.ptp(model['outline']['pts'][:, 1]):.2f}", "", "", "", "", f"rms {np.sqrt(np.mean(sd ** 2)):.3f}",
                 otol, st_o, (note_o + "; " if note_o else "") + f"edge deviation {sd.max():+.3f} / {sd.min():+.3f} mm "
                 f"(+ = part larger than drawing); {out_bad.mean():.0%} of the edge outside tolerance, "
                 f"{1 - found_o.mean():.0%} not visible", dim_o, "yes" if det_o else "no"])
    results.append(dict(id="outline", status=st_o, dimension_check=dim_o, detected=det_o, rms_mm=float(np.sqrt(np.mean(sd ** 2))), max_excess_mm=float(sd.max()),
                        max_shortfall_mm=float(sd.min()), fraction_out=float(out_bad.mean())))
    ctx.draw("polyline", z=1, points=q.tolist(), color=CYAN, width=1, closed=True)
    for a, b_ in _runs(out_bad):
        seg = q[np.arange(a, b_) % len(q)]
        if len(seg) > 1:
            ctx.draw("polyline", z=2, points=seg.tolist(), color=RED, width=3)

    for f in model["features"]:
        fid = f["id"]
        if f["kind"] == "hole":
            r = res[fid]
            st, note = r.get("status"), r.get("note", "")
            mx = my = md = dpos = dd = ""
            det = None
            if "meas_x" in r:
                ls = sim.local_scale(f["cx"], f["cy"])  # px per mm at this hole (varies under perspective)
                dpos_v = math.hypot(r["meas_x"] - r["exp_x"], r["meas_y"] - r["exp_y"]) / ls
                dd_v = 2 * r["meas_r"] / ls - f["d"]
                st, note = verdict(dpos_v, dd_v, f, u_pos=unc_pos, u_size=unc_size)
                if r.get("seen") == "round edge":
                    note = (note + "; " if note else "") + r["note"]
                if not r["quality_ok"]:
                    st, note = "UNCERTAIN", "weak wall fit; " + note
                det = detected(dpos_v, dd_v, f["d"], fid)
                parallax = fid in on_recess and isinstance(sim, Proj) and dpos_v > max(det_pos, 0.5 * f["d"])
                if parallax:
                    # on the recess floor, seen at an angle: its image position is shifted by parallax, so only
                    # its size can be checked from this photo
                    st, note = verdict(None, dd_v, f, u_size=unc_size)
                    note = (note + "; " if note else "") + (f"on the recess/pocket floor (lower than the fitted face): "
                            f"appears {dpos_v:.1f} mm from the drawing position because the photo is taken at an angle "
                            "(parallax); position not checkable from this photo, size checked")
                mxy = sim.inverse([[r["meas_x"], r["meas_y"]]])[0]
                mx, my, md = f"{mxy[0]:.3f}", f"{mxy[1]:.3f}", f"{2 * r['meas_r'] / ls:.3f}"
                dpos, dd = f"{dpos_v:.3f}", f"{dd_v:+.3f}"
            dim = st
            st = combine(dim, det)
            par = "meas_x" in r and "parallax" in note
            if par:
                st = "DETECTED"
                dim = f"size {dim.lower()}, position n/a" if dim in ("OK", "DEVIATES", "CAN'T VERIFY") else dim
            if "meas_x" in r:
                ctx.draw("circle", z=4, x=r["meas_x"], y=r["meas_y"], r=r["meas_r"], color=STATUS_COL[st], width=2.5)
                if par:
                    ctx.draw("line", z=4, points=[[r["exp_x"], r["exp_y"]], [r["meas_x"], r["meas_y"]]],
                             color=STATUS_COL[st], width=1.5)
            ctx.draw("circle", z=3, x=r["exp_x"], y=r["exp_y"], r=r["exp_r"], color=CYAN, width=1)
            lab_ = f"{fid} {st if st != 'OK' else ''}".strip()
            if "meas_x" in r and "parallax" in note:
                lab_ = f"{fid} on pocket floor"
            lx_, ly_ = (r["meas_x"], r["meas_y"] - r["meas_r"]) if "meas_x" in r and "parallax" in note else \
                (r["exp_x"], r["exp_y"] - r["exp_r"])
            ctx.draw("text", z=6, x=lx_, y=ly_ - 0.025 * ctx.part["r"], text=lab_, color=STATUS_COL[st], size=12)
            rows.append([fid, "pin/boss? (round edge)" if r.get("seen") == "round edge" else "hole", f["rank"], f"{f['cx']:.3f}", f"{f['cy']:.3f}", f"{f['d']:.3f}", mx, my, md,
                         dpos, dd, f"{f['tol']}/{f['pos_tol']}", st, note, dim,
                         "" if det is None else ("yes" if det else "no")])
            results.append(dict(id=fid, kind="hole", status=st, dimension_check=dim, detected=det, nominal=dict(x=f["cx"], y=f["cy"], d=f["d"]),
                                position_dev_mm=float(dpos) if dpos else None, diameter_dev_mm=float(dd) if dd else None,
                                note=note))
        elif f["kind"] == "opening" and (_encloses(f, model) or check_opening(ctx, f, sim).get("status") == "MISSING"):
            r = check_contour(ctx, f, sim, max(4.0, 0.03 * s * model["outline"]["size"]))
            dev = r["off"] * mm
            fr = float(r["found"].mean())
            if fr < 0.5:
                st, note = "UNCERTAIN", f"edge visible on only {fr:.0%} of the contour"
            else:
                st, note = verdict(float(np.percentile(np.abs(dev[r["found"]]), 90)), None, f)
            dim = st
            det = bool(fr >= 0.5 and np.median(np.abs(dev[r["found"]])) <= max(det_pos, 1.0)) if r["found"].any() else False
            st = combine(dim, det)
            note = (note + "; " if note else "") + (f"inner contour (recess/step edge): edge found on {fr:.0%}, "
                    f"median offset {np.median(dev[r['found']]) if r['found'].any() else 0:+.2f} mm")
            ctx.draw("polyline", z=3, points=r["exp_poly"].tolist(), color=CYAN, width=1, closed=True)
            mpts = r["meas"][r["found"]]
            if len(mpts) > 2:
                for i0, i1 in _runs(r["found"]):
                    seg = r["meas"][i0:i1]
                    if len(seg) > 1:
                        ctx.draw("polyline", z=4, points=seg.tolist(), color=STATUS_COL[st], width=2)
            ce = r["exp_poly"][int(np.argmin(r["exp_poly"][:, 1]))]
            ctx.draw("text", z=6, x=float(ce[0]), y=float(ce[1]), text=f"{fid} {st if st != 'OK' else ''}".strip(),
                     color=STATUS_COL[st], size=12, dy=-10)
            rows.append([fid, f"contour ({f['shape']})", f["rank"], f"{f['cx']:.3f}", f"{f['cy']:.3f}",
                         f"{f['length']:.2f} x {f['width']:.2f}", "", "", "", "", "",
                         f"{f['tol']}/{f['pos_tol']}", st, note, dim, "yes" if det else "no"])
            results.append(dict(id=fid, kind="contour", status=st, dimension_check=dim, detected=det, note=note, edge_found=fr))
        elif f["kind"] == "opening":
            r = check_opening(ctx, f, sim)
            st, note = r.get("status"), r.get("note", "")
            mx = my = mL = dpos = ""
            det = None
            if "meas_x" in r:
                c_exp = sim(np.array([[f["cx"], f["cy"]]]))[0]
                ls = sim.local_scale(f["cx"], f["cy"])
                dpos_v = math.hypot(r["meas_x"] - c_exp[0], r["meas_y"] - c_exp[1]) / ls
                dL, dW = (r["meas_L"] / ls - f["length"]), (r["meas_W"] / ls - f["width"])
                st, note = verdict(dpos_v, max(dL, dW, key=abs), f, extra_bad=r["iou"] < 0.8)
                det = r["iou"] >= 0.5 and detected(dpos_v, max(dL, dW, key=abs), min(f["length"], f["width"]))
                note = (note + "; " if note else "") + f"overlap with drawing {r['iou']:.0%}; L {dL:+.2f} W {dW:+.2f} mm"
                mxy = sim.inverse([[r["meas_x"], r["meas_y"]]])[0]
                mx, my = f"{mxy[0]:.3f}", f"{mxy[1]:.3f}"
                mL = f"{r['meas_L'] / ls:.2f} x {r['meas_W'] / ls:.2f}"
                dpos = f"{dpos_v:.3f}"
            dim = st
            st = combine(dim, det)
            if "meas_x" in r:
                ctx.draw("polyline", z=4, points=r["meas_contour"], color=STATUS_COL[st], width=2.5, closed=True)
            ctx.draw("polyline", z=3, points=r["exp_poly"].tolist(), color=CYAN, width=1, closed=True)
            ce = sim(np.array([[f["cx"], f["cy"]]]))[0]
            ctx.draw("text", z=6, x=ce[0], y=ce[1], text=f"{fid} {st if st != 'OK' else ''}".strip(),
                     color=STATUS_COL[st], size=12)
            rows.append([fid, f"opening ({f['shape']})", f["rank"], f"{f['cx']:.3f}", f"{f['cy']:.3f}",
                         f"{f['length']:.2f} x {f['width']:.2f}", mx, my, mL, dpos, "",
                         f"{f['tol']}/{f['pos_tol']}", st, note, dim, "" if det is None else ("yes" if det else "no")])
            results.append(dict(id=fid, kind="opening", status=st, dimension_check=dim, detected=det, note=note))
        else:
            rows.append([fid, f["kind"], "", f"{f['cx']:.3f}", f"{f['cy']:.3f}", f"{f['d']:.3f}", "", "", "", "", "",
                         "", "NOT MEASURED",
                         "concentric ring (counterbore/chamfer/boss edge)" if f["kind"] == "ring" else "visual check",
                         "", ""])
            results.append(dict(id=fid, kind=f["kind"], status="NOT MEASURED"))

    ctx.table("drawing check; nominal and measured positions in drawing mm (y down), tolerances diameter/position",
              ["feature", "kind", "prominence_rank", "nominal_x", "nominal_y", "nominal_size", "measured_x",
               "measured_y", "measured_size", "position_dev_mm", "size_dev_mm", "tol_mm", "status", "note",
               "dimension_check", "detected"], rows)
    ctx.data["drawing"]["results"] = results
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    ctx.data["drawing"]["status_counts"] = counts

    # messages
    ctx.say(f"drawing: {path} -> outline {ctx.data['drawing']['outline_size_mm'][0]:.2f} x "
            f"{ctx.data['drawing']['outline_size_mm'][1]:.2f} mm, {len(holes)} holes, "
            f"{sum(f['kind'] == 'opening' for f in model['features'])} openings")
    ctx.say(f"registration: {s:.3f} px/mm ({reg['scale_source']}), rotation {reg['angle_deg']:.1f} deg"
            f"{', MIRRORED (other face?)' if sim.mirror else ''}, outline overlap {reg['coarse_iou']:.1%}, "
            f"edge scatter {rms_px:.2f} px, holes agree {reg['hole_agreement']:.0%}; "
            f"uncertainty: hole size +-{unc_size:.2f}, hole position +-{unc_pos:.2f}, edges +-{unc_edge:.2f} mm")
    for r in rows:
        ctx.say(f"  {r[0]:<8} {r[1]:<22} {r[12]:<13} pos dev {r[9] or '-':>6} mm  size dev {r[10] or '-':>7} mm  {r[13]}")
    ctx.say("summary: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    meas = [r for r in results if r["status"] != "NOT MEASURED"]
    n_det = sum(1 for r in meas if r["status"] in ("OK", "DETECTED"))
    ctx.data["drawing"]["detected"] = dict(found=n_det, checked=len(meas))
    dims = {}
    for r in meas:
        dims[r.get("dimension_check")] = dims.get(r.get("dimension_check"), 0) + 1
    ctx.data["drawing"]["dimension_check_counts"] = dims
    ctx.say(f"detected: {n_det} of {len(meas)} checked features (detection tolerance: position max({det_pos}, 0.5 x size), "
            f"size max({det_size}, 0.5 x size) mm); dimensional check +-tol: " + ", ".join(f"{v} {k}" for k, v in dims.items()))
    if reg["coarse_iou"] < 0.85:
        ctx.warn(f"The photo outline overlaps the drawing only {reg['coarse_iou']:.0%}: wrong drawing, the other face, "
                 "a damaged part, or the outline was not separated cleanly (reflections). Check the image.")
    if sim.mirror:
        ctx.note("The photo matches the MIRROR image of the drawing view: it probably shows the other face.")
    if not fixed:
        ctx.note("Scale comes from fitting the drawing, so a part that is uniformly too big or too small cannot be "
                 "detected. For absolute size, calibrate the camera once and pass --mm-per-px.")
    n_cv = counts.get("CAN'T VERIFY", 0)
    if n_cv:
        ctx.warn(f"{n_cv} feature(s) are within the photo's uncertainty of their tolerance limit, "
                 f"or too small to measure ({mm:.3f} mm per px). A closer / sharper photo would decide them.")
    lo_, hi_ = q.min(0), q.max(0)
    m_ = 0.2 * float((hi_ - lo_).max())
    ctx.view = (float(lo_[0] + hi_[0]) / 2, float(lo_[1] + hi_[1]) / 2, float(hi_[0] - lo_[0]) / 2 + m_,
                float(hi_[1] - lo_[1]) / 2 + m_)
    ctx.add_legend("dwg", "Cyan = drawing (expected); red edge = outline outside tolerance", CYAN, order=10)
    ctx.add_legend("st", "Measured: green OK, yellow-green DETECTED (found, outside +-tol), red DEVIATES, magenta MISSING",
                   GREEN, order=11)
    ctx.add_legend("info", f"Scale {mm:.4f} mm/px ({reg['scale_source']}).", MAG, order=99)
    ctx.drawing_model = model
    return dict(sim=sim, model=model, results=results)


def _encloses(f, model):
    """True when the contour contains other drawing features (a recess or step edge, not a plain opening)."""
    poly = f["pts"].astype(np.float32).reshape(-1, 1, 2)
    return any(g is not f and cv2.pointPolygonTest(poly, (float(g["cx"]), float(g["cy"])), False) > 0
               for g in model["features"])


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
