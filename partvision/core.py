"""Shared state passed to every step, plus geometry helpers.

A step reads the image and earlier results from the Context, and writes back:
  ctx.results[step]  - anything later steps need (numpy arrays, contours...)
  ctx.data[...]      - JSON-safe measurements for the report
  ctx.table(...)     - rows for the CSV
  ctx.feature(...)   - one line in the CSV 'features' section
  ctx.dims[name]     - a length in px that --known-mm can use as the scale reference
  ctx.draw(...)      - drawing primitives for the annotated image
  ctx.add_legend(...), ctx.say(...), ctx.warn(...), ctx.note(...)
"""
import math
import os

import cv2
import numpy as np

from .config import C


def odd(v):
    v = int(round(v))
    return v if v % 2 else v + 1


def fit_circle(pts, iters=5, floor=1.0):
    """Least-squares circle with iterative median/MAD outlier rejection.
    Returns x, y, r, residual std of inliers, inlier count, inlier mask."""
    pts = np.asarray(pts, float)
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        A = np.c_[2 * pts[keep, 0], 2 * pts[keep, 1], np.ones(keep.sum())]
        s = np.linalg.lstsq(A, (pts[keep] ** 2).sum(1), rcond=None)[0]
        r = math.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0))
        res = np.hypot(pts[:, 0] - s[0], pts[:, 1] - s[1]) - r
        med = np.median(res[keep])
        keep = np.abs(res - med) < max(floor, 3 * 1.4826 * np.median(np.abs(res[keep] - med)))
        if keep.sum() < 3:
            keep[:] = True
            break
    return float(s[0]), float(s[1]), float(r), float(res[keep].std()), int(keep.sum()), keep


def runs_of(flags):
    """Contiguous True runs on a circular array -> [(start, end_exclusive)], end may exceed n."""
    n = len(flags)
    if flags.all():
        return [(0, n)]
    if not flags.any():
        return []
    s0 = int(np.argmin(flags))
    out, i = [], 0
    while i < n:
        if flags[(s0 + i) % n]:
            j = i
            while j < n and flags[(s0 + j) % n]:
                j += 1
            out.append((s0 + i, s0 + j))
            i = j
        else:
            i += 1
    return out


def angle_deg(x, y, ox, oy):
    """Angle of (x, y) about (ox, oy), clockwise on screen, 0 = 3 o'clock."""
    return float(math.degrees(math.atan2(y - oy, x - ox)) % 360)


class Context:

    def __init__(self, path, opt):
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        self.path, self.opt, self.img = path, opt, img
        self.H, self.W = img.shape[:2]
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.gs = cv2.GaussianBlur(self.gray, (3, 3), 0).astype(np.float32)
        self.sN, self.gN = 1.0, self.gray
        self.data = {"image": os.path.abspath(path), "image_size_wh": [self.W, self.H],
                     "laplacian_var": float(cv2.Laplacian(self.gray, cv2.CV_64F).var())}
        self.results, self.dims, self.skipped = {}, {}, []
        self.warnings, self.notes, self.summary = [], [], []
        self.tables, self.features, self.layers, self.legend = [], [], [], []
        self.part = None
        self.origin = None  # (x, y, label): centre for angles and radii
        self.frame = None   # part coordinate frame for non-round parts
        self.view = None    # (cx, cy, half_w, half_h) crop for the annotated image
        self.mm = None
        self.NORM_R = float(C["image"]["normalised_radius_px"])  # detection-scale radius (config)

    # ---- messages
    def warn(self, m):
        self.warnings.append(m)

    def note(self, m):
        self.notes.append(m)

    def say(self, m):
        self.summary.append(m)

    # ---- outputs
    def table(self, title, header, rows, length_cols=()):
        self.tables.append(dict(title=title, header=list(header), rows=rows, length_cols=list(length_cols)))

    def feature(self, name, size_px="", start="", end="", span="", note=""):
        self.features.append([name, size_px, start, end, span, note])

    def draw(self, kind, z=0, debug=False, **kw):
        self.layers.append(dict(kind=kind, z=z, debug=debug, **kw))

    def add_legend(self, key, text, rgb, order=50):
        self.legend.append(dict(key=key, text=text, rgb=rgb, order=order))

    def opt_get(self, name, default=None):
        return getattr(self.opt, name, default) if self.opt is not None else default

    # ---- image access
    def set_detection_scale(self, radius_px):
        self.sN = self.NORM_R / radius_px
        self.gN = cv2.resize(self.gray, None, fx=self.sN, fy=self.sN,
                             interpolation=cv2.INTER_AREA if self.sN < 1 else cv2.INTER_CUBIC)
        self.data["detection_scale"] = self.sN

    def ray_profile(self, cx, cy, theta, rs):
        xs = (cx + rs * np.cos(theta)).astype(np.float32).reshape(-1, 1)
        ys = (cy + rs * np.sin(theta)).astype(np.float32).reshape(-1, 1)
        return cv2.remap(self.gs, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()

    def ray_edge_fit(self, cx, cy, rmin, rmax, n=180, passes=3, floor=1.0):
        """Strongest radial edge on each ray, then a robust circle fit; re-centre and repeat."""
        k = C["analysis"]["ray_edge_fit"]
        rs = np.arange(max(rmin, 1), rmax, k["radial_step_px"], dtype=np.float32)
        for _ in range(passes):
            pts = []
            for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
                p = cv2.GaussianBlur(self.ray_profile(cx, cy, th, rs).reshape(-1, 1), (1, odd(k["profile_blur"])), 0).ravel()
                i = int(np.argmax(np.abs(np.gradient(p))))
                pts.append((cx + rs[i] * np.cos(th), cy + rs[i] * np.sin(th)))
            cx, cy, r, std, used, keep = fit_circle(pts, floor=floor)
        pts = np.array(pts, np.float32)
        ell = cv2.fitEllipse(pts[keep]) if keep.sum() >= 5 else None
        return dict(x=cx, y=cy, r=r, d=2 * r, std=std, rays=used, of=n, ellipse=ell)

    def disc_stat(self, x, y, r0, r1, fn=np.median):
        """Statistic of gray values in the annulus r0 <= d < r1 (r0 = 0 gives a disc)."""
        m = int(math.ceil(r1)) + 1
        xa, xb = int(max(x - m, 0)), int(min(x + m + 1, self.W))
        ya, yb = int(max(y - m, 0)), int(min(y + m + 1, self.H))
        if xb <= xa or yb <= ya:
            return float("nan")
        yy, xx = np.ogrid[ya:yb, xa:xb]
        d = np.hypot(xx - x, yy - y)
        v = self.gray[ya:yb, xa:xb][(d >= r0) & (d < r1)]
        return float(fn(v)) if v.size else float("nan")

    def polar(self, r, a_deg, cx=None, cy=None):
        """Point at radius r and screen-clockwise angle a about (cx, cy), default the origin."""
        t = math.radians(a_deg)
        cx = self.origin[0] if cx is None else cx
        cy = self.origin[1] if cy is None else cy
        return cx + r * math.cos(t), cy + r * math.sin(t)

    def to_frame(self, x, y):
        """(x, y) in image px -> (u, v) along the part's own long and short axes."""
        if not self.frame:
            return None, None
        o, u, v = self.frame["origin"], self.frame["u"], self.frame["v"]
        d = np.array([x - o[0], y - o[1]])
        return float(d @ u), float(d @ v)
