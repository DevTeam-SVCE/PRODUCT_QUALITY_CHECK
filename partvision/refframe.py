"""A4 reference frame geometry: detection, edge refinement, alignment, sharpness and rectification.

Pure functions on numpy images: no threads, files or UI (those live in camera.py and camera_ui.py), so every
check can be exercised on synthetic frames.

Conventions
  * Pixel coordinates use the OpenCV pixel-centre convention (as cv2.remap / cv2.warpPerspective).
  * Corners are ordered clockwise on screen, starting with the corner nearest the image's top-left:
    TL, TR, BR, BL. Side i runs from corner i to corner i+1.
  * The reference frame plane is in mm, origin at corner 0, x along side 0, y along side 3, so the frame
    keeps the orientation it has in the photo (portrait: 210 x 297, landscape: 297 x 210).
"""
import json
import math

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------- geometry

def to_full(q, s):
    """Processing-resolution px -> full-resolution px (cv2.resize maps pixel centres, hence the 0.5)."""
    return (np.asarray(q, np.float64) + 0.5) / s - 0.5


def to_small(q, s):
    return (np.asarray(q, np.float64) + 0.5) * s - 0.5


def order_quad(pts):
    """Clockwise on screen (image y down), starting at the corner nearest the image's top-left."""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)


def quad_geometry(q):
    """Side lengths, interior angles, aspect (short/long of the mean opposite sides), keystone, area."""
    q = np.asarray(q, np.float64)
    sides = np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])
    ang = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        ang.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    h, v = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    q32 = q.astype(np.float32).reshape(-1, 1, 2)
    return dict(sides=sides, angles=np.array(ang), aspect=float(min(h, v) / max(h, v, 1e-9)), landscape=bool(h >= v),
                keystone=float(max(max(sides[0], sides[2]) / max(min(sides[0], sides[2]), 1e-9),
                                   max(sides[1], sides[3]) / max(min(sides[1], sides[3]), 1e-9)) - 1),
                area=float(abs(cv2.contourArea(q32))), convex=bool(cv2.isContourConvex(q32)))


def plane_points(landscape, ref_mm):
    """The frame's corners in mm (same order as the image corners) and its (width, height) in mm."""
    s, l = float(min(ref_mm)), float(max(ref_mm))
    w, h = (l, s) if landscape else (s, l)
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64), (w, h)


def _intersect(l1, l2):
    (p, v), (q, w) = l1, l2
    A = np.array([v, -w]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s = np.linalg.solve(A, q - p)
    return p + s[0] * v


def _robust_line(pts, iters=5, floor=0.35):
    """Total-least-squares line with MAD outlier rejection -> (point, direction, inlier mask, rms)."""
    keep = np.ones(len(pts), bool)
    c = v = res = None
    for _ in range(iters):
        if keep.sum() < 4:
            return None
        c = pts[keep].mean(0)
        v = np.linalg.svd(pts[keep] - c)[2][0]
        res = (pts - c) @ np.array([-v[1], v[0]])
        new = np.abs(res) <= max(floor, 3 * 1.4826 * np.median(np.abs(res[keep])))
        if (new == keep).all():
            break
        keep = new
    if keep.sum() < 4:
        return None
    return c, v, keep, float(np.sqrt(np.mean(res[keep] ** 2)))


# --------------------------------------------------------------------------------------------- detection

def _approx_quad(c):
    peri = cv2.arcLength(c, True)
    for src in (c, cv2.convexHull(c)):
        for eps in (0.012, 0.02, 0.035):
            a = cv2.approxPolyDP(src, eps * peri, True)
            if len(a) == 4:
                return a.reshape(4, 2).astype(np.float64)
    return None


def find_candidates(gray, cfg, ref_mm=(210.0, 297.0)):
    """Full search for A4-like quadrilaterals in a (processing-resolution) gray image.

    Two independent proposals, so neither low paper/background contrast nor a busy background alone
    defeats it: closed edge rings (Canny) and bright regions (Otsu). Returns (candidates best first,
    partial) where partial is True when a large frame-like region is cut by the image border."""
    H, W = gray.shape[:2]
    target = min(ref_mm) / max(ref_mm)
    b = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(b, 20, 50, L2gradient=True), np.ones((3, 3), np.uint8))
    otsu, bright = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # a second, higher level separates a (near-)saturated frame from a bright background that Otsu merges with it
    hi = 0.5 * (otsu + float(np.percentile(b[::4, ::4], 99)))
    brighter = (b > hi).astype(np.uint8) * 255
    min_area = float(cfg["detect_min_area_frac"]) * W * H
    atol, angtol = float(cfg["detect_aspect_tol"]), float(cfg["detect_angle_tol_deg"])
    cands, partial = [], False
    for src, mask, mode in (("edges", edges, cv2.RETR_LIST), ("bright", bright, cv2.RETR_EXTERNAL),
                            ("brighter", brighter, cv2.RETR_EXTERNAL)):
        for c in cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(c)
            if area < min_area or area > 0.97 * W * H:
                continue
            x, y, w, h = cv2.boundingRect(c)
            touches = x <= 2 or y <= 2 or x + w >= W - 3 or y + h >= H - 3
            quad = _approx_quad(c)
            # a frame partly outside the view is cut by one border or at one corner; a region spanning the view
            # between opposite borders is a band (table edge, mat), and one covering most of it is background
            cut = touches and src.startswith("bright") and area < 0.8 * W * H and \
                not ((x <= 2 and x + w >= W - 3) or (y <= 2 and y + h >= H - 3))
            if quad is None:
                # a bright, straight-sided region cut by the image border: the frame, partly outside the view
                if cut and \
                        area / max(cv2.contourArea(cv2.convexHull(c)), 1) > 0.95 and \
                        4 <= len(cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)) <= 6:
                    partial = True
                continue
            q = order_quad(quad)
            g = quad_geometry(q)
            if not g["convex"] or g["area"] < min_area:
                continue
            aerr = abs(g["aspect"] / target - 1)
            if aerr > atol or np.abs(g["angles"] - 90).max() > angtol or area / g["area"] < 0.85:
                if cut and area / g["area"] > 0.85:
                    partial = True
                continue
            touches = touches or bool((q[:, 0] <= 2).any() or (q[:, 1] <= 2).any() or
                                      (q[:, 0] >= W - 3).any() or (q[:, 1] >= H - 3).any())
            cands.append(dict(quad=q, area_frac=g["area"] / (W * H), aspect_err=aerr, touches=touches, source=src,
                              score=g["area"] / (W * H) * (1 - 0.5 * aerr / atol) * (0.5 if touches else 1.0)))
    cands.sort(key=lambda d: -d["score"])
    return cands, partial or any(c["touches"] for c in cands[:1])


# ------------------------------------------------------------------------------------- edge refinement

def refine_quad(gf, quad, search, n=24, step=0.5, min_inliers=0.6, min_grad=1.0):
    """Sub-pixel corners from the four edges: n gradient profiles across each side, the outermost strong
    edge of the dominant polarity on each (the frame's outer boundary, not an inner printed line), a robust
    line per side, corners from line intersections. Tolerates a part or finger covering some of an edge
    (outliers), and reports a side as occluded when too few samples lie on its line.

    gf: float32 gray image (lightly blurred). Returns dict(quad, inliers, rms, contrast) or None."""
    q = np.asarray(quad, np.float64)
    H, W = gf.shape[:2]
    offs = np.arange(-search, search + 1e-9, step, dtype=np.float64)
    m = len(offs)
    if m < 7:
        return None
    lines, inl, rms, contrast = [], [], [], []
    r = np.arange(n)
    for i in range(4):
        a, d = q[i], q[(i + 1) % 4] - q[i]
        L = float(np.linalg.norm(d))
        if L < 12:
            return None
        t = d / L
        nrm = np.array([t[1], -t[0]])  # outward for clockwise order
        P = a + np.linspace(0.06, 0.94, n)[:, None] * d
        X = (P[:, 0, None] + offs[None] * nrm[0]).astype(np.float32)
        Y = (P[:, 1, None] + offs[None] * nrm[1]).astype(np.float32)
        inside = (X.min(1) >= 0) & (Y.min(1) >= 0) & (X.max(1) <= W - 1) & (Y.max(1) <= H - 1)
        prof = cv2.remap(gf, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
        der = np.zeros_like(prof)
        der[:, 1:-1] = (prof[:, 2:] - prof[:, :-2]) / (2 * step)
        sgn = 1.0 if der.max(1)[inside].sum() > (-der).max(1)[inside].sum() else -1.0
        sd = der * sgn
        peak = sd.max(1)
        loc = np.zeros_like(sd, bool)
        loc[:, 1:-1] = (sd[:, 1:-1] >= sd[:, :-2]) & (sd[:, 1:-1] > sd[:, 2:]) & (sd[:, 1:-1] >= 0.5 * peak[:, None])
        k = m - 1 - np.argmax(loc[:, ::-1], axis=1)  # outermost qualifying peak
        valid = inside & loc.any(1) & (peak >= min_grad) & (k > 1) & (k < m - 2)
        k = np.clip(k, 1, m - 2)
        y0, y1, y2 = sd[r, k - 1], sd[r, k], sd[r, k + 1]
        den = y0 - 2 * y1 + y2
        off = np.clip(np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0), -0.5, 0.5)
        pts = P + (offs[k] + off * step)[:, None] * nrm
        fit = _robust_line(pts[valid]) if valid.sum() >= 4 else None
        if fit is None:
            return None
        c, v, keep, e = fit
        inl.append(float(keep.sum()) / n)
        rms.append(e)
        lines.append((c, v))
        q3 = max(2, m // 4)
        contrast.append(float(np.median(np.abs(prof[valid, -q3:].mean(1) - prof[valid, :q3].mean(1)))))
    corners = [_intersect(lines[i - 1], lines[i]) for i in range(4)]
    if any(p is None for p in corners):
        return None
    nq = np.array(corners)
    if np.abs(nq - q).max() > search + 3:
        return None
    return dict(quad=nq, inliers=inl, rms=rms, contrast=contrast,
                occluded=[x < min_inliers for x in inl])


# ------------------------------------------------------------------------------------- image quality

def edge_width(gf, quad, segs=10, avg=5, half=None, step=0.25):
    """Blur along the frame's edges (slanted-edge method, simplified): on each side, segments of `avg`
    parallel profiles across the fitted edge are averaged and the full width at half maximum of the edge's
    derivative is measured. The value is per side, so directional motion blur shows up on the sides
    perpendicular to the motion; the reported width is the worst side's median. gf: float32, NOT blurred."""
    H, W = gf.shape[:2]
    half = half or max(6.0, 0.005 * math.hypot(W, H))
    offs = np.arange(-half, half + 1e-9, step)
    m = len(offs)
    along = (np.arange(avg) - (avg - 1) / 2.0)
    q = np.asarray(quad, np.float64)
    side_w, amps = [], []
    for i in range(4):
        a, d = q[i], q[(i + 1) % 4] - q[i]
        L = float(np.linalg.norm(d))
        t = d / L
        nrm = np.array([t[1], -t[0]])
        ws = []
        for u in np.linspace(0.12, 0.88, segs):
            P = a + u * d + along[:, None] * t
            X = (P[:, 0, None] + offs[None] * nrm[0]).astype(np.float32)
            Y = (P[:, 1, None] + offs[None] * nrm[1]).astype(np.float32)
            if X.min() < 0 or Y.min() < 0 or X.max() > W - 1 or Y.max() > H - 1:
                continue
            p = cv2.remap(gf, X, Y, cv2.INTER_LINEAR).mean(0).astype(np.float64)
            der = np.gradient(p, step)
            k = int(np.argmax(np.abs(der)))
            sd = der * np.sign(der[k])
            pk = sd[k]
            if pk < 0.5 or k < 2 or k > m - 3:
                continue
            hm = pk / 2
            lo = k
            while lo > 0 and sd[lo] > hm:
                lo -= 1
            hi = k
            while hi < m - 1 and sd[hi] > hm:
                hi += 1
            if lo == 0 or hi == m - 1:
                continue
            xl = lo + (hm - sd[lo]) / (sd[lo + 1] - sd[lo])
            xr = hi - (hm - sd[hi]) / (sd[hi - 1] - sd[hi])
            fw = (xr - xl) * step
            ws.append(fw)
            span = int(round(2 * fw / step))
            amps.append(float(sd[max(0, k - span):k + span + 1].sum() * step))
        if ws:
            side_w.append(float(np.median(ws)))
    if not side_w:
        return None
    return dict(width_px=max(side_w), per_side_px=side_w, contrast=float(np.median(amps)) if amps else 0.0)


def edge_width_limit(cfg, W, H):
    return max(float(cfg["min_edge_width_limit_px"]), float(cfg["max_edge_width_frac"]) * math.hypot(W, H))


def sheet_thumb(gray, quad, landscape, size=(96, 136), inset=0.05):
    """The inside of the frame resampled to a small fixed-size image (motion, exposure, scene change)."""
    w, h = (size[1], size[0]) if landscape else size
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    dst = (dst - [w / 2, h / 2]) * (1 + 2 * inset) + [w / 2, h / 2]  # source quad maps outside: crop inset
    M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst.astype(np.float32))
    return cv2.warpPerspective(gray, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def thumb_change(a, b, level=20.0):
    """(mean |a-b|, share of pixels changed by more than `level`), after removing a global brightness shift
    (auto exposure). None when the thumbnails are not comparable (orientation changed)."""
    if a is None or b is None or a.shape != b.shape:
        return None
    d = a.astype(np.float32) - b.astype(np.float32)
    d = np.abs(d - float(np.median(d)))
    return float(d.mean()), float((d > level).mean())


def exposure(thumb):
    return dict(median=float(np.median(thumb)), clipped=float((thumb >= 250).mean()), dark=float((thumb <= 5).mean()))


def illumination(gray, grid=(6, 8), keep=0.88, iters=4):
    """Smooth illumination of the (white) reference sheet: the bright level (90th percentile) of each grid cell,
    then a robust quadratic surface through them. Cells darker than the fit (covered by the part or its cast
    shadow) are dropped and the surface refitted. Returns (surface, uniformity = min/max of the surface)."""
    g = np.asarray(gray, np.float32)
    H, W = g.shape[:2]
    gy, gx = grid if H >= W else grid[::-1]
    ys, xs, vs = [], [], []
    for i in range(gy):
        for j in range(gx):
            c = g[i * H // gy:(i + 1) * H // gy, j * W // gx:(j + 1) * W // gx]
            ys.append((i + 0.5) / gy - 0.5)
            xs.append((j + 0.5) / gx - 0.5)
            vs.append(float(np.percentile(c, 90)))
    x, y, v = np.array(xs), np.array(ys), np.array(vs)
    A = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
    ok = v >= keep * np.median(v)
    coef = None
    for _ in range(iters):
        if ok.sum() < 8:
            break
        coef = np.linalg.lstsq(A[ok], v[ok], rcond=None)[0]
        new = v >= keep * (A @ coef)
        if (new == ok).all():
            break
        ok = new
    if coef is None:
        return np.full((H, W), float(np.median(v)), np.float32), 1.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx, yy = (xx + 0.5) / W - 0.5, (yy + 0.5) / H - 0.5
    surf = (coef[0] + coef[1] * xx + coef[2] * yy + coef[3] * xx * xx + coef[4] * xx * yy + coef[5] * yy * yy)
    surf = np.clip(surf, 1.0, 255.0).astype(np.float32)
    return surf, float(surf.min() / surf.max())


def flatten(img, surface, level=225.0):
    """Divide out the illumination surface: the sheet becomes evenly `level` bright, the part keeps its contrast
    relative to the sheet. Brightness only; geometry is untouched."""
    k = (level / surface)[..., None] if img.ndim == 3 else level / surface
    return np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------- camera and pose

def load_intrinsics(path):
    """JSON from calibrate_camera.py: camera_matrix (3x3), dist_coeffs, image_size [w, h]."""
    with open(path) as f:
        d = json.load(f)
    return dict(K=np.array(d["camera_matrix"], np.float64), dist=np.array(d["dist_coeffs"], np.float64).ravel(),
                size=tuple(int(v) for v in d["image_size"]), rms=d.get("rms"), path=path)


def camera_model(intr, W, H, fov_deg):
    """(K, dist, kind) for a W x H frame. Calibrated intrinsics are rescaled to another resolution of the
    same aspect ratio (approximate: a different sensor crop mode would need a new calibration)."""
    if intr is not None:
        w0, h0 = intr["size"]
        if (w0, h0) == (W, H):
            return intr["K"], intr["dist"], "calibrated"
        if abs(w0 / h0 - W / H) < 0.01:
            K = intr["K"].copy()
            K[0] *= W / w0
            K[1] *= H / h0
            return K, intr["dist"], "calibrated (rescaled from %dx%d)" % (w0, h0)
    f = 0.5 * math.hypot(W, H) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], np.float64), None, \
        "assumed (diagonal FOV %.0f deg, no lens correction)" % fov_deg


def undistort_points(pts, K, dist):
    if dist is None:
        return np.asarray(pts, np.float64)
    return cv2.undistortPoints(np.asarray(pts, np.float64).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)


def distort_points(pts, K, dist):
    """Inverse of undistort_points (undistorted px -> original image px)."""
    if dist is None:
        return np.asarray(pts, np.float64)
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    n = (np.linalg.inv(K) @ np.c_[p, np.ones(len(p))].T).T
    return cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)


_SIDE_NAMES = {0: "right", 1: "bottom", 2: "left", 3: "top"}


def pose(quad_u, K, ref_mm):
    """Camera pose from the four (undistorted) corners: planar PnP (IPPE). Returns the tilt between the
    camera's optical axis and the frame's normal, the camera's distance to the plane, which side of the
    frame is farthest, and the reprojection error of the best A4 fit."""
    g = quad_geometry(quad_u)
    obj2, _ = plane_points(g["landscape"], ref_mm)
    obj = np.c_[obj2, np.zeros(4)]
    img = np.asarray(quad_u, np.float64).reshape(-1, 1, 2)
    # SQPnP (globally optimal). Not IPPE: near a frontal view IPPE's tilt jumps by ~1 deg for 0.1 px of corner
    # noise, while SQPnP / the iterative solver stay within ~0.05 deg (tests/test_camera.py checks this).
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    err = float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj, rvec, tvec, K, None)[0] - img) ** 2, axis=2))))
    R = cv2.Rodrigues(rvec)[0]
    t = np.asarray(tvec).ravel()
    nz = R[:, 2]
    tilt = math.degrees(math.acos(min(1.0, abs(float(nz[2])))))
    z = ((R @ obj.T).T + t)[:, 2]
    side_depth = [(z[j] + z[(j + 1) % 4]) / 2 for j in range(4)]
    far = int(np.argmax(side_depth))
    mid = (np.asarray(quad_u)[far] + np.asarray(quad_u)[(far + 1) % 4]) / 2 - np.asarray(quad_u).mean(0)
    a = math.degrees(math.atan2(mid[1], mid[0])) % 360
    far_name = _SIDE_NAMES[int(((a + 45) % 360) // 90)]
    return dict(tilt_deg=tilt, distance_mm=float(abs(nz @ t)), far_side=far_name,
                depth_spread=float((z.max() - z.min()) / max(z.mean(), 1e-9)), reprojection_px=err,
                rvec=np.asarray(rvec).ravel(), tvec=t)


def alignment(quad_u, K, ref_mm, cfg):
    """Every alignment number plus pass/fail per gate and a 0..1 score (share of the tolerance unused)."""
    g = quad_geometry(quad_u)
    p = pose(quad_u, K, ref_mm)
    target = min(ref_mm) / max(ref_mm)
    m = dict(corner_angles_deg=[round(float(x), 3) for x in g["angles"]],
             corner_angle_dev_deg=float(np.abs(g["angles"] - 90).max()), keystone=g["keystone"],
             aspect=g["aspect"], aspect_error=abs(g["aspect"] / target - 1), landscape=g["landscape"],
             tilt_deg=p["tilt_deg"] if p else 90.0, far_side=p["far_side"] if p else None,
             distance_mm=p["distance_mm"] if p else None, pnp_reprojection_px=p["reprojection_px"] if p else None)
    ratios = dict(tilt=m["tilt_deg"] / cfg["max_tilt_deg"],
                  corner_angles=m["corner_angle_dev_deg"] / cfg["max_corner_angle_dev_deg"],
                  keystone=m["keystone"] / cfg["max_keystone"])
    m["ratios"] = ratios
    m["aligned"] = bool(ratios["tilt"] <= 1 and ratios["corner_angles"] <= 1)
    m["perspective_ok"] = bool(ratios["keystone"] <= 1)
    m["aspect_ok"] = bool(m["aspect_error"] <= cfg["aspect_tol"])
    m["score"] = float(max(0.0, 1 - max(ratios.values())))
    return m


# --------------------------------------------------------------------------------------- rectification

def native_px_per_mm(quad, ref_mm):
    g = quad_geometry(quad)
    _, (w, h) = plane_points(g["landscape"], ref_mm)
    s = g["sides"]
    return np.array([s[0] / w, s[1] / h, s[2] / w, s[3] / h])


def rectify(img, quad_u, ref_mm, px_per_mm=None, inset_mm=0.0):
    """Perspective-correct the frame's plane: homography from the 4 corners to the frame's size in mm.
    Output: the inside of the frame (minus inset_mm) at px_per_mm (default: the frame's own resolution in
    the photo, so nothing is lost). Isotropic: the same mm per px horizontally and vertically."""
    g = quad_geometry(quad_u)
    obj, (w, h) = plane_points(g["landscape"], ref_mm)
    ppm = float(px_per_mm) if px_per_mm else float(native_px_per_mm(quad_u, ref_mm).max())
    mm_from_img = cv2.getPerspectiveTransform(np.asarray(quad_u, np.float32), obj.astype(np.float32)).astype(np.float64)
    S = np.array([[ppm, 0, -inset_mm * ppm], [0, ppm, -inset_mm * ppm], [0, 0, 1]], np.float64)
    Hr = S @ mm_from_img
    size = (int(round((w - 2 * inset_mm) * ppm)), int(round((h - 2 * inset_mm) * ppm)))
    out = cv2.warpPerspective(img, Hr, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # check: the corners land where the mm model puts them
    back = cv2.perspectiveTransform(np.asarray(quad_u, np.float64).reshape(-1, 1, 2), Hr).reshape(-1, 2)
    err = float(np.abs(back - (obj - inset_mm) * ppm).max())
    return dict(image=out, H_rect_from_img=Hr, H_mm_from_img=mm_from_img, px_per_mm=ppm, size_wh=size,
                plane_wh_mm=(w, h), landscape=g["landscape"], corner_check_px=err,
                cond=float(np.linalg.cond(mm_from_img)))
