"""Part / background segmentation methods for the `part` step.

Each method takes the gray image (uint8) and returns a list of candidate regions (dicts with at least
`contour`, `inner`, `pol`) or [] when it finds nothing. The `part` step picks one method from the config
(`segmentation.method`) or, with "auto", runs them in `segmentation.auto_order` and takes the first usable
result. Add a method by decorating a function with @method("name"); nothing else has to change.

  otsu  - one global brightness threshold. Robust on plain, evenly lit backgrounds, but a cast shadow on a
          light background is as dark as the part and merges with it; a shiny face as bright as the background
          is cut away.
  edges - the part is what the background cannot reach without crossing a *sharp* edge. A shadow's border is
          soft (its penumbra is many px wide) while a part's border is sharp, so shadows stay background even
          when they are darker than the part. Sharpness is measured scale-free: the gradient at a fine blur vs.
          a coarse blur (a step edge loses most of its gradient when blurred, a soft ramp keeps it).

All thresholds live in config.py (section "segmentation"); sizes are fractions of the image's shorter side.
"""
import math

import cv2
import numpy as np

from .config import C

METHODS = {}


def method(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def _cfg():
    return C["segmentation"]


def _disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _odd(v):
    v = int(round(v))
    return v if v % 2 else v + 1


def _otsu_dark(gray):
    """Otsu threshold on a lightly blurred copy: (threshold, mask of the darker class)."""
    s = _cfg()
    kb = _odd(max(s["otsu_blur_min_px"], min(gray.shape) * s["otsu_blur_frac"]))
    blur = cv2.GaussianBlur(gray, (kb, kb), 0)
    t, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(t), dark


def _regions(mask, pol, holes_mask=None):
    """Top-level regions of a binary mask as candidates. Inner contours (openings) come from `holes_mask`
    restricted to each region when given, else from the mask's own holes."""
    cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    hier = hier[0]
    out = []
    for i in range(len(cs)):
        if hier[i][3] != -1:
            continue
        area = cv2.contourArea(cs[i])
        if holes_mask is None:
            inner = [cs[j] for j in range(len(cs)) if hier[j][3] == i]
        else:
            reg = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(reg, cs, i, 255, -1)
            ics, ih = cv2.findContours(cv2.bitwise_and(reg, holes_mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            # openings: regions of the background class fully inside the part (not touching its outline)
            inner = []
            if ics:
                edge_band = cv2.dilate(255 - reg, _disk(1))
                for j, c in enumerate(ics):
                    if ih[0][j][3] != -1:
                        continue
                    m = np.zeros(mask.shape, np.uint8)
                    cv2.drawContours(m, [c], -1, 255, -1)
                    if not cv2.countNonZero(cv2.bitwise_and(m, edge_band)):
                        inner.append(c)
        out.append(dict(contour=cs[i], area=area, inner=[c for c in inner
                                                          if cv2.contourArea(c) > _cfg()["min_inner_frac"] * area],
                        pol=pol))
    return out


@method("otsu")
def otsu(gray):
    """Otsu in both polarities (the original segmentation)."""
    t, dark = _otsu_dark(gray)
    cands = []
    for pol, mask in (("dark part on light background", dark), ("light part on dark background", 255 - dark)):
        for c in _regions(mask, pol):
            c["threshold"] = t
            cands.append(c)
    return cands


def _gradient(g, sigma):
    gb = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3)) / 8


def sharp_edges(gray):
    """Binary map of sharp edges: fine-scale gradient well above the image noise AND much stronger than the
    coarse-scale gradient at the same place (soft shadow borders fail the second test), with hysteresis so a
    faint stretch of the outline stays connected to its strong parts."""
    s = _cfg()
    g = gray.astype(np.float32)
    g1, g2 = _gradient(g, s["edge_sigma_fine_px"]), _gradient(g, s["edge_sigma_coarse_px"])
    noise = max(float(np.median(g1)), s["edge_noise_floor"])
    sharp = g1 > s["edge_sharpness_ratio"] * g2
    lo = (sharp & (g1 > s["edge_k_low"] * noise)).astype(np.uint8)
    hi = sharp & (g1 > s["edge_k_high"] * noise)
    n, lab = cv2.connectedComponents(lo, connectivity=8)
    keep = np.unique(lab[hi])
    keep = keep[keep > 0]
    return np.isin(lab, keep).astype(np.uint8) * 255, noise


@method("edges")
def edges(gray):
    """Flood the background from the image border without crossing sharp edges; the rest is the part."""
    s = _cfg()
    E, noise = sharp_edges(gray)
    side = min(gray.shape)
    gap = max(1, round(s["edge_gap_frac"] * side))
    k = _disk(gap)
    free = (cv2.dilate(E, k) == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    border = border[border > 0]
    fg = np.where(np.isin(lab, border), 0, 255).astype(np.uint8)
    fg = cv2.erode(fg, k)
    # thin spurs (a background scratch or paper fibre edge joined to the outline): remove them with an
    # opening, then restore the true outline everywhere except the spurs
    r = max(1, round(s["spur_frac"] * side))
    opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _disk(r))
    fg = cv2.bitwise_and(fg, cv2.dilate(opened, _disk(r)))
    # fill each region: openings are found separately, by brightness, below
    cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(fg)
    cv2.drawContours(solid, cs, -1, 255, -1)
    if not cv2.countNonZero(solid):
        return []
    # polarity: is the part darker or lighter than the background around it?
    t, dark = _otsu_dark(gray)
    ring = cv2.dilate(solid, _disk(3 * gap)) & ~solid
    part_med = float(np.median(gray[solid > 0]))
    bg_med = float(np.median(gray[ring > 0])) if cv2.countNonZero(ring) else float(np.median(gray[solid == 0]))
    is_dark = part_med <= bg_med
    pol = "dark part on light background" if is_dark else "light part on dark background"
    # openings: background-class pixels (by Otsu) enclosed by the part
    bg_class = (255 - dark) if is_dark else dark
    cands = _regions(solid, pol, holes_mask=bg_class)
    for c in cands:
        c.update(threshold=t, edge_noise=noise)
    return cands


def pick(cands, W, H):
    """Choose the part among candidates: drop specks and regions as big as the background; among regions at
    least `keep_frac_of_largest` of the biggest, prefer one that does not touch the image border."""
    s = _cfg()
    lo, hi = s["min_area_frac"] * W * H, s["max_area_frac"] * W * H
    out = []
    for c in cands:
        if not lo <= c["area"] <= hi:
            continue
        x, y, w, h = cv2.boundingRect(c["contour"])
        (mx, my), r = cv2.minEnclosingCircle(c["contour"])
        c.update(bbox=(x, y, w, h), x=mx, y=my, r=r, fill=c["area"] / (math.pi * r ** 2 + 1e-9),
                 touches=x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1)
        out.append(c)
    if not out:
        return None
    biggest = max(c["area"] for c in out)
    out = [c for c in out if c["area"] >= s["keep_frac_of_largest"] * biggest]
    out.sort(key=lambda c: (c["touches"], -c["area"]))
    return out[0]


def segment(gray, name=None):
    """Run the configured method ('auto': the first in auto_order whose pick is usable). Returns (cand, method,
    tried) where tried lists (method, reason) for methods that were skipped."""
    s = _cfg()
    name = name or s["method"]
    order = list(s["auto_order"]) if name == "auto" else [name]
    H, W = gray.shape
    tried = []
    for m in order:
        if m not in METHODS:
            raise SystemExit(f"unknown segmentation method '{m}'. Available: {', '.join(sorted(METHODS))}")
        c = pick(METHODS[m](gray), W, H)
        if c is None:
            tried.append((m, "no region"))
            continue
        if name == "auto" and c["touches"]:
            tried.append((m, "touches the image border"))
            continue
        return c, m, tried
    # auto: nothing usable -> fall back to the last method's raw pick, border or not
    for m in reversed(order):
        c = pick(METHODS[m](gray), W, H)
        if c is not None:
            return c, m, tried
    return None, None, tried
