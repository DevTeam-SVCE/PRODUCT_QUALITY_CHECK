"""Holes of any shape inside a part of any shape."""
import math

import cv2
import numpy as np

from ..core import angle_deg, odd
from ..holefit import hole_wall_fit
from ..config import C
from ..registry import step

GREEN, YEL = (40, 220, 40), (255, 240, 40)


def _describe(ctx, c, source):
    """Shape descriptors of one opening; precise wall fit when it is round."""
    area = cv2.contourArea(c)
    if area <= 0:
        return None
    M = cv2.moments(c)
    x, y = M["m10"] / M["m00"], M["m01"] / M["m00"]
    perim = cv2.arcLength(c, True)
    circ = 4 * math.pi * area / max(perim ** 2, 1)
    solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
    (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(c)
    L, Wd = max(rw, rh), min(rw, rh)
    aspect = L / max(Wd, 1e-6)
    approx = cv2.approxPolyDP(c, 0.04 * perim, True)
    if circ >= 0.8 and aspect <= 1.25:
        shape = "round"
    elif aspect >= 1.6 and solidity >= 0.85:
        shape = "slot"
    elif len(approx) == 4 and area / max(rw * rh, 1) >= 0.85:
        shape = "rectangular"
    else:
        shape = "irregular"
    box = cv2.boxPoints(((rcx, rcy), (rw, rh), rang))
    e = box[1] - box[0] if np.linalg.norm(box[1] - box[0]) >= np.linalg.norm(box[2] - box[1]) else box[2] - box[1]
    h = dict(x=float(x), y=float(y), area=float(area), eq_d=float(2 * math.sqrt(area / math.pi)), length=float(L),
             width=float(Wd), aspect=float(aspect), circularity=float(circ), solidity=float(solidity),
             axis_deg=float(math.degrees(math.atan2(e[1], e[0])) % 180), shape=shape, source=source, fit=None)
    h["d"] = h["eq_d"]
    if shape == "round":
        rm = h["eq_d"] / 2
        inside = ctx.disc_stat(x, y, 0, max(2, 0.5 * rm))
        around = ctx.disc_stat(x, y, 1.3 * rm, 1.8 * rm)
        pol = -1.0 if inside < around else 1.0
        f = hole_wall_fit(ctx, x, y, (inside + around) / 2, pol, 0.3 * rm, 1.6 * rm, floor=max(0.5, 0.05 * rm))
        if f and f["std"] <= max(0.6, 0.1 * rm) and f["rays"] >= 0.75 * f["of"] and abs(f["r"] - rm) < 0.25 * rm:
            h.update(x=f["x"], y=f["y"], d=f["d"], fit=dict(std=f["std"], rays=f["rays"], of=f["of"]))
    return h, c


@step("holes", needs=("part",))
def holes(ctx):
    """Openings showing background, plus blind holes darker (or brighter) than the surrounding surface."""
    p = ctx.part
    sN, NR = ctx.sN, ctx.NORM_R
    found, skipped = [], 0
    bg = float(np.median(ctx.gray[p["mask_outer"] == 0]))
    surf = float(np.median(ctx.gray[p["mask_material"] > 0]))
    for c in p["inner"]:
        # A background-coloured region inside the outline is an opening only if it has a solid interior
        # that matches the background; thin shiny highlights also pass Otsu and are dropped here.
        x, y, w, h = cv2.boundingRect(c)
        m = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(m, [c - [x - 1, y - 1]], -1, 1, -1)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        rin = float(dt.max())
        core = ctx.gray[y - 1:y + h + 1, x - 1:x + w + 1][dt[:ctx.gray[y - 1:y + h + 1, x - 1:x + w + 1].shape[0],
                                                             :ctx.gray[y - 1:y + h + 1, x - 1:x + w + 1].shape[1]]
                                                          >= 0.5 * max(rin, 1)]
        solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
        if rin < 0.015 * p["r"] or not core.size or abs(np.median(core) - bg) > 0.35 * abs(surf - bg) \
                or (rin < 0.04 * p["r"] and solidity < 0.8):  # thin and curved: a highlight, not a slot
            skipped += 1
            continue
        r = _describe(ctx, c, "opening (background visible)")
        if r:
            found.append(r)
    if skipped:
        ctx.note(f"{skipped} light region(s) inside the outline were not counted as openings (thin or not "
                 "background-coloured: highlights or reflections).")

    # Surface holes on the normalised image: local contrast against a wide median background
    mat = cv2.resize(p["mask_material"], (ctx.gN.shape[1], ctx.gN.shape[0]), interpolation=cv2.INTER_NEAREST)
    er = cv2.erode(mat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(0.02 * NR),) * 2))
    g = ctx.gN.astype(np.float32)
    # Background estimate from part material only: outside the part, substitute the local mean of nearby
    # material (normalised convolution), so corners and edges are not compared with the backdrop.
    k = odd(0.15 * NR)
    m = (mat > 0).astype(np.float32)
    local = cv2.blur(g * m, (2 * k, 2 * k)) / np.maximum(cv2.blur(m, (2 * k, 2 * k)), 1e-3)
    filled = np.where(mat > 0, g, local).clip(0, 255).astype(np.uint8)
    bgm = cv2.medianBlur(filled, k).astype(np.float32)
    pol_opt = ctx.opt_get("hole_polarity", "dark") or "dark"
    diff = (bgm - g) if pol_opt == "dark" else (g - bgm)
    vals = g[er > 0]
    if vals.size:
        spread = (np.percentile(vals, 50) - np.percentile(vals, 0.5)) if pol_opt == "dark" else \
            (np.percentile(vals, 99.5) - np.percentile(vals, 50))
        thr = max(float(C["image"]["min_part_contrast"]), 0.4 * spread)
        black = np.percentile(vals, 0.5) if pol_opt == "dark" else np.percentile(vals, 99.5)
        cand = ((diff > thr) & (er > 0)).astype(np.uint8)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
        amin, amax = math.pi * (0.008 * NR) ** 2, 0.2 * (er > 0).sum()
        for i in range(1, n):
            if not amin <= st[i, 4] <= amax:
                continue
            x_, y_, w_, h_ = st[i, :4]
            comp = (lab[y_:y_ + h_, x_:x_ + w_] == i).astype(np.uint8)
            dc = diff[y_:y_ + h_, x_:x_ + w_][comp > 0]
            if dc.max() < max(40, 0.6 * spread) or np.median(dc) < 1.25 * thr:
                continue  # not hole-dark across its area: a stain, streak or shadow
            # A real hole has a solid interior close to the darkest level on the part. Streaks, scratches and
            # shadow lines are thin: their inscribed radius is small and they have no dark core.
            dt = cv2.distanceTransform(np.pad(comp, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            rin = float(dt.max())
            if rin < 0.015 * NR:
                continue  # thinner than 3% of the part radius across: a scratch or edge line
            core = g[y_:y_ + h_, x_:x_ + w_][dt >= 0.5 * rin]
            if abs(np.median(core) - black) > 0.2 * spread:
                continue  # interior not hole-dark: a stain or shadow
            cc, _ = cv2.findContours(np.pad(comp, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cc, key=cv2.contourArea).astype(np.float32)
            c[:, 0, 0] += x_ - 1
            c[:, 0, 1] += y_ - 1
            c = (c / sN).astype(np.int32)
            if cv2.contourArea(c) and cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1) < 0.75:
                continue  # ragged outline: rust or texture
            r = _describe(ctx, c, "blind/surface hole (%s than surface)" % ("darker" if pol_opt == "dark" else "brighter"))
            if r and all(math.hypot(r[0]["x"] - q[0]["x"], r[0]["y"] - q[0]["y"]) > 0.5 * q[0]["eq_d"] for q in found):
                found.append(r)
        ctx.data["surface_hole_threshold"] = dict(threshold=thr, spread=spread, polarity=pol_opt)

    # number in reading order along the part's own axes (u along the long side)
    med = np.median([h["eq_d"] for h, _ in found]) if found else 1
    for h, _ in found:
        h["u"], h["v"] = ctx.to_frame(h["x"], h["y"])
        if h["u"] is None:
            h["u"], h["v"] = h["x"], h["y"]
    found.sort(key=lambda hc: (round(hc[0]["v"] / max(med, 1)), hc[0]["u"]))
    ox, oy = ctx.origin[:2]
    items = []
    for i, (h, c) in enumerate(found, 1):
        h["n"] = f"H{i}"
        h["R"] = float(math.hypot(h["x"] - ox, h["y"] - oy))
        h["ang"] = angle_deg(h["x"], h["y"], ox, oy)
        h["contains_origin"] = cv2.pointPolygonTest(c, (float(ox), float(oy)), False) >= 0
        items.append(h)
        if h["shape"] == "round":
            ctx.draw("circle", z=4, x=h["x"], y=h["y"], r=h["d"] / 2 + 0.15 * h["d"], color=GREEN, width=2.5)
        else:
            ctx.draw("polyline", z=4, points=c[:, 0, :].tolist(), color=GREEN, width=2.5, closed=True)
        ctx.draw("text", z=6, x=h["x"], y=h["y"] - 0.5 * h["width"] - 0.03 * ctx.part["r"], text=h["n"], color=YEL,
                 size=13)
    ctx.data["holes_generic"] = items
    shapes = {}
    for h in items:
        shapes[h["shape"]] = shapes.get(h["shape"], 0) + 1
    rows = [[h["n"], h["shape"], h["source"], f'{h["x"]:.2f}', f'{h["y"]:.2f}', f'{h["u"]:.2f}', f'{h["v"]:.2f}',
             f'{h["d"]:.2f}', f'{h["length"]:.2f}', f'{h["width"]:.2f}', f'{h["axis_deg"]:.1f}',
             f'{h["circularity"]:.3f}', "edge fit" if h["fit"] else "area"] for h in items]
    ctx.table("holes and openings; u/v = position along the part's long/short axis from frame origin 0; "
              "diameter = edge fit for round holes, else area-equivalent",
              ["hole", "shape", "source", "x_px", "y_px", "u_px", "v_px", "diameter_px", "length_px", "width_px",
               "axis_deg", "circularity", "diameter_method"], rows,
              length_cols=["u_px", "v_px", "diameter_px", "length_px", "width_px"])
    round_d = [h["d"] for h in items if h["shape"] == "round"]
    if round_d:
        ctx.dims["hole"] = float(np.median(round_d))
    central = [h for h in items if h["contains_origin"] and h["shape"] == "round"]
    if central:
        ctx.dims["bore"] = central[0]["d"]
    ctx.add_legend("holes", f"Green + H# = hole/opening ({len(items)}: "
                   + ", ".join(f"{v} {k}" for k, v in shapes.items()) + "), numbered along the part's long axis",
                   GREEN, order=10)
    if any(h["shape"] == "irregular" for h in items):
        ctx.warn(f"{sum(h['shape'] == 'irregular' for h in items)} irregular opening(s): check they are real "
                 "features, not shadows or stains.")
    if not items:
        ctx.note("No holes or openings found.")
    ctx.say(f"holes/openings: {len(items)} ({shapes})")
    for h in items:
        ctx.say("%4s %-11s d=%7.2f L=%7.2f W=%7.2f at (%7.1f,%7.1f) u,v=(%7.1f,%7.1f) circ=%.2f %s" % (
            h["n"], h["shape"], h["d"], h["length"], h["width"], h["x"], h["y"], h["u"], h["v"], h["circularity"],
            "[edge fit]" if h["fit"] else ""))
    return dict(items=items)
