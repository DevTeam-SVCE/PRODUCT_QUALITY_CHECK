"""Outline shape: round / rectangular / polygon / irregular, size, part frame and cut-ins."""
import math

import cv2
import numpy as np

from ..config import C
from ..core import fit_circle
from ..registry import step


def is_round(ctx):
    """Whether later steps treat the part as round (measured by the outline step, or forced with --shape)."""
    return bool(ctx.results.get("outline", {}).get("round"))


@step("outline", needs=("part",))
def outline(ctx):
    """Classify the outline, measure its size and set the part's own coordinate frame."""
    p = ctx.part
    c = p["contour"]
    k = C["analysis"]["outline"]
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    circ = 4 * math.pi * area / max(perim ** 2, 1)
    hull = cv2.convexHull(c)
    solidity = area / max(cv2.contourArea(hull), 1)
    (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(c)
    L, Wd = max(rw, rh), min(rw, rh)
    rect_fill = area / max(rw * rh, 1)
    approx = cv2.approxPolyDP(c, k["corner_approx_frac"] * perim, True)
    pts = c[:, 0, :].astype(float)
    ccx, ccy, cr, cstd, cused, _ = fit_circle(pts, iters=6, floor=k["circle_fit_floor_frac"] * p["r"])
    inl = cused / len(pts)
    tilt, shape = None, None
    if inl >= k["round_min_inliers"] and cstd <= k["round_max_scatter_frac"] * cr \
            and cr >= k["round_min_radius_frac"] * p["r"]:
        shape = "round"
    else:
        # a round part photographed at an angle looks elliptical: test an ellipse fit before giving up
        # robust: refit on inliers so a notch or cut-out does not drag the ellipse
        keep = np.ones(len(pts), bool)
        for _ in range(5):
            (ex, ey), (ea, eb), eang = cv2.fitEllipse(pts[keep].astype(np.float32))
            t = math.radians(eang)
            dx, dy = pts[:, 0] - ex, pts[:, 1] - ey
            xr, yr = dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)
            mean_r = (ea + eb) / 4
            res = (np.hypot(xr / (ea / 2), yr / (eb / 2)) - 1) * mean_r
            med = np.median(res[keep])
            keep = np.abs(res - med) < max(k["ellipse_tol_min_px"], 3 * 1.4826 * np.median(np.abs(res[keep] - med)))
            if keep.sum() < k["ellipse_min_points"]:
                break
        einl = float(np.mean(np.abs(res) <= max(k["ellipse_tol_min_px"], k["ellipse_tol_frac"] * mean_r)))
        ratio = min(ea, eb) / max(ea, eb)
        if einl >= k["round_min_inliers"] and k["ellipse_min_axis_ratio"] <= ratio < k["ellipse_max_axis_ratio"]:
            shape = "round"
            tilt = float(math.degrees(math.acos(ratio)))
            ctx.warn(f"The outline is an ellipse (axis ratio {ratio:.3f}): a round part seen about {tilt:.0f} deg "
                     "off top-down (or a genuinely oval part). If round: measurements are distorted; retake from directly above.")
    if shape == "round":
        pass
    elif len(approx) == 4 and rect_fill >= k["rect_min_fill"]:
        shape = "rectangular"
    elif solidity >= k["polygon_min_solidity"] and len(approx) <= k["polygon_max_corners"]:
        shape = f"polygon ({len(approx)} corners)"
    else:
        shape = "irregular"
    forced = ctx.opt_get("shape") or "auto"
    treat_round = shape == "round" if forced == "auto" else forced == "round"
    if forced != "auto" and treat_round != (shape == "round"):
        ctx.note(f"Outline measured as {shape}, analysed as {'round' if treat_round else 'not round'} (--shape {forced}).")

    # Part frame: u along the long side of the min-area rectangle, v across it, origin at its corner.
    box = cv2.boxPoints(((rcx, rcy), (rw, rh), rang))
    e0, e1 = box[1] - box[0], box[2] - box[1]
    if L / max(Wd, 1e-6) < k["square_aspect"]:  # near-square: no clear long side, take the side nearest horizontal
        u = e0 if abs(e0[0]) >= abs(e1[0]) else e1
    else:
        u = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    u = u / np.linalg.norm(u)
    if u[0] < 0 or (abs(u[0]) < 1e-9 and u[1] < 0):
        u = -u
    v = np.array([-u[1], u[0]])  # +90 deg on screen: points 'down' when u points right
    proj = (box - [rcx, rcy]) @ np.c_[u, v]
    origin = np.array([rcx, rcy]) + proj[:, 0].min() * u + proj[:, 1].min() * v
    ctx.frame = dict(origin=origin, u=u, v=v, L=L, W=Wd, angle=float(math.degrees(math.atan2(u[1], u[0]))))

    # Cut-ins: convexity defects deeper than a fraction of the part size
    cut_ins = []
    try:
        hidx = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hidx)
    except cv2.error:
        defects = None
    min_depth = max(k["cut_in_min_depth_px"], k["cut_in_min_depth_frac"] * math.sqrt(area))
    if defects is not None:
        for s, e, f_, d in defects.reshape(-1, 4):
            if d / 256 >= min_depth:
                S, E, F = c[s, 0].astype(float), c[e, 0].astype(float), c[f_, 0].astype(float)
                cut_ins.append(dict(start=S.tolist(), end=E.tolist(), deepest=F.tolist(), depth=d / 256,
                                    mouth_width=float(np.linalg.norm(E - S))))
    cut_ins.sort(key=lambda d: -d["depth"])
    for i, d in enumerate(cut_ins, 1):
        d["name"] = f"C{i}"

    ctx.data["outline"] = dict(shape=shape, area_px2=area, perimeter_px=perim, circularity=circ, solidity=solidity,
                               min_rect_length=L, min_rect_width=Wd, min_rect_angle_deg=ctx.frame["angle"],
                               rect_fill=rect_fill, corners=len(approx), analysed_as="round" if treat_round else "other", circle_fit=dict(x=ccx, y=ccy, d=2 * cr,
                               std=cstd, inlier_fraction=inl), cut_ins=cut_ins if shape != "round" else [])
    ctx.say(f"outline: {shape}; min-area rectangle {L:.1f} x {Wd:.1f} px at {ctx.frame['angle']:.1f} deg; "
            f"circularity {circ:.3f}, solidity {solidity:.3f}")
    if shape == "round":
        ctx.dims["outer"] = 2 * cr
        return dict(shape=shape, round=treat_round)

    ctx.dims["length"], ctx.dims["width"] = L, Wd
    ctx.feature("length (min-area rectangle)", f"{L:.2f}", note=f"long axis at {ctx.frame['angle']:.1f} deg")
    ctx.feature("width (min-area rectangle)", f"{Wd:.2f}")
    for d in cut_ins:
        ctx.feature(f"cut-in {d['name']}", f"{d['depth']:.2f}", note=f"depth; mouth width {d['mouth_width']:.1f} px")
        ctx.say(f"cut-in {d['name']}: depth {d['depth']:.1f} px, mouth {d['mouth_width']:.1f} px")
    WHITE, LBLUE, OR = (235, 235, 235), (120, 200, 255), (255, 150, 30)
    ctx.draw("polyline", z=0, points=pts[::max(1, len(pts) // 1500)].tolist(), color=WHITE, width=1.5, closed=True)
    ctx.draw("polyline", z=0, points=box.tolist(), color=LBLUE, width=1, closed=True)
    for p0, p1 in ((box[0], box[1]), (box[1], box[2])):  # label two adjacent sides
        n = np.linalg.norm(p1 - p0)
        mid = (p0 + p1) / 2
        out = mid - np.array([rcx, rcy])
        out = out / max(np.linalg.norm(out), 1e-9)
        tx, ty = mid + out * 0.06 * L
        ctx.draw("text", z=6, x=float(tx), y=float(ty), text=f"{'L' if abs(n - L) < 1e-3 else 'W'} {n:.0f}",
                 color=LBLUE, size=14)
    o = ctx.frame["origin"]
    ctx.draw("text", z=6, x=float(o[0] - 0.03 * L), y=float(o[1] - 0.03 * L), text="0", color=WHITE, size=12)
    for d in cut_ins:
        ctx.draw("polyline", z=2, points=[d["start"], d["deepest"], d["end"]], color=OR, width=2.5)
        ctx.draw("text", z=6, x=d["deepest"][0], y=d["deepest"][1], text=d["name"], color=OR, size=13, dy=-14)
    ctx.add_legend("outline", "White = part outline; light blue = min-area rectangle (L x W px), 0 = frame origin",
                   LBLUE, order=15)
    if cut_ins:
        ctx.add_legend("orange", (f"cut-ins C1-C{len(cut_ins)}" if len(cut_ins) > 1 else "cut-in C1") + " (convexity defects)", OR, order=20)
    return dict(shape=shape, round=treat_round)
