"""Checking a photo against its drawing, as five pipeline steps (the "drawing" pipeline). The work is done by the
partvision.drawing_check package; each step here only passes results along:

  drawing_load      load the drawing (--drawing DXF/JSON)
  drawing_align     align the drawing to the photo (outline search, hole pairs, refine, perspective)
  drawing_outline   measure the outline edge under that alignment; the photo's uncertainties
  drawing_features  measure and judge every drawing feature (one registered check per feature kind)
  drawing_report    table, counts, console summary, warnings, legend
"""
import math
from types import SimpleNamespace

import cv2
import numpy as np

from .. import drawing as dwg
from ..config import C
from ..core import odd
from ..drawing_check.align import find_alignment
from ..drawing_check.checks import check_hole
from ..drawing_check.common import _encloses, _k
from ..drawing_check.features import FEATURE_CHECKS, kind_of
from ..drawing_check.judge import Judge
from ..drawing_check.outline import fit_outline
from ..drawing_check.report import report
from ..drawing_check.transform import Proj
from ..registry import run, step

DRAWING_STEPS = ["drawing_load", "drawing_align", "drawing_outline", "drawing_features", "drawing_report"]


@step("drawing_load", needs=("part",))
def drawing_load(ctx):
    """Load the drawing (--drawing DXF/JSON): outline, holes, openings and their tolerances."""
    path = ctx.opt_get("drawing")
    if not path:
        raise SystemExit("the drawing check needs --drawing FILE.dxf or FILE.json")
    tol = float(ctx.opt_get("tol") or C["tolerance"]["default_mm"])
    model, notes, info = dwg.load(path, tol, ctx.opt_get("pos_tol") or C["tolerance"]["position_mm"])
    for n in notes:
        ctx.note(n)
    fixed = ctx.opt_get("mm_per_px")
    return dict(path=path, model=model, info=info, fixed=fixed, fixed_scale=1.0 / float(fixed) if fixed else None,
                g5=cv2.GaussianBlur(ctx.gray, (odd(max(3, 5 / ctx.sN)),) * 2, 0),
                holes=[f for f in model["features"] if f["kind"] == "hole"],
                sign=1.0 if ctx.part["polarity"].startswith("dark") else -1.0)


@step("drawing_align", needs=("drawing_load",))
def drawing_align(ctx):
    """Align the drawing to the photo: the best of all starting alignments, refined."""
    D = ctx.results["drawing_load"]
    model, fixed = D["model"], D["fixed"]
    sim, reg, res, dpts = find_alignment(ctx, model, D["fixed_scale"], D["g5"], D["sign"])
    s = sim.s  # px per mm
    mm = 1.0 / s
    ctx.mm = mm
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
    ctx.data["drawing"] = dict(file=D["path"], info=D["info"], registration=reg,
                               outline_size_mm=[float(np.ptp(model["outline"]["pts"][:, 0])),
                                                float(np.ptp(model["outline"]["pts"][:, 1]))])
    return dict(sim=sim, reg=reg, res=res, dpts=dpts)


@step("drawing_outline", needs=("drawing_align",))
def drawing_outline(ctx):
    """Measure the part outline under the alignment, and the photo's measurement uncertainties."""
    D, A = ctx.results["drawing_load"], ctx.results["drawing_align"]
    return fit_outline(ctx, D["model"], A["sim"], A["dpts"], A["res"], A["reg"], D["fixed"], D["sign"])


@step("drawing_features", needs=("drawing_outline",))
def drawing_features(ctx):
    """Measure and judge the outline and every drawing feature, each by the check registered for its kind."""
    D, A, O = ctx.results["drawing_load"], ctx.results["drawing_align"], ctx.results["drawing_outline"]
    model, sim, res, reg, holes = D["model"], A["sim"], A["res"], A["reg"], D["holes"]
    det_pos = float(ctx.opt_get("detect_pos_tol") or C["tolerance"]["detect_position_mm"])
    det_size = float(ctx.opt_get("detect_size_tol") or C["tolerance"]["detect_size_mm"])
    dsf = _k("detection")["size_frac"]
    reg.update(detection_tolerance=dict(position_mm=f"max({det_pos}, {dsf:g} x nominal size)",
                                        size_mm=f"max({det_size}, {dsf:g} x nominal size)"))

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
            if g["id"] in on_recess and g["d"] / 2 * sim.local_scale(g["cx"], g["cy"]) \
                    < _k("hole_check")["recess_max_radius_px"]:
                ex = [v for k_, v in others.items() if k_ != g["id"] and k_ not in on_recess]
                res[g["id"]] = check_hole(ctx, g, sim, D["g5"], reach_k=_k("hole_check")["recess_reach_radii"],
                                          exclude=ex)

    judge = Judge(O["unc_edge"], O["unc_pos"], O["unc_size"], det_pos, det_size, on_recess)
    S = SimpleNamespace(model=model, sim=sim, res=res, mm=1.0 / sim.s, q=O["q"], off_o=O["off_o"],
                        found_o=O["found_o"], band0=O["band0"], judge=judge, rows=[], results=[])
    FEATURE_CHECKS["outline"](ctx, model["outline"], S)
    for f in model["features"]:
        FEATURE_CHECKS[kind_of(ctx, f, model, sim)](ctx, f, S)
    return dict(rows=S.rows, results=S.results, judge=judge)


@step("drawing_report", needs=("drawing_features",))
def drawing_report(ctx):
    """Table, counts, console summary, warnings and legend of the drawing check."""
    D, A, O, F = (ctx.results[n] for n in DRAWING_STEPS[:4])
    report(ctx, D["path"], D["model"], A["sim"], A["reg"], F["rows"], F["results"], O["q"], O, F["judge"], D["fixed"])
    return dict(sim=A["sim"], model=D["model"], results=F["results"])


@step("drawing_match", needs=("part",))
def drawing_match(ctx):
    """The whole drawing check (all drawing_* steps), for --steps lists written before it was split."""
    run(ctx, DRAWING_STEPS)
    return ctx.results["drawing_report"]
