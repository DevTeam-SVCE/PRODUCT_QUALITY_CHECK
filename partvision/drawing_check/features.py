"""Per-feature checks of the drawing check, one function per feature kind, registered with @feature_check(kind).

Each takes (ctx, f, S): the drawing feature f and the shared state S of the check (mapping, hole results,
uncertainties, Judge, output rows). It measures the feature, judges it, draws it on the overlay and appends one
table row and one result. Add a new drawing feature kind by registering a function for it here.
"""
import math

import numpy as np

from ..config import C
from .checks import check_contour, check_opening
from .common import CYAN, RED, STATUS_COL, _encloses, _k, _runs
from .transform import Proj

FEATURE_CHECKS = {}


def feature_check(kind):
    def deco(fn):
        FEATURE_CHECKS[kind] = fn
        return fn
    return deco


def kind_of(ctx, f, model, sim):
    """Which check applies to drawing feature f. An opening that encloses other features is a recess/step edge,
    and an opening not found as a region is checked by its edge too: both as an inner contour."""
    if f["kind"] == "hole":
        return "hole"
    if f["kind"] == "opening" and (_encloses(f, model) or check_opening(ctx, f, sim).get("status") == "MISSING"):
        return "contour"
    if f["kind"] == "opening":
        return "opening"
    return f["kind"] if f["kind"] in FEATURE_CHECKS else "other"


@feature_check("outline")
def check_outline_feature(ctx, f, S):
    """The part outline: measured edge along each normal of the drawing outline (+ = part larger than drawing)."""
    model, mm, q, found_o, ko = S.model, S.mm, S.q, S.found_o, _k("outline_check")
    verdict, combine, det_pos, rows, results = S.judge.verdict, S.judge.combine, S.judge.det_pos, S.rows, S.results
    sd = S.off_o * mm
    otol = model["outline"]["tol"]
    out_bad = found_o & (np.abs(sd) > otol + C["tolerance"]["outline_guard_px"] * mm)  # edges come in 0.5 px steps
    if found_o.mean() < ko["min_visible"]:
        st_o, note_o = "UNCERTAIN", f"edge visible on only {found_o.mean():.0%} of the outline"
    else:
        st_o, note_o = verdict(float(np.percentile(np.abs(sd[found_o]), ko["p95_percentile"])), None,
                               dict(tol=otol, pos_tol=otol))
    dim_o = st_o
    det_o = bool(found_o.mean() >= ko["min_visible"] and np.median(np.abs(sd[found_o])) <= max(det_pos, ko["detect_min_mm"])) \
        if found_o.any() else False
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


@feature_check("hole")
def check_hole_feature(ctx, f, S):
    sim, res, on_recess, rows, results = S.sim, S.res, S.judge.on_recess, S.rows, S.results
    verdict, detected, combine = S.judge.verdict, S.judge.detected, S.judge.combine
    unc_pos, unc_size, det_pos, dsf = S.judge.unc_pos, S.judge.unc_size, S.judge.det_pos, S.judge.dsf
    fid = f["id"]
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
        parallax = fid in on_recess and isinstance(sim, Proj) and dpos_v > max(det_pos, dsf * f["d"])
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


@feature_check("contour")
def check_contour_feature(ctx, f, S):
    sim, mm, band0, ko, rows, results = S.sim, S.mm, S.band0, _k("outline_check"), S.rows, S.results
    verdict, combine, det_pos = S.judge.verdict, S.judge.combine, S.judge.det_pos
    fid = f["id"]
    r = check_contour(ctx, f, sim, band0)
    dev = r["off"] * mm
    fr = float(r["found"].mean())
    if fr < ko["contour_min_visible"]:
        st, note = "UNCERTAIN", f"edge visible on only {fr:.0%} of the contour"
    else:
        st, note = verdict(float(np.percentile(np.abs(dev[r["found"]]), ko["contour_percentile"])), None, f)
    dim = st
    det = bool(fr >= ko["contour_min_visible"] and np.median(np.abs(dev[r["found"]]))
               <= max(det_pos, ko["detect_min_mm"])) if r["found"].any() else False
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


@feature_check("opening")
def check_opening_feature(ctx, f, S):
    sim, rows, results = S.sim, S.rows, S.results
    verdict, detected, combine = S.judge.verdict, S.judge.detected, S.judge.combine
    fid = f["id"]
    r = check_opening(ctx, f, sim)
    st, note = r.get("status"), r.get("note", "")
    mx = my = mL = dpos = ""
    det = None
    if "meas_x" in r:
        c_exp = sim(np.array([[f["cx"], f["cy"]]]))[0]
        ls = sim.local_scale(f["cx"], f["cy"])
        dpos_v = math.hypot(r["meas_x"] - c_exp[0], r["meas_y"] - c_exp[1]) / ls
        dL, dW = (r["meas_L"] / ls - f["length"]), (r["meas_W"] / ls - f["width"])
        st, note = verdict(dpos_v, max(dL, dW, key=abs), f, extra_bad=r["iou"] < _k("opening_check")["ok_min_iou"])
        det = r["iou"] >= _k("opening_check")["detected_min_iou"] \
            and detected(dpos_v, max(dL, dW, key=abs), min(f["length"], f["width"]))
        note = (note + "; " if note else "") + f"overlap with drawing {r['iou']:.0%}; L {dL:+.2f} W {dW:+.2f} mm" \
            + f" ({r['method']})"
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


@feature_check("other")
def not_measured_feature(ctx, f, S):
    """Concentric rings and anything else: listed, not measured from the photo."""
    rows, results, fid = S.rows, S.results, f["id"]
    rows.append([fid, f["kind"], "", f"{f['cx']:.3f}", f"{f['cy']:.3f}", f"{f['d']:.3f}", "", "", "", "", "",
                 "", "NOT MEASURED",
                 "concentric ring (counterbore/chamfer/boss edge)" if f["kind"] == "ring" else "visual check",
                 "", ""])
    results.append(dict(id=fid, kind=f["kind"], status="NOT MEASURED"))
