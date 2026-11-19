"""Output of the drawing check: the measurement table, counts, console summary, warnings, view and legend."""
from ..config import C
from .common import CYAN, GREEN, MAG, _k


def report(ctx, path, model, sim, reg, rows, results, q, out, judge, fixed):
    """Write the table and summary of the checked features. out: result of outline.fit_outline."""
    ko = _k("outline_check")
    s = sim.s
    mm = 1.0 / s
    holes = [f for f in model["features"] if f["kind"] == "hole"]
    rms_px, unc_size, unc_pos, unc_edge = out["rms_px"], out["unc_size"], out["unc_pos"], out["unc_edge"]
    det_pos, det_size, dsf = judge.det_pos, judge.det_size, judge.dsf
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
    ctx.say(f"detected: {n_det} of {len(meas)} checked features (detection tolerance: position max({det_pos}, {dsf:g} x size), "
            f"size max({det_size}, {dsf:g} x size) mm); dimensional check +-tol: " + ", ".join(f"{v} {k}" for k, v in dims.items()))
    if reg["coarse_iou"] < ko["warn_min_iou"]:
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
    m_ = C["image"]["view_margin_frac"] * float((hi_ - lo_).max())
    ctx.view = (float(lo_[0] + hi_[0]) / 2, float(lo_[1] + hi_[1]) / 2, float(hi_[0] - lo_[0]) / 2 + m_,
                float(hi_[1] - lo_[1]) / 2 + m_)
    ctx.add_legend("dwg", "Cyan = drawing (expected); red edge = outline outside tolerance", CYAN, order=10)
    ctx.add_legend("st", "Measured: green OK, yellow-green DETECTED (found, outside +-tol), red DEVIATES, magenta MISSING",
                   GREEN, order=11)
    ctx.add_legend("info", f"Scale {mm:.4f} mm/px ({reg['scale_source']}).", MAG, order=99)
    ctx.drawing_model = model
