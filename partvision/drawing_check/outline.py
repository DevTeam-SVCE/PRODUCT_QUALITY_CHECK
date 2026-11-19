"""The part outline under the chosen alignment: its own alignment when it sits at a different height from the
holes, the measured edge along its normals, and the photo's measurement uncertainties derived from them."""
import math

import numpy as np

from .common import _k, _nearest
from .edges import _normals, edge_offsets, inner_limits
from .transform import fit_similarity


def fit_outline(ctx, model, sim, dpts, res, reg, fixed, sign):
    """Measure the outline edge and the uncertainties (edge, hole position, hole size) in mm."""
    ke, ko = _k("edges"), _k("outline_check")
    s = sim.s
    mm = 1.0 / s
    q = sim(dpts)
    # Outline at a different height from the holes/openings (stepped or stacked part, close camera): one
    # 2-D mapping cannot fit both. If aligning the outline on its own fits its edge clearly better, check the
    # outline's SHAPE with that alignment; holes and openings keep the joint alignment.
    band0 = max(ke["band_min_px"], ke["band_frac"] * s * model["outline"]["size"])
    strong = lambda st_: st_ >= max(ke["strong_min"], ke["strong_frac"] * float(np.median(st_)))
    hole_px = [(*sim(np.array([[g_["cx"], g_["cy"]]]))[0], g_["d"] / 2 * sim.local_scale(g_["cx"], g_["cy"]))
               for g_ in model["features"] if g_["kind"] == "hole"]

    def _p95(qq):
        o_, s_ = edge_offsets(ctx, qq, band0, sign, inner_limits(qq, hole_px))
        k_ = strong(s_)
        return (float(np.percentile(np.abs(o_[k_]), ko["p95_percentile"])) if k_.sum() > _k("refine")["min_points"]
                else 99.0), o_, s_, k_
    p_joint, o_j, s_j, k_j = _p95(q)
    own = None
    npts = _k("refine")["min_points"]
    if k_j.sum() > npts and not fixed:
        cur = fit_similarity(dpts[k_j], (q + o_j[:, None] * _normals(q))[k_j], sim.mirror)
        for _ in range(ko["own_iters"]):
            qq = cur(dpts)
            o_, s_ = edge_offsets(ctx, qq, max(ko["own_band_min_px"], ko["own_band_frac"] * band0), sign,
                                  inner_limits(qq, hole_px))
            k_ = strong(s_)
            if k_.sum() < npts:
                break
            cur = fit_similarity(dpts[k_], (qq + o_[:, None] * _normals(qq))[k_], sim.mirror)
        p_own = _p95(cur(dpts))[0]
        # second candidate: fit to the segmented silhouette (robust where a raised plate's edges distract)
        cont_ = ctx.part["contour"][:, 0, :].astype(float)
        sil = fit_similarity(dpts, q, sim.mirror)
        for _ in range(ko["silhouette_iters"]):
            qq = sil(dpts)
            idx_, dist_ = _nearest(qq, cont_)
            kk = dist_ < max(ko["silhouette_trim_min_px"], ko["silhouette_trim_medians"] * np.median(dist_))
            if kk.sum() < npts:
                break
            sil = fit_similarity(dpts[kk], cont_[idx_[kk]], sim.mirror)
        p_sil = _p95(sil(dpts))[0]
        if p_sil < p_own:
            cur, p_own = sil, p_sil
        reg.update(outline_p95_px_joint=p_joint, outline_p95_px_own=p_own, outline_own_scale=cur.s)
        if p_own < ko["own_better_ratio"] * p_joint - ko["own_better_px"]:
            own = cur
            q = cur(dpts)
            ctx.warn(f"The outline and the holes/openings do not fit one flat mapping (outline-only fit: scale "
                     f"{100 * (cur.s / s - 1):+.1f}%, edge error {p_own / cur.s:.2f} mm vs {p_joint / s:.2f} mm with the "
                     "holes): they are at different heights (stepped or stacked part) or the camera is close and "
                     "angled. The outline SHAPE is checked with its own alignment; its position relative to the "
                     "holes is not checked from this photo.")
    reg["outline_alignment"] = "own (shape check)" if own else "joint with holes/openings"
    # outline: measured edge along each normal of the drawing outline (+ = part larger than drawing)
    off_o, st_o_ = edge_offsets(ctx, q, band0, sign, inner_limits(q, hole_px))
    found_o = strong(st_o_)
    fo = off_o[found_o]
    rms_px = float(np.sqrt(np.mean((fo - np.median(fo)) ** 2))) if len(fo) else 99.0
    reg.update(outline_rms_px=rms_px, outline_rms_mm=rms_px * mm, outline_edge_found=float(found_o.mean()))
    unc_edge = math.sqrt(ko["edge_unc_px"] ** 2 + rms_px ** 2) * mm  # edges/contours: 1 px + scatter of the outline edge
    # holes: size is a direct 1 px edge measurement; position adds the scatter of hole centres about the fit
    e_h = [math.hypot(r["meas_x"] - r["exp_x"], r["meas_y"] - r["exp_y"]) for r in res.values()
           if "meas_x" in r and r.get("quality_ok")]
    # robust: 2-D scatter from the median distance (rms ~ 1.2 x median), so a misplaced hole cannot hide itself
    hole_rms_px = float(ko["hole_rms_from_median"] * np.median(e_h)) if len(e_h) >= ko["min_holes_for_scatter"] \
        else rms_px
    unc_pos = math.sqrt(ko["edge_unc_px"] ** 2 + hole_rms_px ** 2) * mm
    unc_size = ko["edge_unc_px"] * mm
    reg.update(uncertainty_edge_mm=unc_edge, uncertainty_hole_position_mm=unc_pos, uncertainty_hole_size_mm=unc_size,
               hole_centre_rms_px=hole_rms_px)
    return dict(q=q, band0=band0, off_o=off_o, found_o=found_o, rms_px=rms_px, unc_edge=unc_edge,
                unc_pos=unc_pos, unc_size=unc_size)
