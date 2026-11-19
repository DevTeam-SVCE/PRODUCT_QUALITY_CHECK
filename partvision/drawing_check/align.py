"""Aligning the drawing to the photo.

  register           outline-overlap search (coarse raster search + edge ICP)
  hole_hypotheses    alignments from pairs of drawing holes matched to round spots (independent of the outline)
  refine             joint fit on outline edge points + measured hole / opening centres
  perspective_refine homography on the measured holes for a photo taken at an angle
  find_alignment     runs all starting points, refines each, and picks the best candidate
"""
import itertools
import math

import cv2
import numpy as np

from .. import drawing as dwg
from ..config import C
from ..core import odd
from .checks import check_hole, check_opening
from .common import _k, _encloses, _nearest, _plain_openings, _resample
from .edges import _normals, edge_offsets
from .transform import Proj, Sim, fit_similarity


def register(ctx, model, fixed_scale, mirror_opt):
    k = _k("register")
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
    m = int(k["margin_frac"] * max(bw, bh))
    x0, y0 = bx - m, by - m
    f = k["raster_px"] / (max(bw, bh) + 2 * m)
    Wl, Hl = int((bw + 2 * m) * f) + 1, int((bh + 2 * m) * f) + 1
    img_mask = np.zeros((Hl, Wl), np.uint8)
    cv2.fillPoly(img_mask, [((cont - [x0, y0]) * f).astype(np.int32)], 1)
    g = cv2.resize(ctx.gray[max(y0, 0):by + bh + m, max(x0, 0):bx + bw + m], None, fx=f, fy=f)
    g = cv2.copyMakeBorder(g, max(0, -y0), 0, max(0, -x0), 0, cv2.BORDER_REPLICATE)[:Hl, :Wl]
    g = cv2.copyMakeBorder(g, 0, Hl - g.shape[0], 0, Wl - g.shape[1], cv2.BORDER_REPLICATE)
    D = np.abs(g.astype(np.float32) - cv2.medianBlur(g, odd(k["hole_bg_window_frac"] * k["raster_px"])).astype(np.float32))
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
                hs = float(np.mean(np.clip(D[q[ok, 1].astype(int), q[ok, 0].astype(int)] / k["hole_contrast_full"],
                                           0, 1)))
        return iou + k["hole_evidence_weight"] * hs, iou

    def make(theta, mir):
        M = np.diag([1.0, -1.0]) if mir else np.eye(2)
        c, sn = math.cos(theta), math.sin(theta)
        A = s0 * np.array([[c, -sn], [sn, c]]) @ M
        return Sim(s0, theta, ci - A @ cd, mir)

    for mir in mirrors:
        for a in np.arange(0, 360, k["coarse_step_deg"]):
            sc = score(make(math.radians(a), mir))
            if best is None or sc[0] > best[0]:
                best = (sc[0], sc[1], a, mir)
    _, _, a_best, mir = best
    for a in np.arange(a_best - k["fine_span_deg"], a_best + k["fine_span_deg"] + 0.01, k["fine_step_deg"]):
        sc = score(make(math.radians(a), mir))
        if sc[0] > best[0]:
            best = (sc[0], sc[1], a, mir)
    sim = make(math.radians(best[2]), best[3])
    coarse_iou = best[1]

    # refine on the outline edge (trimmed nearest-point ICP)
    dpts = _resample(opts, k["outline_samples"])
    size_px = s0 * model["outline"]["size"]
    for it in range(k["icp_iters"]):
        q = sim(dpts)
        idx, dist = _nearest(q, cont)
        thr = max(k["icp_gate_min_px"], size_px * (k["icp_gate_wide_frac"] if it < k["icp_wide_iters"]
                                                    else k["icp_gate_frac"]))
        keep = dist < thr
        if keep.sum() > k["min_points"]:
            keep &= dist <= max(k["icp_gate_min_px"], k["icp_trim_medians"] * np.median(dist[keep]))
        if keep.sum() < k["min_points"]:
            break
        sim = fit_similarity(dpts[keep], cont[idx[keep]], sim.mirror, fixed_scale=fixed_scale)
    q = sim(dpts)
    _, dist = _nearest(q, cont)
    return sim, dict(coarse_iou=coarse_iou, mirror=sim.mirror, angle_deg=math.degrees(sim.theta) % 360,
                     px_per_mm=sim.s, outline_rms_px=float(np.sqrt(np.mean(np.minimum(dist, k["rms_clip_medians"]
                                                                                     * np.median(dist)) ** 2))),
                     outline_within_2px=float(np.mean(dist <= k["within_px"]))), dpts


def _blobs(ctx, rmin, rmax, box, k_ref=None):
    """Round dark (blind/through, dark inside) and light (background seen through) spots: hole candidates.
    The local background is a median over ~6 hole radii; sized from the typical hole (k_ref), not the
    largest, so shadows and dark recesses nearby do not swamp small holes."""
    kb = _k("blobs")
    x0, y0, x1, y1 = box
    g = ctx.gray[y0:y1, x0:x1]
    k = odd(min(kb["median_window_max_px"], max(kb["median_window_min_px"], kb["median_window_radii"] * (k_ref or rmax))))
    med = cv2.medianBlur(g, k).astype(np.float32)
    out = []
    for dark, D in ((1.0, med - g.astype(np.float32)), (0.0, g.astype(np.float32) - med)):
        b = (D > C["image"]["spot_contrast"]).astype(np.uint8)
        n, lab, st, ce = cv2.connectedComponentsWithStats(b)
        for i in range(1, n):
            a = st[i, 4]
            r = math.sqrt(a / math.pi)
            w, h = st[i, 2], st[i, 3]
            if not (kb["size_ratio"][0] * rmin <= r <= kb["size_ratio"][1] * rmax) \
                    or max(w, h) > kb["max_aspect"] * min(w, h) or a < kb["min_fill"] * w * h:
                continue
            out.append((ce[i, 0] + x0, ce[i, 1] + y0, r, float(D[lab == i].mean()), dark))
    out.sort(key=lambda b: -b[3])
    return np.array(out[:kb["max_count"]], float).reshape(-1, 5)  # x, y, r, strength, dark(1)/light(0)


def hole_hypotheses(ctx, model, s0, mirrors, fixed_scale, k_best=None):
    """Alignments from pairs of drawing holes matched to pairs of round spots in the photo. Independent of
    the outline, so it still works when reflections or shadows spoil the part's silhouette."""
    k = _k("hole_pairs")
    k_best = k["keep_per_orientation"] if k_best is None else k_best
    nh = k["max_holes"]
    allh = sorted([f for f in model["features"] if f["kind"] == "hole"], key=lambda f: -f["d"])
    holes = allh[:nh] + [f for f in allh[nh:] if all(abs(f["d"] - g["d"]) > k["distinct_size_mm"]
                                                     for g in allh[:nh])][:k["extra_holes"]]
    if len(holes) < 2:
        return []
    Hc = np.array([[f["cx"], f["cy"]] for f in holes])
    Hr = np.array([f["d"] / 2 for f in holes])
    ft = k["fixed_scale_tol"]
    smin, smax = ((1 - ft) * fixed_scale, (1 + ft) * fixed_scale) if fixed_scale else \
        (k["scale_range"][0] * s0, k["scale_range"][1] * s0)
    bx, by, bw, bh = ctx.part["bbox"]
    m = int(k["margin_frac"] * max(bw, bh))
    box = (max(bx - m, 0), max(by - m, 0), min(bx + bw + m, ctx.W), min(by + bh + m, ctx.H))
    small = Hr[Hr < k["small_hole_frac"] * model["outline"]["size"]]  # bolt/fixing holes, not the bore
    k_ref = float(k["ref_size_factor"] * small.max() * s0) if len(small) else None  # largest fixing hole, never the bore
    B = _blobs(ctx, max(k["min_spot_px"], smin * Hr.min()), smax * Hr.max(), box, k_ref)
    if len(B) < 2:
        return []
    gb = cv2.GaussianBlur(ctx.gray, (odd(k["blur_px"]),) * 2, 0).astype(np.float32)
    G = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    gref = float(np.percentile(G[box[1]:box[3], box[0]:box[2]], k["edge_ref_percentile"])) + 1e-6
    opts = _resample(model["outline"]["pts"], k["outline_samples"])
    st_r, st_p = k["size_tol_radii"], k["size_tol_px"]
    hyps = []
    for i, j in itertools.combinations(range(len(holes)), 2):
        dd = np.linalg.norm(Hc[j] - Hc[i])
        if dd < 1e-6:
            continue
        for a, b in itertools.permutations(range(len(B)), 2):
            sc = np.linalg.norm(B[b, :2] - B[a, :2]) / dd
            if not smin <= sc <= smax:
                continue
            if abs(B[a, 2] - sc * Hr[i]) > st_r * sc * Hr[i] + st_p or abs(B[b, 2] - sc * Hr[j]) > st_r * sc * Hr[j] + st_p:
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
                match = (dist[np.arange(len(P)), nb] <= np.maximum(k["match_min_px"], k["match_radii"] * sc * Hr)) & \
                        (np.abs(B[nb, 2] - sc * Hr) <= st_r * sc * Hr + st_p)
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
    for sc_, sim in hyps[:k["max_hypotheses"]]:
        T = sim
        for tol_k in k["upgrade_match_radii"]:
            P = T(AHc)
            ls = T.s
            dist = np.hypot(P[:, None, 0] - B[None, :, 0], P[:, None, 1] - B[None, :, 1])
            nb = dist.argmin(1)
            m = (dist[np.arange(len(P)), nb] <= np.maximum(k["match_min_px"], tol_k * ls * AHr)) & \
                (np.abs(B[nb, 2] - ls * AHr) <= st_r * ls * AHr + st_p)
            if m.sum() < k["upgrade_min_holes"] or len(set(nb[m])) < k["upgrade_min_holes"]:
                break
            H, inl = cv2.findHomography(AHc[m].astype(np.float32), B[nb[m], :2].astype(np.float32), cv2.RANSAC,
                                        max(k["ransac_min_px"], k["ransac_radii"] * ls * float(np.median(AHr))))
            if H is None:
                break
            try:
                Tn = Proj(H, ref)
            except np.linalg.LinAlgError:
                break
            if Tn.mirror != sim.mirror or not k["scale_ratio"][0] < Tn.s / sim.s < k["scale_ratio"][1]:
                break
            T = Tn
        P = T(AHc)
        dist = np.hypot(P[:, None, 0] - B[None, :, 0], P[:, None, 1] - B[None, :, 1])
        nb = dist.argmin(1)
        ls = T.s
        n_ok = int(((dist[np.arange(len(P)), nb] <= np.maximum(k["tight_match_min_px"], k["tight_match_radii"] * ls * AHr)) &
                    (np.abs(B[nb, 2] - ls * AHr) <= st_r * ls * AHr + st_p)).sum())
        # perspective-upgraded guesses rank first (by tight matches); the rest keep their original score
        up.append(((1, n_ok + (sc_ - int(sc_))) if isinstance(T, Proj) else (0, sc_), T))
    hyps = [(k[1], T) for k, T in sorted(up, key=lambda h: (-h[0][0], -h[0][1]))]
    out = []
    for mir in mirrors:  # the best few for EACH orientation, so one cannot crowd out the other
        kept = []
        for sc_, sim in hyps:
            if sim.mirror != mir:
                continue
            if all(abs((sim.theta - o.theta + math.pi) % (2 * math.pi) - math.pi) > math.radians(k["same_angle_deg"]) or
                   np.linalg.norm(sim(ref[None])[0] - o(ref[None])[0]) > k["same_place_px"] for o in kept):
                kept.append(sim)
            if len(kept) >= k_best:
                break
        out += kept
    return out


def _iou(ctx, sim, opts):
    bx, by, bw, bh = ctx.part["bbox"]
    m = int(_k("hole_pairs")["margin_frac"] * max(bw, bh))
    x0, y0 = max(bx - m, 0), max(by - m, 0)
    x1, y1 = min(bx + bw + m, ctx.W), min(by + bh + m, ctx.H)
    dm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(dm, [(sim(opts) - [x0, y0]).astype(np.int32)], 1)
    im = (ctx.part["mask_outer"][y0:y1, x0:x1] > 0).astype(np.uint8)
    return float((dm & im).sum() / max((dm | im).sum(), 1))


def refine(ctx, model, sim, dpts, fixed_scale, holes, g5, sign):
    """Fit on measured edge points along the outline plus measured hole centres; score hole agreement."""
    k, ke = _k("refine"), _k("edges")
    size_px = sim.s * model["outline"]["size"]
    band = max(ke["band_min_px"], ke["band_frac"] * size_px)
    ok = np.zeros(len(dpts), bool)
    openings = [f for f in model["features"] if f["kind"] == "opening" and not _encloses(f, model)]
    info = {}
    for _ in range(k["iters"]):
        q = sim(dpts)
        off, st = edge_offsets(ctx, q, band, sign)
        ok = st >= max(ke["strong_min"], ke["strong_frac"] * float(np.median(st)))
        if ok.sum() > k["min_points"]:
            med = np.median(off[ok])
            mad = np.median(np.abs(off[ok] - med))
            ok &= np.abs(off - med) <= k["mad_k"] * 1.4826 * mad + k["mad_floor_px"]
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
            if "meas_x" in r and r["iou"] >= _k("opening_check")["datum_min_iou"]:
                src.append([f["cx"], f["cy"]])
                dst.append([r["meas_x"], r["meas_y"]])
        src, dst = np.array(src).reshape(-1, 2), np.array(dst).reshape(-1, 2)
        keep = np.ones(len(src), bool)
        if len(src) >= k["min_datums"]:
            e = np.hypot(*(sim(src) - dst).T)
            keep = e <= max(k["datum_outlier_min_px"], k["datum_outlier_medians"] * np.median(e))
        # inverse-variance weighting: each group counts by its own measured precision. Outline samples are
        # strongly correlated along the edge, so all of them together count as ~40 independent points.
        sig_o, sig_d = 1.0, 1.0
        if ok.sum() >= k["min_points"]:
            so = fit_similarity(dpts[ok], tgt[ok], sim.mirror, fixed_scale=fixed_scale)
            ro = np.hypot(*(so(dpts[ok]) - tgt[ok]).T)
            sig_o = max(k["sigma_floor_px"], 1.4826 * float(np.median(ro)))
            info["scale_outline"] = so.s
        if keep.sum() >= k["min_datums"]:
            sd = fit_similarity(src[keep], dst[keep], sim.mirror, fixed_scale=fixed_scale)
            rd = np.hypot(*(sd(src[keep]) - dst[keep]).T)
            sig_d = max(k["sigma_floor_px"], float(np.sqrt(np.mean(rd ** 2))))
            info["scale_features"] = sd.s
        S = np.vstack([dpts[ok], src[keep]])
        D = np.vstack([tgt[ok], dst[keep]])
        if len(S) < k["min_fit_points"] and keep.sum() < k["min_datums"]:
            break
        w = np.r_[np.full(ok.sum(), k["outline_equiv_points"] / max(ok.sum(), 1) / sig_o ** 2),
                  np.full(keep.sum(), 1.0 / sig_d ** 2)]
        info.update(outline_sigma_px=sig_o, feature_sigma_px=sig_d, n_datum=int(keep.sum()))
        sim = fit_similarity(S, D, sim.mirror, w=w, fixed_scale=fixed_scale)
        band = max(k["band_final_min_px"], band * k["band_shrink"])
    res = {f["id"]: check_hole(ctx, f, sim, g5) for f in holes}
    hole_score = _agreement(holes, res)
    info.update(score=hole_score + k["edge_score_weight"] * float(ok.mean()), hole_agreement=hole_score,
                edge_found=float(ok.mean()),
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




def _feature_agreement(ctx, model, sim, holes, res):
    """Alignment evidence from the inner features: hole agreement plus, for each plain opening (window, slot),
    its overlap with the opening segmented at the aligned place. Openings tell the orientation of a part whose
    silhouette is ambiguous (thick or stepped parts, where the outline is the union of both faces)."""
    vals = [_hole_agree(res[f["id"]]) for f in holes]
    for f in _plain_openings(model):
        r = check_opening(ctx, f, sim)
        vals.append(float(r.get("iou", 0.0)) if "meas_x" in r else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def perspective_refine(ctx, model, T, res, holes, g5, rounds=None):
    """A flat face photographed at an angle maps to the drawing by a homography. With 6+ measured holes,
    fit one (RANSAC) and keep it only when it brings clearly more holes onto their measured centres."""
    k = _k("perspective")
    rounds = k["rounds"] if rounds is None else rounds
    ref = np.array(dwg._centroid(model["outline"]["pts"]))
    best = (T, res, _agreement(holes, res))
    for _ in range(rounds):
        T_, res_, ag_ = best
        good = [f for f in holes if "meas_x" in res_[f["id"]] and res_[f["id"]].get("quality_ok")]
        if len(good) < k["min_holes"]:
            break
        src = np.array([[f["cx"], f["cy"]] for f in good], np.float32)
        dst = np.array([[res_[f["id"]]["meas_x"], res_[f["id"]]["meas_y"]] for f in good], np.float32)
        thr = max(k["ransac_min_px"], k["ransac_radii"] * float(np.median([res_[f["id"]]["exp_r"] for f in good])))
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, thr)
        if H is None or inl is None or inl.sum() < k["min_holes"]:
            break
        P = Proj(H, ref)
        if P.mirror != T.mirror or not k["scale_ratio"][0] < P.s / T.s < k["scale_ratio"][1]:
            break
        res2 = {f["id"]: check_hole(ctx, f, P, g5) for f in holes}
        ag2 = _agreement(holes, res2)
        if ag2 > ag_ + k["min_gain"]:
            best = (P, res2, ag2)
        else:
            break
    return best


def find_alignment(ctx, model, fixed_scale, g5, sign):
    """Every starting alignment (outline search for each orientation, hole-pair matches), refined; the best
    candidate wins. Returns (mapping, registration info, hole results at that mapping, outline sample points)."""
    kc, ke = _k("candidates"), _k("edges")
    holes = [f for f in model["features"] if f["kind"] == "hole"]
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
    w_h = min(1.0, max(kc["min_hole_weight"], (n_feat - kc["few_features"]) / (kc["full_weight_features"]
                                                                                 - kc["few_features"])))

    def cand_score(ag, ef, iou):
        return w_h * ag + kc["edge_weight"] * ef + (kc["iou_weight"] + kc["iou_weight_few"] * (1 - w_h)) * iou

    for sim0, reg0 in starts:
        if isinstance(sim0, Proj):
            if fixed_scale:
                continue
            res0 = {f["id"]: check_hole(ctx, f, sim0, g5) for f in holes}
            T, res_p, ag = perspective_refine(ctx, model, sim0, res0, holes, g5)
            q0 = T(dpts)
            off0, st0 = edge_offsets(ctx, q0, max(ke["band_min_px"], kc["perspective_band_frac"] * T.s
                                                  * model["outline"]["size"]), sign)
            ef = float((st0 >= max(ke["strong_min"], ke["strong_frac"] * float(np.median(st0)))).mean())
            reg0 = dict(reg0, coarse_iou=_iou(ctx, T, model["outline"]["pts"]), start="hole-pair match + perspective")
            fa = _feature_agreement(ctx, model, T, holes, res_p)
            cands.append((cand_score(fa, ef, reg0["coarse_iou"]), T, reg0, res_p,
                          dict(score=ag + _k("refine")["edge_score_weight"] * ef, hole_agreement=ag,
                               feature_agreement=fa, edge_found=ef,
                               perspective=True)))
            continue
        simr, res_h, rinfo = refine(ctx, model, sim0, dpts, fixed_scale, holes, g5, sign)
        if len(holes) >= _k("perspective")["min_holes"] and not fixed_scale:
            T, res_p, ag = perspective_refine(ctx, model, simr, res_h, holes, g5)
            if isinstance(T, Proj):
                simr, res_h = T, res_p
                rinfo = dict(rinfo, hole_agreement=ag, score=ag + _k("refine")["edge_score_weight"] * rinfo["edge_found"],
                             perspective=True,
                             feature_agreement=_feature_agreement(ctx, model, T, holes, res_p))
        reg0 = dict(reg0, coarse_iou=_iou(ctx, simr, model["outline"]["pts"]))
        cands.append((cand_score(rinfo["feature_agreement"], rinfo["edge_found"], reg0["coarse_iou"]), simr, reg0,
                      res_h, rinfo))
    # identical solutions found from different starts: keep one of each (they do not count as ambiguity)
    uniq = []
    for c in sorted(cands, key=lambda c: -c[0]):
        rp = np.array(dwg._centroid(model["outline"]["pts"]))[None]
        if all(c[1].mirror != u[1].mirror or abs((c[1].theta - u[1].theta + math.pi) % (2 * math.pi) - math.pi) >
               math.radians(kc["same_angle_deg"]) or np.linalg.norm(c[1](rp)[0] - u[1](rp)[0]) > kc["same_place_px"]
               for u in uniq):
            uniq.append(c)
    cands = uniq
    cands.sort(key=lambda c: -c[0])
    if len(cands) >= 2 and cands[0][1].mirror and not cands[1][1].mirror and cands[1][0] >= kc["mirror_tie"] * cands[0][0]:
        cands[0], cands[1] = cands[1], cands[0]  # a symmetric part fits both ways: report it as drawn
    _, sim, reg, res, rinfo = cands[0]
    reg.update(rinfo)
    if rinfo.get("scale_outline") and rinfo.get("scale_features"):
        ratio = rinfo["scale_features"] / rinfo["scale_outline"]
        reg["feature_to_outline_scale_ratio"] = ratio
        if abs(ratio - 1) > kc["scale_mismatch_warn"]:
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
        if kc["ambiguous"][0] * cands[0][0] < cands[1][0] < kc["ambiguous"][1] * cands[0][0]:
            ctx.warn("Two different alignments fit almost equally well (see registration.candidates in the "
                     "report); check the overlay, or force the orientation with --mirror yes/no.")
    return sim, reg, res, dpts
