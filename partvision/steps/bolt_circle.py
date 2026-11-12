"""Round holes on a common circle about the origin (bolt circle), for round parts."""
import math

import cv2
import numpy as np

from ..core import angle_deg, fit_circle, odd
from ..holefit import measure_round_hole
from ..config import C as CFG
from ..registry import step

GREEN, BLUE, WHITE, RED, YEL, CYAN = ((40, 220, 40), (40, 170, 255), (255, 255, 255), (255, 70, 70), (255, 240, 40),
                                      (60, 230, 230))


@step("bolt_circle", needs=("part",))
def bolt_circle(ctx):
    """Hough sweep -> dominant radius -> validate (hole-dark core, round wall) -> refine, type, pitch."""
    W, H, sN = ctx.W, ctx.H, ctx.sN
    bx, by = ctx.origin[:2]
    oR = ctx.part["r"]
    bore = ctx.results.get("bore", {}).get("bore")
    rR = ctx.results.get("round_rim", {}).get("r", oR)
    g5 = cv2.GaussianBlur(ctx.gray, (odd(max(3, 5 / sN)),) * 2, 0)
    gN5 = cv2.GaussianBlur(ctx.gN, (5, 5), 0)

    def hough(p2, rmin, rmax, mind, param1=100):
        c = cv2.HoughCircles(gN5, cv2.HOUGH_GRADIENT, 1, mind * sN, param1=param1, param2=p2,
                             minRadius=max(2, int(rmin * sN)), maxRadius=max(3, int(math.ceil(rmax * sN))))
        return np.zeros((0, 3)) if c is None else c[0] / sN

    def dominant_radius(cands):
        rad = np.hypot(cands[:, 0] - bx, cands[:, 1] - by)
        lo, hi = 0.3 * oR, 1.05 * oR
        if bore:
            lo = max(lo, 1.3 * bore["r"])
        sel = (rad > lo) & (rad < hi)
        if sel.sum() < 3:
            return None, rad
        bw = max(2.0, 0.02 * oR)
        hist, ed = np.histogram(rad[sel], bins=np.arange(lo, hi + bw, bw))
        k = int(hist.argmax())
        near = sel & (rad >= ed[k] - bw) & (rad < ed[k + 1] + bw)
        return float(np.median(rad[near])), rad

    # coarse pass: broad radius range, only to learn the bolt-circle radius and the hole size
    rmin0, rmax0 = 0.02 * oR, 0.09 * oR
    dom = None
    for P1 in (100, 60):
        for p2c in (14, 10, 8):
            c0 = hough(p2c, rmin0, rmax0, 1.5 * rmin0, P1)
            dom, rad0 = dominant_radius(c0) if len(c0) else (None, None)
            if dom:
                inb = np.abs(rad0 - dom) / dom <= 0.03
                if inb.sum() >= 3:
                    break
                dom = None
        if dom:
            break
    if not dom:
        ctx.warn("No bolt circle found: fewer than 3 hole-like circles share a radius about the centre.")
        return dict(ok=False)
    r_med0 = float(np.median(c0[inb, 2]))
    ctx.data["coarse_pass"] = dict(radius_range=[round(rmin0, 1), round(rmax0, 1)], param1=P1, param2=p2c,
                                   dominant_R=dom, typical_hole_r=r_med0)

    # fine pass: radius range from the typical hole, param2 sweep
    RMIN, RMAX, MIND = max(3 / sN, 0.6 * r_med0), 1.5 * r_med0, 1.5 * r_med0
    f2 = max(1.0, r_med0 * sN / 12)  # accumulator votes grow with the (normalised) hole circumference
    P2S = sorted(set(int(round(v * f2)) for v in (10, 12, 14, 16, 18, 20, 24, 28)))
    P2_BASE = int(round(14 * f2))
    ctx.data["hole_search"] = dict(radius_range=[round(RMIN, 1), round(RMAX, 1)], minDist=round(MIND, 1), param1=P1,
                                   param2_values=P2S, param2_base=P2_BASE,
                                   prescribed_width_scaled_range=[round(6 * W / 1200), round(16 * W / 1200)])
    runs = {p2: hough(p2, RMIN, RMAX, MIND, P1) for p2 in P2S}
    dom2, _ = dominant_radius(runs[P2_BASE]) if len(runs[P2_BASE]) else (None, None)
    dom = dom2 or dom
    ctx.data["dominant_radius_hough"] = dom

    pool = []  # union over the sweep, strongest param2 first
    for p2 in sorted(P2S, reverse=True):
        for x, y, r in runs[p2]:
            if all(math.hypot(x - q["x"], y - q["y"]) >= 0.6 * r_med0 for q in pool):
                pool.append(dict(x=float(x), y=float(y), r=float(r), max_param2=p2))
    for q in pool:
        q["R"] = float(math.hypot(q["x"] - bx, q["y"] - by))
        q["ang"] = angle_deg(q["x"], q["y"], bx, by)
        q["dev_pct"] = (q["R"] - dom) / dom * 100
        q["in_band"] = abs(q["dev_pct"]) <= 3
        q["core"] = ctx.disc_stat(q["x"], q["y"], 0, max(2, 0.33 * r_med0))
        q["surface"] = ctx.disc_stat(q["x"], q["y"], 1.6 * r_med0, 2.1 * r_med0)
    ref = [q for q in pool if q["in_band"] and q["max_param2"] >= P2_BASE] or [q for q in pool if q["in_band"]]
    core_med = float(np.median([q["core"] for q in ref]))
    contrast_med = float(np.median([q["surface"] - q["core"] for q in ref]))  # >0: holes darker than surface
    hole_pol = -1.0 if contrast_med > 0 else 1.0
    C = abs(contrast_med)
    ctx.data["hole_levels"] = dict(core_median=core_med, surface_minus_core_median=contrast_med,
                                   holes_are="darker than surface" if hole_pol < 0 else "brighter than surface")
    if C < CFG["image"]["min_contrast"]:
        ctx.warn(f"Holes differ from the surface by only {C:.0f} gray levels: hole validation is unreliable.")
    for q in pool:
        local = (q["core"] - q["surface"]) * hole_pol
        q["hole_like"] = bool(abs(q["core"] - core_med) <= 0.35 * C and local >= 0.5 * C)

    holes, off_circle, rejected = [], [], []
    for q in pool:
        if q["hole_like"]:
            m = measure_round_hole(ctx, q["x"], q["y"], r_med0, hole_pol, g5)
            if m is None:
                q["reason"] = "hole-coloured core but no closed wall could be traced"
                rejected.append(q)
                continue
            m["max_param2"] = q["max_param2"]
            if any(math.hypot(m["x"] - o["x"], m["y"] - o["y"]) < 0.5 * r_med0 for o in holes + off_circle):
                continue  # two seeds converged on the same hole
            if q["in_band"]:
                holes.append(m)
            elif m["quality_ok"] and 0.6 <= m["r"] / r_med0 <= 1.6:
                off_circle.append(m)
            else:
                q["reason"] = ("off bolt circle by %.1f%% and outline not round (wall fit %.2f px, %d/%d rays): "
                               "likely a pocket, shadow or recess" % (q["dev_pct"], m["fit_std"], m["rays"], m["of"]))
                rejected.append(q)
        else:
            why = []
            if not q["in_band"]:
                why.append("off bolt circle by %.1f%% (limit 3%%)" % q["dev_pct"])
            why.append("core gray %.0f vs typical hole core %.0f, local surface %.0f: not hole-like (surface mark/edge)"
                       % (q["core"], core_med, q["surface"]))
            q["reason"] = "; ".join(why)
            rejected.append(q)
    if len(holes) < 3:
        ctx.warn(f"Only {len(holes)} validated holes on the bolt circle; no pattern measured.")
        return dict(ok=False)

    for h in holes + off_circle:
        h["R"] = float(math.hypot(h["x"] - bx, h["y"] - by))
        h["ang"] = angle_deg(h["x"], h["y"], bx, by)
    holes.sort(key=lambda h: h["ang"])
    angs = np.array([h["ang"] for h in holes])
    gaps = (np.roll(angs, -1) - angs) % 360
    uniform = gaps.max() - gaps.min() < 1.0
    if uniform:
        s = int(np.argmin([min(abs(a - 270), 360 - abs(a - 270)) for a in angs]))
        numbering = "pitch is uniform, so hole 1 is the hole nearest 12 o'clock"
    else:
        s = int(np.argmax(gaps))
        numbering = "hole 1 is the hole just before the largest angular gap"
    ctx.data["numbering"] = numbering
    holes = holes[s:] + holes[:s]
    for i, h in enumerate(holes, 1):
        h["n"] = i
        if not h["quality_ok"]:
            ctx.warn(f"Hole {i}: weak wall fit (scatter {h['fit_std']:.2f} px, {h['rays']}/{h['of']} rays"
                     + (f", blob diameter {h['blob_d']:.1f} vs {h['d']:.1f}" if h["blob_d"] else "") + ").")
        if h["max_param2"] < P2_BASE:
            ctx.warn(f"Hole {i} was only found at the permissive param2={h['max_param2']}; check it on the image.")
    off_circle.sort(key=lambda h: (h["R"], h["ang"]))
    for i, h in enumerate(off_circle, 1):
        h["n"] = "A%d" % i
    if off_circle:
        ctx.warn(f"{len(off_circle)} hole-like feature(s) off the main bolt circle (A1...): verify them on the image.")

    # hole types, relative to this part's own typical hole
    r_med = float(np.median([h["r"] for h in holes]))
    d_med = 2 * r_med
    for h in holes:
        if h["ring_dip"] > 0.25 * max(h["local_contrast"], 1) and h["ring_dip_at_r"] < h["r"] + 0.5 * r_med \
                and h["ring_outer_r"]:
            h["type"] = "ringed"
            h["ring_d"] = 2 * h["ring_outer_r"]
        elif h["d"] < 0.9 * d_med:
            h["type"] = "small"
        elif h["d"] > 1.1 * d_med:
            h["type"] = "large"
        else:
            h["type"] = "standard"
    types = {}
    for h in holes:
        types.setdefault(h["type"], []).append(h["n"])
    ctx.data["hole_types"] = types
    if r_med < 8:  # real pixels: below this the groove/ring signal is only 1-2 px wide
        ctx.warn(f"Holes are only about {2 * r_med:.0f} px across: hole types (ringed/small/large) and diameters "
                 "are unreliable.")
    groups = {}
    for t, ns in types.items():
        if t != "standard" and len(ns) >= 2:
            a = sorted(holes[n - 1]["ang"] for n in ns)
            sp = [round((a[(i + 1) % len(a)] - a[i]) % 360, 2) for i in range(len(a))]
            groups[t] = dict(holes=ns, spacing_deg=sp, equally_spaced=bool(max(abs(v - 360 / len(a)) for v in sp) < 1.0))
    ctx.data["type_group_spacing"] = groups

    # param2 sensitivity, deduplicated against the final holes
    fx = np.array([[h["x"], h["y"]] for h in holes])
    tbl = []
    for p2 in P2S:
        c = runs[p2]
        rr = np.hypot(c[:, 0] - bx, c[:, 1] - by) if len(c) else np.zeros(0)
        inb = np.abs(rr - dom) / dom <= 0.03
        dmin = np.array([np.hypot(*(fx - c[i, :2]).T).min() for i in range(len(c))]) if len(c) else np.zeros(0)
        near = dmin < 0.6 * r_med
        uniq = {int(np.hypot(*(fx - c[i, :2]).T).argmin()) for i in np.where(near)[0]}
        tbl.append(dict(param2=p2, raw=int(len(c)), within_3pct=int(inb.sum()), distinct_final_holes=len(uniq),
                        within_3pct_but_rejected=int((inb & ~near).sum())))
    ctx.data["param2_table"] = tbl
    ctx.data["rejected_candidates"] = [{k: q[k] for k in ("x", "y", "r", "R", "ang", "dev_pct", "core", "surface",
                                                          "max_param2", "reason")} for q in rejected]

    # independent census: hole-coloured round blobs near the bolt circle, no Hough involved
    thr = core_med + 0.3 * contrast_med
    b = (((g5.astype(np.float32) - thr) * hole_pol) > 0).astype(np.uint8)
    n, lab, st, ce = cv2.connectedComponentsWithStats(b)
    census = []
    amin, amax = 0.3 * math.pi * r_med ** 2, 3.0 * math.pi * r_med ** 2
    for i in range(1, n):
        rr = float(math.hypot(ce[i, 0] - bx, ce[i, 1] - by))
        if abs(rr - dom) < 2 * r_med and amin < st[i, 4] < amax:
            x_, y_, w_, h_ = st[i, :4]
            cc, _ = cv2.findContours((lab[y_:y_ + h_, x_:x_ + w_] == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
            c_ = max(cc, key=cv2.contourArea)
            circ = 4 * math.pi * cv2.contourArea(c_) / max(cv2.arcLength(c_, True) ** 2, 1)
            if circ >= 0.6:
                census.append(dict(x=float(ce[i, 0]), y=float(ce[i, 1]), area=int(st[i, 4]), R=rr, circularity=circ,
                                   matches_hole=bool(np.hypot(*(fx - ce[i, :2]).T).min() < 0.5 * r_med)))
    matched = sum(c["matches_hole"] for c in census)
    ctx.data["blob_census"] = dict(count=len(census), matched=matched, threshold=thr, items=census)
    if len(census) != len(holes) or matched != len(holes):
        ctx.warn(f"Independent blob count disagrees: {len(census)} round blobs near the bolt circle "
                 f"({matched} match the {len(holes)} holes). Check the annotated image for missed or extra holes.")

    # bolt circle, tilt
    xy = np.array([[h["x"], h["y"]] for h in holes], np.float32)
    bcx, bcy, bcR, bcstd, _, _ = fit_circle(xy, iters=1)
    Rs = np.array([h["R"] for h in holes])
    ctx.data["bolt_circle"] = dict(mean_R=float(Rs.mean()), std_R=float(Rs.std()), min_R=float(Rs.min()),
                                   max_R=float(Rs.max()), fit_x=bcx, fit_y=bcy, fit_R=bcR, fit_d=2 * bcR,
                                   fit_std=bcstd, offset_from_origin=float(math.hypot(bcx - bx, bcy - by)))
    tilt = {}
    if len(holes) >= 5:
        el = cv2.fitEllipse(xy)
        axr = min(el[1]) / max(el[1])
        tilt.update(bolt_axis_ratio=float(axr), bolt_tilt_deg=float(math.degrees(math.acos(min(axr, 1)))))
    if bore and bore["ellipse"]:
        bax = min(bore["ellipse"][1]) / max(bore["ellipse"][1])
        tilt.update(bore_axis_ratio=float(bax), bore_tilt_deg=float(math.degrees(math.acos(min(bax, 1)))))
    ctx.data["tilt"] = tilt
    if tilt.get("bolt_tilt_deg", 0) > 6:
        ctx.warn(f"The bolt circle looks elliptical (axis ratio {tilt['bolt_axis_ratio']:.3f}, about "
                 f"{tilt['bolt_tilt_deg']:.0f} deg of tilt): angles, pitches and ratios are distorted, and short or "
                 "shallow notches may be tilt artefacts. Retake top-down.")
    if Rs.std() > 0.01 * dom:
        ctx.warn(f"Hole distances from the centre vary by {Rs.std():.1f} px (std): tilt, an off-centre bore, "
                 "or a wrong hole.")
    if bore and ctx.data["bolt_circle"]["offset_from_origin"] > 0.01 * oR:
        ctx.warn(f"Bolt-circle centre is {ctx.data['bolt_circle']['offset_from_origin']:.1f} px from the bore centre.")

    # spacing
    spacing = []
    for i, h in enumerate(holes):
        nx = holes[(i + 1) % len(holes)]
        spacing.append(dict(a=h["n"], b=nx["n"], pitch=float((nx["ang"] - h["ang"]) % 360),
                            dist=float(math.hypot(nx["x"] - h["x"], nx["y"] - h["y"]))))
    pitches = np.array([p["pitch"] for p in spacing])
    typ = float(np.median(pitches))
    for p in spacing:
        p["big"] = bool(p["pitch"] > 1.5 * typ)
    ctx.data["pitch"] = dict(median=typ, big_gap_threshold=1.5 * typ, min=float(pitches.min()),
                             max=float(pitches.max()), uniform=bool(uniform),
                             big_gaps=[(p["a"], p["b"]) for p in spacing if p["big"]])
    ctx.data["holes"] = holes
    ctx.data["off_circle_holes"] = off_circle
    ctx.data["spacing"] = spacing
    ctx.dims["bolt"] = 2 * bcR
    ctx.dims["hole"] = d_med

    # outputs
    rows = []
    for h in holes + off_circle:
        rows.append([h["n"], "bolt circle" if h in holes else "off circle", f'{h["ang"]:.2f}', f'{h["R"]:.2f}',
                     f'{h["d"]:.2f}', f'{h["x"]:.2f}', f'{h["y"]:.2f}', h.get("type", ""),
                     f'{h["ring_d"]:.2f}' if "ring_d" in h else ""])
    ctx.table("per-hole table; angles clockwise on screen about the %s; group off circle = not on the bolt circle"
              % ctx.origin[2],
              ["hole", "group", "angle_deg", "radius_from_centre_px", "diameter_px", "x_px", "y_px", "type",
               "ring_outer_diameter_px"], rows,
              length_cols=["radius_from_centre_px", "diameter_px", "ring_outer_diameter_px"])
    ctx.table("neighbour-spacing table; big_gap means pitch > 1.5 x median pitch (%.2f deg)" % (1.5 * typ),
              ["from_hole", "to_hole", "pitch_deg", "centre_distance_px", "big_gap"],
              [[p["a"], p["b"], f'{p["pitch"]:.2f}', f'{p["dist"]:.2f}', "yes" if p["big"] else "no"] for p in spacing],
              length_cols=["centre_distance_px"])
    ctx.feature("bolt circle", f"{2 * bcR:.2f}", note=f"{len(holes)} holes")

    ctx.draw("circle", z=0, x=bcx, y=bcy, r=bcR, color=BLUE, width=1)
    for h in holes:
        ctx.draw("circle", z=4, x=h["x"], y=h["y"], r=h["r"] + 0.3 * r_med, color=GREEN, width=2.5)
        x, y = ctx.polar(h["R"] - (h["r"] + 0.055 * rR), h["ang"])
        ctx.draw("text", z=6, x=x, y=y, text=str(h["n"]), color=YEL, size=13)
    for h in off_circle:
        ctx.draw("circle", z=4, x=h["x"], y=h["y"], r=h["r"] + 0.3 * r_med, color=CYAN, width=2.5)
        x, y = ctx.polar(h["R"] - (h["r"] + 0.055 * rR), h["ang"])
        ctx.draw("text", z=6, x=x, y=y, text=h["n"], color=CYAN, size=13)
    for p, h in zip(spacing, holes):
        x, y = ctx.polar(rR * (1.15 if p["big"] else 1.103), h["ang"] + p["pitch"] / 2)
        ctx.draw("text", z=6, x=x, y=y, text=f'{p["pitch"]:.0f}', color=RED if p["big"] else WHITE,
                 size=16 if p["big"] else 14)
    for q in rejected:
        ctx.draw("circle", z=9, debug=True, x=q["x"], y=q["y"], r=q["r"], color=(255, 0, 0), width=1)
    ctx.add_legend("holes", f"Green + number = hole ({len(holes)} on bolt circle; {numbering})", GREEN, order=10)
    ctx.add_legend("pitch", "Outer figures = angle to next hole in degrees (red = big gap, > 1.5 x median pitch)",
                   (255, 110, 90), order=30)
    if off_circle:
        ctx.add_legend("off", "Cyan + A# = hole-like feature off the bolt circle (verify)", CYAN, order=40)
    ctx.add_legend("info", "Blue ring = bolt circle.", BLUE, order=91)

    ctx.say(f"hole search radius {RMIN:.1f}-{RMAX:.1f} px (typical r {r_med0:.1f}), param2 sweep {P2S}")
    for t in tbl:
        ctx.say("  param2 %3d: raw %3d, within 3%% %3d, distinct holes %3d, in-band rejected %d" % (
            t["param2"], t["raw"], t["within_3pct"], t["distinct_final_holes"], t["within_3pct_but_rejected"]))
    ctx.say(f"holes on bolt circle: {len(holes)}  (blob census {len(census)}, matched {matched})  off-circle: "
            f"{len(off_circle)}  rejected candidates: {len(rejected)}")
    ctx.say(f"bolt circle d={2 * bcR:.2f} px, hole R std {Rs.std():.2f} px; types {types}")
    for h in holes + off_circle:
        ctx.say("%3s %-8s ang=%6.2f R=%7.2f d=%6.2f (%7.1f,%7.1f) fit=%.2f rays=%d/%d blob_d=%s%s" % (
            h["n"], h.get("type", "-"), h["ang"], h["R"], h["d"], h["x"], h["y"], h["fit_std"], h["rays"], h["of"],
            "%.1f" % h["blob_d"] if h["blob_d"] else "-", " ring_d=%.1f" % h["ring_d"] if "ring_d" in h else ""))
    for p in spacing:
        ctx.say("%2d->%2d pitch=%6.2f dist=%7.1f %s" % (p["a"], p["b"], p["pitch"], p["dist"], "BIG" if p["big"] else ""))
    return dict(ok=True, holes=holes, off_circle=off_circle, dom=dom, r_med=r_med, r_med0=r_med0,
                bcx=bcx, bcy=bcy, bcR=bcR)
