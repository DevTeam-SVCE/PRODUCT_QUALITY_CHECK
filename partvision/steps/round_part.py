"""Round parts: central bore, outer rim + notches, rings/steps."""
import math

import cv2
import numpy as np

from ..core import fit_circle, odd, runs_of
from ..config import C
from ..registry import step

MAG, OR = (255, 0, 230), (255, 150, 30)


@step("bore", needs=("part",))
def bore(ctx):
    """Central bore: Hough near the part centre (normalised scale), then an edge fit at full resolution."""
    p = ctx.part
    ox, oy, oR, sN, NR = p["x"], p["y"], p["r"], ctx.sN, ctx.NORM_R
    g9 = cv2.GaussianBlur(ctx.gN, (9, 9), 0)
    trials, hb = {}, None
    for param1 in (100, 60, 40):  # lower Canny thresholds only if nothing is found
        for p2 in (20, 30, 40, 50):
            c = cv2.HoughCircles(g9, cv2.HOUGH_GRADIENT, 1.2, 0.1 * NR, param1=param1, param2=p2,
                                 minRadius=int(0.10 * NR), maxRadius=int(0.40 * NR))
            pick = None
            if c is not None:
                near = [v / sN for v in c[0] if math.hypot(v[0] / sN - ox, v[1] / sN - oy) < 0.2 * oR]
                if near:
                    pick = [round(float(v), 1) for v in near[0]]
            trials[f"param1={param1},param2={p2}"] = pick
            hb = hb or pick
        if hb:
            break
    ctx.data["bore_hough_trials"] = trials
    if not hb:
        ctx.origin = (ox, oy, "enclosing-circle centre")
        ctx.data["bore"] = None
        ctx.warn("No central bore found: angles and radii are measured about the part's enclosing-circle centre.")
        ctx.say("bore: NOT FOUND")
        return dict(bore=None)
    b = ctx.ray_edge_fit(hb[0], hb[1], hb[2] * 0.8, hb[2] * 1.2, n=360)
    ctx.origin = (b["x"], b["y"], "bore centre")
    ctx.data["bore"] = {**b, "hough": hb}
    if b["rays"] < 0.6 * 360 or b["std"] > 0.01 * oR:
        ctx.warn(f"Bore edge fit is weak ({b['rays']}/360 rays, scatter {b['std']:.2f} px).")
    ctx.dims["bore"] = b["d"]
    ctx.feature("bore", f"{b['d']:.2f}", note="edge fit")
    ctx.draw("circle", z=1, x=b["x"], y=b["y"], r=b["r"], color=MAG, width=3)
    ctx.draw("text", z=6, x=b["x"], y=b["y"], text="BORE", color=MAG, size=15)
    ctx.add_legend("info", "Magenta = central bore.", MAG, order=90)
    ctx.say(f"bore d={b['d']:.2f} px at ({b['x']:.1f},{b['y']:.1f})")
    return dict(bore=b)


@step("round_rim", needs=("part",))
def round_rim(ctx):
    """Outer rim of a round part, judged against the local background on each ray (robust to lit edges)."""
    p = ctx.part
    bx, by = ctx.origin[:2]
    ox, oy, oR, gray = p["x"], p["y"], p["r"], ctx.gray
    angles = np.arange(0, 360, 0.5)
    rs_out = np.arange(1.12 * oR, 0.55 * oR, -0.5, dtype=np.float32)
    outside = rs_out >= 1.06 * oR
    band = (rs_out >= 0.90 * oR) & (rs_out <= 0.96 * oR)
    ann = np.zeros(gray.shape, np.uint8)
    cv2.circle(ann, (int(ox), int(oy)), int(0.97 * oR), 255, -1)
    cv2.circle(ann, (int(ox), int(oy)), int(0.85 * oR), 0, -1)
    part_level = float(np.median(gray[(ann > 0) & (p["mask_outer"] > 0)]))
    bg_level = ctx.disc_stat(ox, oy, 1.06 * oR, 1.12 * oR)
    rim_pol = 1.0 if part_level > bg_level else -1.0
    ctx.data["levels"] = dict(part=part_level, background=bg_level)
    if abs(part_level - bg_level) < C["image"]["min_part_contrast"]:
        ctx.warn(f"Low part/background contrast ({abs(part_level - bg_level):.0f} gray levels): rim edge uncertain.")
    run_len = max(3, int(round(0.01 * oR / 0.5)))
    ks = odd(max(5, 0.02 * oR / 0.5))
    rim_R = np.full(len(angles), np.nan)
    for i, a in enumerate(angles):
        prof = cv2.GaussianBlur(ctx.ray_profile(bx, by, math.radians(a), rs_out).reshape(-1, 1), (1, ks), 0).ravel()
        bg = np.median(prof[outside])
        # rim surface level: the background-side quartile, so holes in the band do not drag it down
        pl = np.percentile(prof[band], 75 if rim_pol < 0 else 25)
        if abs(pl - bg) < 0.3 * abs(part_level - bg_level):
            pl = part_level  # no part at the usual rim radius on this ray (notch or gap)
        inside = (prof - (bg + pl) / 2) * rim_pol > 0
        run = np.convolve(inside.astype(int), np.ones(run_len, int), "valid") == run_len
        hit = np.where(run)[0]
        if len(hit):
            rim_R[i] = rs_out[hit[0]]
    ok = ~np.isnan(rim_R)
    pts = np.c_[bx + rim_R[ok] * np.cos(np.radians(angles[ok])), by + rim_R[ok] * np.sin(np.radians(angles[ok]))]
    rx, ry, rR, rstd, rused, _ = fit_circle(pts, iters=6, floor=0.006 * oR)
    ctx.data["outer_rim_fit"] = dict(x=rx, y=ry, r=rR, d=2 * rR, std=rstd, rays=rused, of=len(angles),
                                     offset_from_origin=float(math.hypot(rx - bx, ry - by)))
    if rused < 0.7 * len(angles):
        ctx.warn(f"Only {rused}/{len(angles)} rim rays agree with the fitted outer circle: outer diameter uncertain.")
    pa = np.degrees(np.arctan2(pts[:, 1] - ry, pts[:, 0] - rx)) % 360
    pr = np.hypot(pts[:, 0] - rx, pts[:, 1] - ry)
    order = np.argsort(pa)
    dev = np.interp(angles, pa[order], pr[order] - rR, period=360)
    tol = 0.015 * rR
    notches = []
    for a0, a1 in runs_of(dev < -tol):
        span = (a1 - a0) * 0.5
        idx = np.arange(a0, a1) % len(angles)
        if span >= 2.0:
            notches.append(dict(start=a0 * 0.5 % 360, end=a1 * 0.5 % 360, span=span, mid=((a0 + a1) * 0.25) % 360,
                                max_depth=float(-dev[idx].min())))
    ctx.data["notch_tolerance_px"] = tol
    ctx.dims["outer"] = 2 * rR
    ctx.view = (bx, by, 1.42 * rR, 1.33 * rR)
    ctx.feature("outer rim", f"{2 * rR:.2f}", note=f"rim fit; enclosing circle {2 * oR:.2f}")
    ctx.say(f"outer rim fit d={2 * rR:.1f} px ({rused}/{len(angles)} rays), enclosing circle d={2 * oR:.1f} px")
    for a, r in zip(angles[ok][::2], rim_R[ok][::2]):
        x, y = ctx.polar(r, a)
        ctx.draw("circle", z=9, debug=True, x=x, y=y, r=0.6, color=(255, 0, 0), width=1)
    return dict(x=rx, y=ry, r=rR, notches=notches, angles=angles, dev=dev, tol=tol)


@step("rim_notches", needs=("round_rim",))
def rim_notches(ctx):
    """Notches on the rim; a dip one hole wide centred on a hole is lost contrast, not a notch."""
    rim = ctx.results["round_rim"]
    bc = ctx.results.get("bolt_circle", {})
    real, dropouts = [], []
    if bc.get("ok"):
        hole_w = math.degrees(2 * bc["r_med0"] / bc["dom"])
        for n in rim["notches"]:
            near = [h for h in bc["holes"] if min(abs(h["ang"] - n["mid"]), 360 - abs(h["ang"] - n["mid"])) < 0.6 * hole_w]
            if near and n["span"] <= 1.5 * hole_w:
                dropouts.append(dict(n, at_hole=near[0]["n"]))
            else:
                real.append(dict(n))
    else:
        real = [dict(n) for n in rim["notches"]]
    notches = sorted(real, key=lambda n: -n["span"])
    for i, n in enumerate(notches, 1):
        n["name"] = "N%d" % i
    ctx.data["notches"] = notches
    ctx.data["rim_dropouts_at_holes"] = dropouts
    if dropouts:
        ctx.note("Rim edge lost next to hole(s) %s (too little contrast between hole and rim edge); those rays are "
                 "rejected by the rim fit and not counted as notches." % ", ".join(str(d["at_hole"]) for d in dropouts))
    rx, ry, rR = rim["x"], rim["y"], rim["r"]
    for n in notches:
        ctx.feature("notch " + n["name"], "", f"{n['start']:.1f}", f"{n['end']:.1f}", f"{n['span']:.1f}",
                    f"max depth {n['max_depth']:.1f} px")
        ctx.say(f"notch {n['name']}: {n['start']:.1f}-{n['end']:.1f} deg, span {n['span']:.1f}, "
                f"depth {n['max_depth']:.1f} px")
        arc = [ctx.polar(rR, a, rx, ry) for a in np.arange(n["start"], n["start"] + n["span"] + 0.25, 0.25)]
        ctx.draw("polyline", z=2, points=arc, color=OR, width=3)
        for a in (n["start"], n["start"] + n["span"]):
            ctx.draw("line", z=2, points=[ctx.polar(rR * 0.952, a, rx, ry), ctx.polar(rR * 1.048, a, rx, ry)],
                     color=OR, width=2)
        lx, ly = ctx.polar(rR * 0.89, n["mid"], rx, ry)
        ctx.draw("text", z=6, x=lx, y=ly, text="NOTCH" if len(notches) == 1 else "NOTCH " + n["name"], color=OR, size=14)
        ctx.draw("text", z=6, x=lx, y=ly, text=f"{n['span']:.0f}° rim", color=OR, size=12, dy=19)
    if notches:
        ctx.add_legend("orange", "notch rim span (measured)", OR, order=20)
    return dict(notches=notches)


@step("rings", needs=("round_rim",))
def rings(ctx):
    """Concentric steps/rings from the median radial profile; strong ones are confirmed by a circle fit."""
    p = ctx.part
    bx, by = ctx.origin[:2]
    oR, rR = p["r"], ctx.results["round_rim"]["r"]
    angles = np.arange(0, 360, 0.5)
    notches = ctx.results.get("rim_notches", {}).get("notches", ctx.results["round_rim"]["notches"])
    bc = ctx.results.get("bolt_circle", {})
    b = ctx.results.get("bore", {}).get("bore")
    prof_r = np.arange(0.12 * oR, 1.03 * rR, 0.5, dtype=np.float32)
    in_notch = np.zeros(len(angles), bool)
    for n in notches:
        a0 = int((n["start"] - 3) * 2)
        for k in range(a0, a0 + int((n["span"] + 6) * 2)):
            in_notch[k % len(angles)] = True
    med = np.median([ctx.ray_profile(bx, by, math.radians(a), prof_r) for a in angles[~in_notch][::2]], axis=0)
    gr = np.gradient(cv2.GaussianBlur(med.reshape(-1, 1), (1, odd(max(5, 0.028 * oR))), 0).ravel())
    win = max(4, int(0.03 * oR / 0.5))
    excl = prof_r > rR - 0.05 * oR
    if bc.get("ok"):
        excl |= np.abs(prof_r - bc["dom"]) < 1.6 * bc["r_med"]
    if b:
        excl |= np.abs(prof_r - b["r"]) < 0.06 * oR
    peaks = [i for i in range(win, len(gr) - win)
             if not excl[i] and abs(gr[i]) == np.abs(gr[i - win:i + win + 1]).max() and abs(gr[i]) > 0.6]
    gmax = max([abs(gr[i]) for i in peaks], default=0)
    out = []
    for i in peaks:
        R0 = float(prof_r[i])
        e = dict(R=R0, d=2 * R0, gradient=round(float(gr[i]), 2),
                 direction="darker outward" if gr[i] < 0 else "lighter outward")
        if abs(gr[i]) >= 0.3 * gmax:
            f = ctx.ray_edge_fit(bx, by, R0 - 0.04 * oR, R0 + 0.04 * oR, n=360, floor=0.004 * oR)
            good = f["std"] <= 0.006 * oR and f["rays"] >= 0.6 * 360 and abs(f["r"] - R0) < 0.03 * oR
            e.update(fit_x=f["x"], fit_y=f["y"], fit_d=f["d"], fit_std=f["std"], rays=f["rays"],
                     status="confirmed ring" if good else "edge found but not a clean circle")
            ctx.draw("circle", z=9, debug=True, x=f["x"], y=f["y"], r=f["r"],
                     color=(0, 255, 255) if good else (255, 0, 0), width=1)
        else:
            e["status"] = "weak edge, unconfirmed"
        out.append(e)
        dd = e.get("fit_d", e["d"])
        if e["status"] == "confirmed ring":
            ctx.dims["ring_%.0f" % dd] = dd
        ctx.feature("ring/step", f"{dd:.2f}", note=f"{e['status']}, {e['direction']}")
        ctx.say(f"ring/step d={dd:.1f} px  {e['status']}  ({e['direction']}, grad {e['gradient']})")
    ctx.data["rings"] = out
    return dict(rings=out)
