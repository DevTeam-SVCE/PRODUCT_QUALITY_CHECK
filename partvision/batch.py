"""Batch inspection: every photo in a folder, matched automatically to the right drawing.

For each photo:
  1. Quick screen: compare the part's silhouette with every drawing's outline (Hu-moment shape distance:
     independent of position, rotation, scale and mirroring). Keep the best few.
  2. Full check against each remaining candidate (the normal drawing_match pipeline).
  3. Pick the candidate with the best match score (outline overlap, holes on target, hole coverage,
     features detected; weights in config matching.weights). Below matching.min_match_score the photo
     is reported as having no matching drawing.
  4. Keep the winner's image and data; also run the drawing-free analysis (the one 'inspect' pipeline, told
     whether the part is round from the matched drawing's outline).

Nothing here knows any file names: drop a new DXF/JSON into the drawings folder and new photos into the
photos folder, and they are picked up. A pairs file can force a pairing when needed.
"""
import argparse
import contextlib
import glob
import io
import json
import math
import os
import re
import shutil
import sys

import cv2
import numpy as np

from . import config
from . import drawing as dwg
from .cli import main as run_one
from .config import C


def _stem(path):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(path))[0]).strip("_")


def _find(folders, exts, exclude_dirs=()):
    out = []
    for d in folders:
        if os.path.isfile(d):
            out.append(d)
            continue
        for f in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts and \
                    not any(os.path.abspath(f).startswith(os.path.abspath(x) + os.sep) for x in exclude_dirs):
                out.append(f)
    return out


def _silhouette(photo):
    """The part's outline, by the same segmentation as the 'part' step (partvision/segment.py)."""
    from . import segment
    g = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, 0
    cand, _, _ = segment.segment(g)
    return (cand["contour"], cand["area"]) if cand else (None, 0)


def _outline_contour(model):
    pts = model["outline"]["pts"]
    k = 1000.0 / max(float(np.ptp(pts, 0).max()), 1e-9)
    return ((pts - pts.min(0)) * k).astype(np.int32).reshape(-1, 1, 2)


def _is_round(model):
    pts = model["outline"]["pts"].astype(np.float32).reshape(-1, 1, 2)
    a, p = abs(cv2.contourArea(pts)), cv2.arcLength(pts, True)
    hull = cv2.contourArea(cv2.convexHull(pts))
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    # round = near-circular hull and square bounding box (a notch does not change either much)
    k = C["matching"]["round_drawing"]
    return 4 * math.pi * hull / max(cv2.arcLength(cv2.convexHull(pts), True) ** 2, 1e-9) > k["min_hull_circularity"] \
        and min(w, h) / max(w, h) > k["min_aspect"]


def _quiet(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_one(argv)
    return rc, buf.getvalue()


_PHOTO_HOLES = {}


def photo_holes(photo):
    """Holes the photo shows on its part (independent of any drawing), cached per photo. The method is config
    matching.hole_count.method:
      features - the analysis pipeline's own hole finder (step 'holes', run as for a part that is not round):
                 only features it measured as clean circles, smaller than matching.big_hole_frac x the part's
                 longest side (the same rule that separates holes from bores in the drawing)
      blobs    - round, enclosed dark spots on the silhouette (the earlier, quicker screen)
    Sizes are relative to the part's area-equivalent radius, which a shadow merged into the silhouette inflates
    much less than the enclosing circle."""
    if photo in _PHOTO_HOLES:
        return _PHOTO_HOLES[photo]
    from .core import Context
    from .registry import run
    hc = C["matching"]["hole_count"]
    if hc["method"] == "features":
        ctx = Context(photo, argparse.Namespace(shape="other"))
        run(ctx, ["part", "outline", "holes"])
        dmax = C["matching"]["big_hole_frac"] * ctx.frame["L"]
        pts = [(h["x"], h["y"]) for h in ctx.results["holes"]["items"]
               if h["shape"] == "round" and h["fit"] and h["d"] < dmax]
        _PHOTO_HOLES[photo] = pts
        return pts
    from .steps import drawing_match as dm
    ctx = Context(photo, argparse.Namespace())
    run(ctx, ["part"])
    req = math.sqrt(ctx.part["area"] / math.pi)
    bx, by, bw, bh = ctx.part["bbox"]
    B = dm._blobs(ctx, hc["min_radius"] * req, hc["max_radius"] * req, (bx, by, bx + bw, by + bh), hc["window"] * req)
    # a hole lies inside the material: its centre at least `edge_clearance` radii from the outline (a notch in
    # the outline or a shadow corner is a dark round-ish spot too, but not enclosed)
    clear = cv2.distanceTransform((ctx.part["mask_outer"] > 0).astype(np.uint8), cv2.DIST_L2, 5)
    pts = [(float(x), float(y)) for x, y, r, _, dark in B
           if clear[int(y), int(x)] >= hc["edge_clearance"] * r
           and dm._surrounded(ctx, x, y, r, dark=bool(dark)) >= hc["surrounded"]]
    _PHOTO_HOLES[photo] = pts
    return pts


def hole_coverage(photo, rep, model):
    """Share of the photo's holes that the aligned drawing explains: explained / max(photo holes, drawing
    holes). A drawing that explains only a subset (e.g. the other face of the part) or none scores low."""
    from .steps import drawing_match as dm
    size = model["outline"]["size"]
    km = C["matching"]
    small = [f for f in model["features"] if f["kind"] == "hole" and f["d"] < km["big_hole_frac"] * size]
    big = [f for f in model["features"] if f["kind"] == "hole" and f["d"] >= km["big_hole_frac"] * size]
    T = rep["drawing"]["registration"]["transform_px_from_mm"]
    if "H" in T:
        inv = dm.Proj(np.array(T["H"]), np.array(dwg._centroid(model["outline"]["pts"]))).inverse
    else:
        A, t = np.array(T["A"]), np.array(T["t"])
        inv = lambda p: (np.asarray(p, float) - t) @ np.linalg.inv(A).T
    openings = [(np.asarray(f["pts"], np.float32).reshape(-1, 1, 2), C["matching"]["opening_margin_frac"] * f["size"])
                for f in model["features"] if f["kind"] == "opening" and f.get("pts") is not None
                and not dm._encloses(f, model)]  # plain openings only; a recess edge encloses real holes
    pts = photo_holes(photo)
    counted, explained = 0, 0
    for x, y in pts:
        mx, my = inv([[x, y]])[0]
        if any(math.hypot(mx - f["cx"], my - f["cy"]) < km["inside_big_hole_diameters"] * f["d"] for f in big):
            continue  # inside the bore or another large hole: not a hole of its own
        if any(cv2.pointPolygonTest(poly, (float(mx), float(my)), True) > -m for poly, m in openings):
            continue  # inside (or on the edge of) an opening: its corners and walls are not holes
        counted += 1
        if any(math.hypot(mx - f["cx"], my - f["cy"]) < max(km["hole_explained_diameters"] * f["d"], km["hole_explained_min_mm"])
               for f in small):
            explained += 1
    if not small and counted == 0:
        return 1.0, counted, explained
    # a pseudo-count keeps one stray spot (scratch, reflection) from zeroing a drawing with few or no holes
    k = float(C["matching"]["coverage_prior"])
    return (explained + k) / (max(counted, len(small), 1) + k), counted, explained


def match_score(rep, coverage_info):
    """Combine the full-check result into one number in [0, 1]."""
    d = rep["drawing"]
    reg = d["registration"]
    w = C["matching"]["weights"]
    n_dwg = sum(1 for r in d["results"] if r.get("kind") == "hole")
    coverage = coverage_info[0]
    det = d.get("detected", {})
    detected = det.get("found", 0) / max(det.get("checked", 1), 1)
    parts = dict(outline_overlap=float(reg.get("coarse_iou", 0)), hole_agreement=float(reg.get("hole_agreement", 0)),
                 hole_coverage=float(coverage), detected=float(detected))
    if n_dwg == 0:  # no holes in the drawing: the evidence is whether the photo has none either
        tw = dict(w, hole_coverage=w["hole_coverage"] + w["hole_agreement"], hole_agreement=0.0)
    else:
        tw = w
    total = sum(tw.values())
    return sum(tw[k] * parts[k] for k in parts) / total, parts


def _size_tiebreak(results):
    """Geometry ranks the drawings; the capture's calibrated size (if any) only breaks near-ties, e.g. two drawings
    of the same shape at different sizes. Results are (score, drawing, name, round, report, parts), best first."""
    tie = float(C["matching"]["size_tiebreak_score"])

    def size_dev(r):
        chk = (r[4].get("calibration") or {}).get("scale_check")
        if chk is None:
            return None
        return (0 if chk["plausible"] else 1, abs(chk["deviation_pct"]))

    near = [r for r in results if results[0][0] - r[0] <= tie]
    if len(near) < 2 or any(size_dev(r) is None for r in near):
        return results
    best = min(near, key=size_dev)
    others = ", ".join(os.path.basename(r[1]) for r in near if r is not best)
    print(f"   size tie-break: {os.path.basename(best[1])} chosen over {others} "
          f"(geometry scores within {tie:g}; its calibrated size fits best)")
    return [best] + [r for r in results if r is not best]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", default=["."], help="folders (or files) with photos (default: .)")
    ap.add_argument("--drawings", nargs="+", default=["dxf_files"],
                    help="folders (or files) with DXF drawings / JSON specs (default: dxf_files)")
    ap.add_argument("--out", default="output", help="output folder (default: output)")
    ap.add_argument("--pairs", default=None, help="optional JSON {photo_file_name: drawing_path} forcing pairings")
    ap.add_argument("--config", default=None, help="JSON file overriding settings in partvision/config.py")
    ap.add_argument("--tol", type=float, default=None, help="tolerance in mm (default: config)")
    ap.add_argument("--keep", action="store_true", help="keep the existing output folder instead of clearing it")
    args = ap.parse_args(argv)
    config.load(args.config)
    tol = args.tol if args.tol is not None else C["tolerance"]["default_mm"]
    out, data = args.out, os.path.join(args.out, "data")
    if not args.keep and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(data, exist_ok=True)
    tmp = os.path.join(out, ".work")
    os.makedirs(tmp, exist_ok=True)

    from .partmeta import is_meta_file
    drawings = [d for d in _find(args.drawings, set(C["matching"]["drawing_extensions"])) if not is_meta_file(d)]
    photos = _find(args.photos, set(C["matching"]["photo_extensions"]), exclude_dirs=[out] + args.drawings)
    if not drawings:
        raise SystemExit(f"no drawings found in {args.drawings}")
    if not photos:
        raise SystemExit(f"no photos found in {args.photos}")
    forced = json.load(open(args.pairs)) if args.pairs else {}

    models = {}
    for d in drawings:
        try:
            m, notes, _ = dwg.load(d, tol)
            models[d] = m
            print(f"drawing {d}: {np.ptp(m['outline']['pts'][:, 0]):.1f} x {np.ptp(m['outline']['pts'][:, 1]):.1f} mm, "
                  f"{sum(f['kind'] == 'hole' for f in m['features'])} holes, "
                  f"{sum(f['kind'] == 'opening' for f in m['features'])} openings")
        except SystemExit as e:
            print(f"drawing {d}: SKIPPED ({e})")
    summary = []
    for photo in photos:
        stem = _stem(photo)
        print(f"\n== {photo}")
        if os.path.basename(photo) in forced:
            cands = [forced[os.path.basename(photo)]]
            print(f"   pairing forced by {args.pairs}: {cands[0]}")
        else:
            sil, _ = _silhouette(photo)
            if sil is None:
                print("   no part found in the photo; skipped")
                summary.append(dict(photo=photo, drawing=None, score=0.0))
                continue
            shape_d = {d: cv2.matchShapes(sil, _outline_contour(m), cv2.CONTOURS_MATCH_I2, 0) for d, m in models.items()}
            ranked = sorted(shape_d, key=shape_d.get)
            if len(ranked) <= int(C["matching"]["check_all_up_to"]):
                cands = ranked  # few drawings: check them all; the screen only orders them
            else:
                cands = ranked[:max(1, int(C["matching"]["full_check_candidates"]))]
            print("   shape screen: " + ", ".join(f"{os.path.basename(d)} {shape_d[d]:.3f}" for d in ranked)
                  + f"  -> full check: {', '.join(os.path.basename(c) for c in cands)}")
        results = []
        for d in cands:
            rnd = _is_round(models[d]) if d in models else False
            name = f"{stem}__{_stem(d)}"
            argv_ = [photo, tmp, "--name", name, "--drawing", d, "--tol", str(tol), "--data-dir", tmp,
                     "--out-image", os.path.join(tmp, name + ".png")]
            if args.config:
                argv_ += ["--config", args.config]
            rc, log = _quiet(argv_)
            try:
                rep = json.load(open(os.path.join(tmp, name + "_report.json")))
            except (OSError, ValueError):
                print(f"   {os.path.basename(d)}: check failed")
                continue
            cov = hole_coverage(photo, rep, models[d]) if d in models else (0.0, 0, 0)
            sc, parts = match_score(rep, cov)
            parts["photo_holes_on_part"], parts["explained_by_drawing"] = cov[1], cov[2]
            size = (rep.get("calibration") or {}).get("scale_check")
            if size is not None:  # calibrated capture: a physically impossible size rules this drawing out
                parts["calibrated_size"] = f"{size['deviation_pct']:+.1f}%"
                if (C["matching"].get("size_gate") == "reject" and not size["plausible"]
                        and not forced.get(os.path.basename(photo))):
                    print(f"   {os.path.basename(d):<24} ruled out: part looks {size['deviation_pct']:+.1f}% vs this "
                          "drawing (A4 frame calibration), more than any part height explains")
                    continue
            results.append((sc, d, name, rnd, rep, parts))
            print(f"   {os.path.basename(d):<24} score {sc:.2f}  " +
                  ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in parts.items() if isinstance(v, float))
                  + (f", calibrated size {parts['calibrated_size']}" if "calibrated_size" in parts else "")
                  + f"  (photo holes on part {cov[1]}, explained {cov[2]})")
        if not results:  # every candidate failed (or was ruled out by size): no matching drawing
            print("   NO MATCHING DRAWING. Writing the drawing-free analysis instead.")
            _quiet([photo, out, "--name", f"{stem}_analysis", "--out-image", os.path.join(out, stem + ".png"),
                    "--data-dir", data] + (["--config", args.config] if args.config else []))
            summary.append(dict(photo=photo, drawing=None, score=0.0, image=os.path.join(out, stem + ".png")))
            continue
        results.sort(key=lambda r: -r[0])
        results = _size_tiebreak(results)
        sc, d, name, rnd, rep, parts = results[0]
        det_frac = parts["detected"]
        if (sc < C["matching"]["min_match_score"] or det_frac < C["matching"]["min_detected_fraction"]) \
                and not forced.get(os.path.basename(photo)):
            print(f"   NO MATCHING DRAWING (best: {os.path.basename(d)}, score {sc:.2f}, {det_frac:.0%} of its features "
                  f"found; need {C['matching']['min_match_score']} and {C['matching']['min_detected_fraction']:.0%}). "
                  "Writing the drawing-free analysis instead.")
            _quiet([photo, out, "--name", f"{stem}_analysis", "--out-image", os.path.join(out, stem + ".png"),
                    "--data-dir", data] + (["--config", args.config] if args.config else []))
            summary.append(dict(photo=photo, drawing=None, best=d, score=round(sc, 3), score_parts=parts,
                                image=os.path.join(out, stem + ".png")))
            continue
        shutil.copy(os.path.join(tmp, name + ".png"), os.path.join(out, stem + ".png"))
        for ext in ("_measurements.csv", "_report.json"):
            shutil.copy(os.path.join(tmp, name + ext), os.path.join(data, f"{stem}_drawing_check{ext}"))
        _quiet([photo, out, "--name", f"{stem}_analysis", "--shape", "round" if rnd else "other", "--no-image",
                "--data-dir", data] + (["--config", args.config] if args.config else []))
        dd = rep["drawing"]
        counts = ", ".join(f"{v} {k}" for k, v in dd["status_counts"].items())
        margin = sc - results[1][0] if len(results) > 1 else None
        print(f"   -> {os.path.basename(d)} (score {sc:.2f}" + (f", next best {margin:+.2f} behind" if margin is not None
              else "") + f"): detected {dd['detected']['found']}/{dd['detected']['checked']}; {counts}")
        summary.append(dict(photo=photo, drawing=d, score=round(sc, 3), score_parts=parts,
                            margin_to_next=None if margin is None else round(margin, 3),
                            detected=dd["detected"], status_counts=dd["status_counts"],
                            image=os.path.join(out, stem + ".png")))
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\n{'photo':<28} {'drawing':<26} score  detected  result")
    for s_ in summary:
        if s_.get("drawing"):
            print(f"{os.path.basename(s_['photo']):<28} {os.path.basename(s_['drawing']):<26} {s_['score']:.2f}   "
                  f"{s_['detected']['found']:>2}/{s_['detected']['checked']:<4}   "
                  + ", ".join(f"{v} {k}" for k, v in s_["status_counts"].items()))
        else:
            print(f"{os.path.basename(s_['photo']):<28} {'(no matching drawing)':<26} {s_['score']:.2f}")
    print(f"\nimages: {out}/<photo>.png   data: {data}/   summary: {out}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
