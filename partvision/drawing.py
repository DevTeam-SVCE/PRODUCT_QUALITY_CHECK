"""Read a drawing (DXF, or a JSON spec typed from a blueprint) into expected features, in mm.

Coordinates are converted to screen convention (y down) so a top view looks the same as a photo of
that face. Annotation is ignored: dimensions, text, hatches, centre/hidden lines, title-block frames.

Feature model (all mm, y down):
  outline  : dict(pts=(N,2) polygon, area, round=bool)
  features : list of dict(id, kind, rank, prominent, tol, pos_tol, ...)
             kind 'hole'    -> cx, cy, d
             kind 'opening' -> pts polygon, shape (round/slot/rectangular/irregular), length, width
             kind 'ring'    -> cx, cy, d  (concentric with a hole: counterbore/chamfer; not measured)
             kind 'visual'  -> cx, cy, d  (listed only; from a spec with "check": "visual")
"""
import json
import math

import cv2
import numpy as np

from .config import C

UNIT_MM = {0: None, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0, 8: 0.0254, 9: 0.0254 * 1e-3, 13: 1e-3, 14: 0.1}


def _shape_of(pts):
    c = pts.astype(np.float32).reshape(-1, 1, 2)
    area = abs(cv2.contourArea(c))
    perim = cv2.arcLength(c, True)
    circ = 4 * math.pi * area / max(perim ** 2, 1e-12)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    L, Wd = max(w, h), min(w, h)
    solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1e-12)
    if circ >= 0.95:
        shape = "round"
    elif L / max(Wd, 1e-9) >= 1.6 and solidity >= 0.9:
        shape = "slot"
    elif area / max(w * h, 1e-12) >= 0.95:
        shape = "rectangular"
    else:
        shape = "irregular"
    return shape, area, L, Wd


def _centroid(pts):
    M = cv2.moments(pts.astype(np.float32).reshape(-1, 1, 2))
    return (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] else tuple(pts.mean(0))


def _inside(poly, x, y):
    return cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) > 0


def _chain(segs, tol):
    """Join open polylines end-to-end into closed loops. Returns loops, leftover open chains."""
    segs = [np.asarray(s, float) for s in segs if len(s) >= 2]
    used = [False] * len(segs)
    loops, open_ = [], []
    for i, s in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        chain = s.copy()
        if np.linalg.norm(chain[0] - chain[-1]) <= tol and len(chain) > 2:
            loops.append(chain[:-1])
            continue
        grew = True
        while grew:
            grew = False
            for j, t in enumerate(segs):
                if used[j]:
                    continue
                if np.linalg.norm(t[0] - chain[-1]) <= tol:
                    chain = np.vstack([chain, t[1:]])
                elif np.linalg.norm(t[-1] - chain[-1]) <= tol:
                    chain = np.vstack([chain, t[::-1][1:]])
                elif np.linalg.norm(t[-1] - chain[0]) <= tol:
                    chain = np.vstack([t[:-1], chain])
                elif np.linalg.norm(t[0] - chain[0]) <= tol:
                    chain = np.vstack([t[::-1][:-1], chain])
                else:
                    continue
                used[j] = grew = True
                if np.linalg.norm(chain[0] - chain[-1]) <= tol:
                    break
            if np.linalg.norm(chain[0] - chain[-1]) <= tol and len(chain) > 3:
                loops.append(chain[:-1])
                chain = None
                break
        if chain is not None:
            open_.append(chain)
    return loops, open_


def load_dxf(path, notes):
    import ezdxf
    from ezdxf import path as ezpath

    doc = ezdxf.readfile(path)
    units = int(doc.header.get("$INSUNITS", 0))
    k = UNIT_MM.get(units) or float(C["drawing"]["default_units_mm"])
    if units == 0:
        notes.append(f"The DXF has no unit setting ($INSUNITS = 0); assuming {k} mm per unit "
                     "(config drawing.default_units_mm).")

    def ltype(e):
        lt = (e.dxf.get("linetype", "BYLAYER") or "BYLAYER").upper()
        if lt in ("BYLAYER", "BYBLOCK"):
            try:
                lt = doc.layers.get(e.dxf.layer).dxf.linetype.upper()
            except Exception:
                lt = "CONTINUOUS"
        return lt

    def walk(entities, depth=0):
        for e in entities:
            if e.dxftype() == "INSERT" and depth < 8:
                yield from walk(e.virtual_entities(), depth + 1)
            else:
                yield e

    circles, segs, skipped = [], [], {}
    for e in walk(doc.modelspace()):
        t = e.dxftype()
        layer = (e.dxf.get("layer", "0") or "0").upper()
        why = None
        if t in C["drawing"]["skip_entity_types"]:
            why = t.lower()
        elif any(w in layer for w in C["drawing"]["skip_layer_words"]):
            why = f"layer {layer}"
        elif any(w in ltype(e) for w in C["drawing"]["skip_linetype_words"]):
            why = f"linetype {ltype(e)}"
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        if t == "CIRCLE":
            c = e.dxf.center
            circles.append((c.x * k, -c.y * k, e.dxf.radius * k, layer))
            continue
        try:
            p = ezpath.make_path(e)
        except Exception:
            skipped[f"unsupported {t}"] = skipped.get(f"unsupported {t}", 0) + 1
            continue
        pts = np.array([(v.x * k, -v.y * k) for v in p.flattening(distance=0.01)], float)
        if len(pts) >= 2:
            segs.append(pts)
    return circles, segs, skipped, dict(units=units, mm_per_unit=k)


def _rect_like_frame(loop, others, circles):
    """An axis-aligned rectangle enclosing another big loop that holds all circles: a drawing border."""
    shape, area, L, Wd = _shape_of(loop)
    if shape != "rectangular":
        return False
    for o in others:
        if o is loop:
            continue
        a = abs(cv2.contourArea(o.astype(np.float32).reshape(-1, 1, 2)))
        if a > 0.02 * area and all(_inside(loop, *p) for p in o[::max(1, len(o) // 20)]):
            return True
    return False


def build(circles, loops, open_, notes, default_tol, default_pos_tol):
    """Pick the outline (the closed loop holding the most circles) and classify everything inside it."""
    cands = [("loop", l) for l in loops]
    for (x, y, r, _) in circles:
        th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        cands.append(("circle", np.c_[x + r * np.cos(th), y + r * np.sin(th)]))
    if not cands:
        raise SystemExit("No closed outline found in the drawing (lines do not join into a loop).")

    def score(c):
        poly = c[1]
        n_in = sum(1 for (x, y, r, _) in circles if _inside(poly, x, y) and 2 * r < 0.98 * np.ptp(poly, 0).max())
        return (n_in, abs(cv2.contourArea(poly.astype(np.float32).reshape(-1, 1, 2))))

    frames = [l for l in loops if _rect_like_frame(l, loops, circles)]
    if frames:
        notes.append(f"{len(frames)} rectangular frame(s) around the drawing were treated as borders and ignored.")
        loops = [l for l in loops if not any(l is f for f in frames)]
        cands = [c for c in cands if not (c[0] == "loop" and any(c[1] is f for f in frames))]
    kind, outline = max(cands, key=score)
    oarea = abs(cv2.contourArea(outline.astype(np.float32).reshape(-1, 1, 2)))
    osize = float(np.ptp(outline, 0).max())
    n_views = sum(1 for l in loops if l is not outline and not _inside(outline, *_centroid(l))
                  and abs(cv2.contourArea(l.astype(np.float32).reshape(-1, 1, 2))) > 0.1 * oarea)
    if n_views:
        notes.append(f"{n_views} other closed shape(s) outside the chosen outline (other views?) were ignored.")

    feats = []
    inner_circles = [c for c in circles if _inside(outline, c[0], c[1]) and 2 * c[2] < 0.98 * osize]
    # concentric groups: smallest = hole, larger = ring (counterbore / chamfer / boss edge)
    groups = []
    for c in sorted(inner_circles, key=lambda c: c[2]):
        for g in groups:
            if math.hypot(c[0] - g[0][0], c[1] - g[0][1]) <= 1e-3 * osize + 1e-3:
                g.append(c)
                break
        else:
            groups.append([c])
    for g in groups:
        x, y, r, layer = g[0]
        feats.append(dict(kind="hole", cx=x, cy=y, d=2 * r, size=2 * r, layer=layer))
        for (_, _, r2, l2) in g[1:]:
            feats.append(dict(kind="ring", cx=x, cy=y, d=2 * r2, size=2 * r2, layer=l2, of_hole=len(feats)))
    for l in loops:
        if l is outline:
            continue
        cx, cy = _centroid(l)
        if not all(_inside(outline, *p) for p in l[::max(1, len(l) // 40)]):
            continue  # must lie completely inside the part outline
        shape, area, L, Wd = _shape_of(l)
        if shape == "round":  # a hole drawn as arcs
            r = math.sqrt(area / math.pi)
            feats.append(dict(kind="hole", cx=cx, cy=cy, d=2 * r, size=2 * r, layer="loop"))
        else:
            feats.append(dict(kind="opening", pts=l, cx=cx, cy=cy, shape=shape, length=L, width=Wd,
                              size=2 * math.sqrt(area / math.pi), area=area))
    if open_:
        notes.append(f"{len(open_)} open line chain(s) did not close into shapes (edges, bend lines or steps "
                     "seen in this view); they are not checked.")
    return _finish(outline, kind == "circle", feats, default_tol, default_pos_tol)


def _finish(outline, is_round, feats, default_tol, default_pos_tol):
    oarea = abs(cv2.contourArea(outline.astype(np.float32).reshape(-1, 1, 2)))
    osize = float(np.ptp(outline, 0).max())
    # prominence: bigger features first; 'prominent' = at least 3% of the part size
    order = sorted([f for f in feats if f["kind"] in ("hole", "opening")], key=lambda f: -f["size"])
    counters = {"hole": 0, "opening": 0, "ring": 0, "visual": 0}
    for rank, f in enumerate(order, 2):
        f["rank"] = rank
    for f in feats:
        f.setdefault("rank", None)
        f["prominent"] = bool(f["kind"] in ("hole", "opening") and f["size"] >= C["drawing"]["prominent_fraction"] * osize)
        f.setdefault("tol", default_tol)
        f.setdefault("pos_tol", default_pos_tol)
    for f in sorted(feats, key=lambda f: (f["rank"] or 999)):
        counters[f["kind"]] += 1
        f.setdefault("id", {"hole": "H", "opening": "W", "ring": "RING", "visual": "V"}[f["kind"]] + str(counters[f["kind"]]))
    return dict(outline=dict(pts=np.asarray(outline, float), area=oarea, size=osize, round=is_round, rank=1,
                             tol=default_tol), features=sorted(feats, key=lambda f: (f["rank"] or 999)))


def load_spec(path, default_tol, default_pos_tol, notes):
    """JSON spec (mm, y down like the drawing as you read it on paper):
    {"outline": {"type": "rect", "length": 90, "width": 56}   or {"type": "circle", "d": 110}
                 or {"type": "polygon", "points": [[x, y], ...]},
     "tolerance": 0.2, "position_tolerance": 0.2,
     "holes": [{"id": "H1", "x": 10, "y": 10, "d": 5, "tol": 0.1, "pos_tol": 0.15}],
     "openings": [{"id": "W1", "type": "rect", "x": 40, "y": 28, "length": 25, "width": 25, "angle": 0},
                  {"id": "S1", "type": "slot", "x": 20, "y": 40, "length": 12, "width": 4, "angle": 90},
                  {"id": "P1", "type": "polygon", "points": [[...], ...]}],
     "visual": [{"id": "pin", "x": 80, "y": 10, "d": 6}]}"""
    s = json.load(open(path))
    tol = float(s.get("tolerance", default_tol))
    ptol = float(s.get("position_tolerance", s.get("tolerance", default_pos_tol)))
    o = s["outline"]
    if o["type"] == "rect":
        L, Wd = float(o["length"]), float(o["width"])
        outline = np.array([[0, 0], [L, 0], [L, Wd], [0, Wd]], float)
    elif o["type"] == "circle":
        r = float(o["d"]) / 2
        th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        outline = np.c_[r + r * np.cos(th), r + r * np.sin(th)]
    else:
        outline = np.array(o["points"], float)
    feats = []
    for h in s.get("holes", []):
        feats.append(dict(kind="hole", id=h.get("id"), cx=float(h["x"]), cy=float(h["y"]), d=float(h["d"]),
                          size=float(h["d"]), tol=float(h.get("tol", tol)), pos_tol=float(h.get("pos_tol", ptol))))
        if h.get("id") is None:
            feats[-1].pop("id")
    for w in s.get("openings", []):
        if w["type"] == "polygon":
            pts = np.array(w["points"], float)
        else:
            L, Wd, a = float(w["length"]), float(w["width"]), math.radians(float(w.get("angle", 0)))
            if w["type"] == "slot":
                r = Wd / 2
                th1 = np.linspace(-np.pi / 2, np.pi / 2, 30)
                half = np.r_[np.c_[(L / 2 - r) + r * np.cos(th1), r * np.sin(th1)],
                             np.c_[-(L / 2 - r) - r * np.cos(th1), -r * np.sin(th1)]]
                pts = half
            else:
                pts = np.array([[-L / 2, -Wd / 2], [L / 2, -Wd / 2], [L / 2, Wd / 2], [-L / 2, Wd / 2]], float)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            pts = pts @ R.T + [float(w["x"]), float(w["y"])]
        shape, area, L_, W_ = _shape_of(pts)
        f = dict(kind="opening", pts=pts, cx=_centroid(pts)[0], cy=_centroid(pts)[1], shape=shape, length=L_,
                 width=W_, size=2 * math.sqrt(area / math.pi), area=area, tol=float(w.get("tol", tol)),
                 pos_tol=float(w.get("pos_tol", ptol)))
        if w.get("id"):
            f["id"] = w["id"]
        feats.append(f)
    for v in s.get("visual", []):
        feats.append(dict(kind="visual", id=v.get("id", "V"), cx=float(v["x"]), cy=float(v["y"]),
                          d=float(v.get("d", 0)), size=float(v.get("d", 0))))
    return _finish(outline, o["type"] == "circle", feats, tol, ptol)


def build_raster(circles, segs, notes, default_tol, default_pos_tol):
    """Topology from a fine raster of the line work: robust to T-junctions, tiny gaps and holes drawn as arcs.
    Outline = outer boundary of the view holding the most holes. Inner features = regions enclosed entirely
    inside it (regions that touch the outline are just the body split by step/edge lines)."""
    pts_all = [s for s in segs] + [np.array([[x - r, y - r], [x + r, y + r]]) for x, y, r, _ in circles]
    if not pts_all:
        raise SystemExit("The drawing has no geometry after removing annotation.")
    allp = np.vstack(pts_all)
    lo, hi = allp.min(0), allp.max(0)
    ext = float((hi - lo).max())
    k = min(float(C["drawing"]["raster_max_px_per_mm"]), float(C["drawing"]["raster_max_px"]) / max(ext, 1e-9))
    pad = 10
    Wc, Hc = int((hi[0] - lo[0]) * k) + 2 * pad, int((hi[1] - lo[1]) * k) + 2 * pad

    def to_px(p):
        return np.round((np.asarray(p) - lo) * k * 16 + pad * 16).astype(np.int32)

    def to_mm(c):
        return (np.asarray(c, float) - pad) / k + lo

    lines = np.zeros((Hc, Wc), np.uint8)
    for sg in segs:
        cv2.polylines(lines, [to_px(sg).reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8, shift=4)
    for x, y, r, _ in circles:
        cv2.circle(lines, tuple(to_px([x, y])), int(round(r * k * 16)), 255, 1, cv2.LINE_8, shift=4)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # seal sub-0.1 mm gaps
    n_frames = 0
    while True:
        filled = lines.copy()
        ff = np.zeros((Hc + 2, Wc + 2), np.uint8)
        cv2.floodFill(filled, ff, (0, 0), 128)
        solid = (filled != 128).astype(np.uint8)  # everything enclosed by line work, plus the lines
        comps, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        comps = [c for c in comps if cv2.contourArea(c) > 25]
        if not comps:
            raise SystemExit("No closed outline found in the drawing.")
        circ_px = [to_px([x, y]) / 16 for x, y, r, _ in circles]

        def n_inside(c):
            return sum(cv2.pointPolygonTest(c, (float(q[0]), float(q[1])), False) > 0 for q in circ_px)

        best = max(comps, key=lambda c: (n_inside(c), cv2.contourArea(c)))
        # a rectangular border around everything: erase it and look again
        ap = cv2.approxPolyDP(best, 0.01 * cv2.arcLength(best, True), True)
        x_, y_, w_, h_ = cv2.boundingRect(best)
        if len(ap) == 4 and cv2.contourArea(best) > 0.97 * w_ * h_ and n_frames < 3:
            inner = cv2.erode(solid, np.ones((5, 5), np.uint8))
            free = ((inner > 0) & (lines == 0)).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
            if n > 1 and st[1:, 4].max() > 0.5 * w_ * h_:  # mostly empty inside: a frame, not a plate
                cv2.drawContours(lines, [best], -1, 0, 5)
                n_frames += 1
                continue
        break
    if n_frames:
        notes.append(f"{n_frames} rectangular border frame(s) around the drawing were ignored.")
    others = [c for c in comps if c is not best and cv2.contourArea(c) > 0.05 * cv2.contourArea(best)]
    if others:
        notes.append(f"{len(others)} other view(s)/shape(s) outside the chosen outline were ignored.")
    outline = to_mm(best[:, 0, :])
    comp_mask = np.zeros_like(lines)
    cv2.drawContours(comp_mask, [best], -1, 1, -1)
    band = comp_mask - cv2.erode(comp_mask, np.ones((7, 7), np.uint8))
    free = ((comp_mask > 0) & (lines == 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    faces, n_body = [], 0
    for i in range(1, n):
        if st[i, 4] < 6:
            continue
        x_, y_, w_, h_ = st[i, :4]
        f = (lab[y_:y_ + h_, x_:x_ + w_] == i).astype(np.uint8)
        fd = cv2.dilate(np.pad(f, 2), np.ones((5, 5), np.uint8))
        if (fd & np.pad(band[y_:y_ + h_, x_:x_ + w_], 2)).any():
            n_body += 1
            continue
        cc, _ = cv2.findContours(cv2.dilate(np.pad(f, 2), np.ones((3, 3), np.uint8)), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        c = max(cc, key=cv2.contourArea)[:, 0, :] + [x_ - 2, y_ - 2]
        faces.append(to_mm(c))
    if n_body > 1:
        notes.append(f"The part body is split by internal edge/step lines into {n_body} regions; those edges are "
                     "not checked.")
    # round loops -> circles; merge with CIRCLE entities
    rounds, contours = [], []
    from .core import fit_circle
    for fpts in faces:
        # traced loops have staircase edges, so judge roundness by circle-fit residual, not circularity
        cx, cy, r, std, used, _ = fit_circle(fpts, iters=3, floor=1.5 / k)
        if used >= 0.95 * len(fpts) and std <= max(1.0 / k, 0.01 * r):
            rounds.append((cx, cy, r, "arcs"))
        else:
            contours.append(fpts)
    oc = outline.astype(np.float32).reshape(-1, 1, 2)
    allc = [c for c in circles if cv2.pointPolygonTest(oc, (float(c[0]), float(c[1])), False) > 0
            and 2 * c[2] < 0.98 * ext]
    for r_ in rounds:
        if not any(math.hypot(r_[0] - c[0], r_[1] - c[1]) < 0.1 and abs(r_[2] - c[2]) < 0.15 for c in allc):
            allc.append(r_)
    feats = []
    groups = []
    for c in sorted(allc, key=lambda c: c[2]):
        for g in groups:
            if math.hypot(c[0] - g[0][0], c[1] - g[0][1]) <= max(0.05, 1e-3 * ext):
                g.append(c)
                break
        else:
            groups.append([c])
    for g in groups:
        x, y, r, layer = g[0]
        feats.append(dict(kind="hole", cx=x, cy=y, d=2 * r, size=2 * r, layer=layer))
        for (_, _, r2, l2) in g[1:]:
            if r2 - g[0][2] > 0.05:
                feats.append(dict(kind="ring", cx=x, cy=y, d=2 * r2, size=2 * r2, layer=l2))
    for l in contours:
        cx, cy = _centroid(l)
        shape, area, L, Wd = _shape_of(l)
        feats.append(dict(kind="opening", pts=l, cx=cx, cy=cy, shape=shape, length=L, width=Wd,
                          size=2 * math.sqrt(area / math.pi), area=area))
    return _finish(outline, False, feats, default_tol, default_pos_tol)


def load(path, default_tol=0.2, default_pos_tol=None):
    """-> (model, notes, info)"""
    notes = []
    ptol = default_pos_tol if default_pos_tol is not None else default_tol
    if path.lower().endswith(".json"):
        return load_spec(path, default_tol, ptol, notes), notes, dict(source="json spec")
    circles, segs, skipped, info = load_dxf(path, notes)
    if skipped:
        notes.append("Ignored in the drawing: " + ", ".join(f"{v} x {k}" for k, v in sorted(skipped.items())))
    model = build_raster(circles, segs, notes, default_tol, ptol)
    info.update(source="dxf", skipped=skipped)
    return model, notes, info


def render_preview(model, path, title="", caption=True):
    """The drawing with the features that will be checked, ranked by prominence."""
    from PIL import Image, ImageDraw, ImageFont
    pts = model["outline"]["pts"]
    lo, hi = pts.min(0), pts.max(0)
    size = 1000
    pad = 70
    k = (size - 2 * pad) / max(np.ptp(pts, 0).max(), 1e-9)
    Wd = int(np.ptp(pts[:, 0]) * k + 2 * pad)
    top = 90 if caption else 10
    Hd = int(np.ptp(pts[:, 1]) * k + 2 * pad) + top
    im = Image.new("RGB", (Wd, Hd), (250, 250, 247))
    dr = ImageDraw.Draw(im)

    def P(x, y):
        return ((x - lo[0]) * k + pad, (y - lo[1]) * k + pad + top)

    def font(sz):
        for n in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"):
            try:
                return ImageFont.truetype(n, sz)
            except OSError:
                pass
        return ImageFont.load_default()

    dr.line([P(*p) for p in pts] + [P(*pts[0])], fill=(20, 20, 20), width=3)
    MAJ, MIN, RING = (220, 40, 40), (120, 120, 120), (60, 120, 220)
    for f in model["features"]:
        col = RING if f["kind"] == "ring" else (MAJ if f["prominent"] else MIN)
        if f["kind"] in ("hole", "ring", "visual"):
            x, y = P(f["cx"], f["cy"])
            r = f["d"] / 2 * k
            dr.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3 if f["kind"] != "ring" else 1)
        else:
            dr.line([P(*p) for p in f["pts"]] + [P(*f["pts"][0])], fill=col, width=3)
        if f["kind"] != "ring":
            x, y = P(f["cx"], f["cy"])
            lab = f"{f['id']}" + (f" #{f['rank']}" if f["rank"] else "")
            dr.text((x, y - f["size"] / 2 * k - 12), lab, fill=col, font=font(15), anchor="mm")
    n_major = sum(f["prominent"] for f in model["features"])
    lines = [f"{title}  outline {np.ptp(pts[:, 0]):.2f} x {np.ptp(pts[:, 1]):.2f} mm (rank #1)",
             f"red = prominent features ({n_major}), grey = minor, blue = concentric rings (counterbore/chamfer, "
             "not measured); #n = prominence rank"]
    for i, s in enumerate(lines if caption else []):
        dr.text((12, 14 + i * 26), s, fill=(30, 30, 30), font=font(15))
    if path:
        im.save(path)
    return im
