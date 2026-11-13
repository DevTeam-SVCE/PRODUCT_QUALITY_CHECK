"""Per-part metadata kept next to each drawing, and the calibrated size check that uses it.

A new part needs only its drawing in the drawings folder (DXF or JSON spec). Optionally, facts the drawing
cannot carry go in a sidecar with the same stem:

    dxf_files/flange.dxf
    dxf_files/flange.part.json        {"height_mm": 30}

or, for a JSON spec, the same key inside the spec itself. height_mm is the height of the face the camera sees
(the face the holes are measured on) above the surface the part stands on, i.e. above the reference frame.

Why it matters: a camera capture (capture.py) carries an independent scale from the A4 reference frame. A face
h above the frame appears larger by Z/(Z-h) (Z: camera distance). With the height known, the part's absolute
size is checked against the drawing; without it the size can only be bounded, because a raised face only ever
looks larger, never smaller. Either way a drawing whose size is physically impossible for the photographed part
is rejected by the matcher, which keeps matching reliable as more (similar-looking) parts are added.
"""
import json
import os

from .config import C

META_SUFFIX = ".part.json"


def is_meta_file(path):
    return path.lower().endswith(META_SUFFIX)


def part_meta(drawing_path):
    """Metadata of a drawing: its NAME.part.json sidecar, plus keys of a JSON spec. {} if none."""
    out = {}
    if not drawing_path:
        return out
    stem = os.path.splitext(drawing_path)[0]
    if drawing_path.lower().endswith(".json") and not is_meta_file(drawing_path):
        try:
            with open(drawing_path) as f:
                spec = json.load(f)
            if "height_mm" in spec:
                out["height_mm"] = float(spec["height_mm"])
        except (OSError, ValueError):
            pass
    side = stem + META_SUFFIX
    if os.path.isfile(side):
        with open(side) as f:
            out.update(json.load(f))
        out["_file"] = side
    return out


def size_check(calib, fitted_mm_per_px, drawing_path, part_size_mm, tol_mm):
    """Compare the drawing-fitted scale with the reference frame's scale.

    Returns a JSON-safe dict:
      ratio            apparent size / drawing size on the frame's plane (> 1: looks larger)
      height_mm        the measured face's height used (drawing metadata, else the capture's --part-height-mm)
      status           OK / DEVIATES (absolute size, needs the height) or UNVERIFIED (height unknown)
      plausible        False when no physically possible height explains the size: the wrong drawing
    """
    m = C["matching"]
    plane = float((calib.get("pixel_to_mm") or {}).get("mm_per_px_x") or calib["mm_per_px"])
    ratio = plane / float(fitted_mm_per_px)
    Z = (calib.get("quality") or {}).get("camera_distance_mm")
    meta = part_meta(drawing_path)
    h = meta.get("height_mm")
    src = "drawing metadata" if h is not None else None
    if h is None and (calib.get("part_plane") or {}).get("height_mm"):
        h, src = float(calib["part_plane"]["height_mm"]), "capture (--part-height-mm)"
    out = dict(ratio=ratio, apparent_size_pct=100 * (ratio - 1), camera_distance_mm=Z, part_size_mm=part_size_mm,
               tol_mm=tol_mm, height_mm=h, height_source=src,
               distance_estimate=(calib.get("camera") or {}).get("model"))
    if h is not None and Z and Z > h:
        expected = Z / (Z - h)
        resid = ratio / expected - 1
        dev = resid * part_size_mm
        out.update(expected_ratio=expected, deviation_pct=100 * resid, deviation_over_part_mm=dev,
                   status="OK" if abs(dev) <= tol_mm else "DEVIATES",
                   plausible=abs(100 * resid) <= float(m["calib_size_tol_pct"]))
    else:
        implied = Z * (1 - 1 / ratio) if Z else None
        hmax = float(m["calib_max_part_height_mm"])
        upper = 100 * (Z / (Z - hmax) - 1) if Z and Z > hmax else 100.0
        out.update(implied_height_mm=implied, deviation_pct=100 * (ratio - 1), status="UNVERIFIED",
                   plausible=-float(m["calib_max_smaller_pct"]) <= 100 * (ratio - 1) <=
                   upper + float(m["calib_size_tol_pct"]))
    return out
