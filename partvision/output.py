"""Annotated image, CSV and JSON writers. They only read what steps put on the Context."""
import csv
import json
import math

import cv2
import numpy as np

from .config import C
from PIL import Image, ImageDraw, ImageFont



def _font(sz, bold=True):
    SS = C["output"]["supersample"]
    for d in ("/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", "C:/Windows/Fonts/", ""):
        for n in (("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf") if bold else
                  ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf")):
            try:
                return ImageFont.truetype(d + n, int(sz * SS))
            except OSError:
                pass
    return ImageFont.load_default(int(sz * SS))


def legend_lines(ctx):
    groups = {}
    for it in sorted(ctx.legend, key=lambda i: i["order"]):
        g = groups.setdefault(it["key"], dict(order=it["order"], rgb=it["rgb"], texts=[]))
        g["texts"].append(it["text"])
    out = []
    for key, g in sorted(groups.items(), key=lambda kv: kv[1]["order"]):
        if key == "orange":
            out.append(("Orange = " + " and ".join(g["texts"]), g["rgb"]))
        else:
            out.append(("  ".join(g["texts"]), g["rgb"]))
    return out


def render(ctx, path, debug=False):
    vx, vy, hw, hh = ctx.view
    img = ctx.img
    H, W = img.shape[:2]
    x0c, y0c = int(round(vx - hw)), int(round(vy - hh))
    x1c, y1c = int(round(vx + hw)), int(round(vy + hh))
    pad = max(0, -x0c, -y0c, x1c - W, y1c - H)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
    crop = src[y0c + pad:y1c + pad, x0c + pad:x1c + pad]
    OUT_W, SS = C["output"]["image_width_px"], C["output"]["supersample"]
    k = OUT_W / crop.shape[1] * SS
    base = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    legend = legend_lines(ctx)
    header = int((24 + 26 * len(legend)) * SS)
    canvas = np.full((base.shape[0] + header, base.shape[1], 3), 26, np.uint8)
    canvas[header:] = base
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)

    def P(x, y):
        return ((x - x0c) * k, (y - y0c) * k + header)

    for L in sorted(ctx.layers, key=lambda L: L["z"]):
        if L["debug"] and not debug:
            continue
        kind = L["kind"]
        if kind == "circle":
            px, py = P(L["x"], L["y"])
            r = L["r"] * k
            dr.ellipse([px - r, py - r, px + r, py + r], outline=L["color"], width=max(1, int(L.get("width", 1) * SS)))
        elif kind in ("polyline", "line"):
            pts = [P(x, y) for x, y in L["points"]]
            if L.get("closed"):
                pts = pts + pts[:1]
            dr.line(pts, fill=L["color"], width=max(1, int(L.get("width", 1) * SS)))
        elif kind == "text":
            px, py = P(L["x"], L["y"])
            dr.text((px, py + L.get("dy", 0) * SS), L["text"], fill=L["color"], font=_font(L.get("size", 13)),
                    anchor="mm", stroke_width=int(1.5 * SS), stroke_fill=(15, 15, 15))
    for i, (s, c) in enumerate(legend):
        dr.text((14 * SS, (14 + i * 26) * SS), s, fill=c, font=_font(14, bold=False))
    pil = pil.resize((pil.width // SS, pil.height // SS), Image.LANCZOS)
    if path:
        pil.save(path)
    return pil


def _mm_header(h):
    return h[:-3] + "_mm" if h.endswith("_px") else h + "_mm"


def write_csv(ctx, path):
    mm = ctx.mm
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for t in ctx.tables + ([dict(title="features", header=["feature", "size_px", "angle_start_deg",
                                                                "angle_end_deg", "span_deg", "note"],
                                     rows=ctx.features, length_cols=["size_px"])] if ctx.features else []):
            w.writerow(["# " + t["title"]])
            idx = [t["header"].index(c) for c in t["length_cols"] if c in t["header"]] if mm else []
            w.writerow(t["header"] + [_mm_header(t["header"][i]) for i in idx])
            for row in t["rows"]:
                extra = []
                for i in idx:
                    try:
                        extra.append(f"{float(row[i]) * mm:.3f}")
                    except (TypeError, ValueError):
                        extra.append("")
                w.writerow(list(row) + extra)
            w.writerow([])


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if math.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def write_json(ctx, path):
    out = dict(ctx.data, warnings=ctx.warnings, notes=ctx.notes)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=_json_default)


STATUS_RGB = {"OK": (40, 200, 40), "DETECTED": (110, 160, 0), "DEVIATES": (230, 50, 50), "MISSING": (220, 0, 200), "CAN'T VERIFY": (240, 150, 20),
              "UNCERTAIN": (240, 150, 20), "NOT MEASURED": (150, 150, 150)}


def _wrap(dr, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if dr.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_panel(ctx, path, title, preview=None):
    """One image per input photo: the annotated photo on the left; on the right the drawing's features,
    a per-feature status table, the alignment details and the warnings."""
    left = render(ctx, None)
    PW = int(C["output"]["panel_width_px"])
    # _font() sizes are for the 2x supersampled canvas; halve them for this 1x panel
    ss = C["output"]["supersample"]
    f_h, f_b, f_s = _font(18 / ss), _font(13 / ss), _font(12 / ss, bold=False)
    tbl = next((t for t in ctx.tables if t["title"].startswith("drawing check")), None)
    rows = tbl["rows"] if tbl else []
    rings = [r for r in rows if r[1] in ("ring", "visual")]
    rows = [r for r in rows if r[1] not in ("ring", "visual")]
    est_h = 60 + (int(PW * 0.62) + 20 if preview is not None else 0) + 24 * (len(rows) + 3) + 30 * 12 + 200
    H = max(left.height, est_h)
    panel = Image.new("RGB", (PW, H), (245, 245, 242))
    dr = ImageDraw.Draw(panel)
    y = 14
    dr.text((16, y), title, fill=(20, 20, 20), font=f_h)
    y += 30
    d = ctx.data.get("drawing", {})
    counts = d.get("status_counts", {})
    x = 16
    det = d.get("detected")
    if det:
        dr.text((16, y), f"Detected {det['found']} of {det['checked']} features", fill=(20, 20, 20), font=f_b)
        y += 22
    for k in ("OK", "DETECTED", "DEVIATES", "MISSING", "CAN'T VERIFY", "UNCERTAIN", "NOT MEASURED"):
        if counts.get(k):
            s = f"{counts[k]} {k}"
            dr.text((x, y), s, fill=STATUS_RGB[k], font=f_b)
            x += int(dr.textlength(s, font=f_b)) + 18
    y += 26
    if preview is not None:
        dr.text((16, y), "Drawing features checked (red = prominent, grey = minor, blue = ring; #n = prominence rank)",
                fill=(90, 90, 90), font=f_s)
        y += 18
        pw = PW - 32
        ph = int(preview.height * pw / preview.width)
        if ph > int(PW * 0.62):
            ph = int(PW * 0.62)
            pw = int(preview.width * ph / preview.height)
        panel.paste(preview.resize((pw, ph), Image.LANCZOS), (16 + (PW - 32 - pw) // 2, y))
        y += ph + 14
    # table
    cols = [("feature", 16), ("kind", 74), ("status", 214), ("pos dev", 302), ("size dev", 376), ("+-tol", 456),
            ("dimension", 498)]
    for name, cx in cols:
        dr.text((cx, y), name, fill=(90, 90, 90), font=f_b)
    y += 20
    dr.line([(16, y - 3), (PW - 16, y - 3)], fill=(180, 180, 180))
    for r in rows:
        kind = (r[1].replace("opening ", "").replace("rectangular", "rect").replace("irregular", "irreg")
                .replace("pin/boss? (round edge)", "pin/boss?"))
        size_dev = r[10] if r[10] and not r[10].startswith("rms") else (r[10] or "")
        dim = r[14] if len(r) > 14 else ""
        vals = [r[0], kind[:20], r[12], (r[9] + " mm") if r[9] else "-", (size_dev + (" mm" if size_dev and
                not size_dev.startswith("rms") else "")) if size_dev else "-", str(r[11]).split("/")[0],
                {"CAN'T VERIFY": "can't verify"}.get(dim, dim.lower().replace("can't verify", "?"))]
        for (name, cx), v in zip(cols, vals):
            col = STATUS_RGB.get(r[12], (30, 30, 30)) if name == "status" else \
                STATUS_RGB.get(dim, (150, 150, 150) if "n/a" in dim else (30, 30, 30)) if name == "dimension" \
                else (30, 30, 30)
            dr.text((cx, y), str(v), fill=col, font=f_s)
        y += 19
    if rings:
        dr.text((16, y), f"+ {len(rings)} concentric ring(s) (counterbore/chamfer/boss): listed, not measured",
                fill=(120, 120, 120), font=f_s)
        y += 19
    y += 10
    reg = d.get("registration", {})
    info = []
    if reg:
        info.append(f"Alignment: {reg.get('px_per_mm', 0):.3f} px/mm, rotated {reg.get('angle_deg', 0):.1f} deg"
                    + (", MIRRORED (other face)" if reg.get("mirror") else ", as drawn")
                    + f"; {reg.get('hole_agreement', 0):.0%} of holes on target, outline overlap "
                      f"{reg.get('coarse_iou', 0):.0%}.")
        info.append(f"Uncertainty: hole size +-{reg.get('uncertainty_hole_size_mm', 0):.2f} mm, hole position "
                    f"+-{reg.get('uncertainty_hole_position_mm', 0):.2f} mm, edges +-{reg.get('uncertainty_edge_mm', 0):.2f} mm."
                    " Dimension check uses a guard band: can't verify = within the uncertainty of the limit.")
        dt = reg.get("detection_tolerance", {})
        if dt:
            info.append(f"Status: OK = found and within +-tol; DETECTED = found within the detection tolerance "
                        f"(position {dt.get('position_mm')}, size {dt.get('size_mm')} mm) but outside or not "
                        "verifiable against +-tol; DEVIATES = outside the detection tolerance; MISSING = not found.")
    for t in info:
        for line in _wrap(dr, t, f_s, PW - 32):
            dr.text((16, y), line, fill=(40, 40, 40), font=f_s)
            y += 17
        y += 4
    for title_, items, col in (("Warnings", ctx.warnings, (200, 60, 30)), ("Notes", ctx.notes, (90, 90, 90))):
        if not items:
            continue
        y += 6
        dr.text((16, y), title_, fill=col, font=f_b)
        y += 19
        for m in items:
            for i, line in enumerate(_wrap(dr, m, f_s, PW - 44)):
                dr.text((16 if i == 0 else 28, y), ("- " if i == 0 else "") + line, fill=col, font=f_s)
                y += 16
            y += 3
    H = max(left.height, y + 16)
    out = Image.new("RGB", (left.width + PW, H), (26, 26, 26))
    out.paste(left, (0, 0))
    out.paste(panel.crop((0, 0, PW, H)), (left.width, 0))
    out.save(path)
    return out
