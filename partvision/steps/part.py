"""STEP 1: separate the part from the background."""
import math

import cv2
import numpy as np

from ..core import odd
from ..config import C
from ..registry import step


@step("part")
def part(ctx):
    """Otsu in both polarities; keep the largest outline that is not cut by the frame."""
    W, H, gray = ctx.W, ctx.H, ctx.gray
    kb = odd(max(7, min(W, H) / 135))
    blur = cv2.GaussianBlur(gray, (kb, kb), 0)
    otsu, th_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cands = []
    for pol, mask in (("dark part on light background", th_dark), ("light part on dark background", 255 - th_dark)):
        cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        hier = hier[0]
        top = [i for i in range(len(cs)) if hier[i][3] == -1]

        def touches_border(j):
            x_, y_, w_, h_ = cv2.boundingRect(cs[j])
            return x_ <= 1 or y_ <= 1 or x_ + w_ >= W - 1 or y_ + h_ >= H - 1

        for i in top:
            area = cv2.contourArea(cs[i])
            if area < 0.002 * W * H or area > 0.6 * W * H:
                continue  # specks, or the background itself
            c = cs[i]
            (x, y), r = cv2.minEnclosingCircle(c)
            inner = [cs[j] for j in range(len(cs)) if hier[j][3] == i and cv2.contourArea(cs[j]) > 2e-4 * area]
            cands.append(dict(pol=pol, contour=c, inner=inner, x=x, y=y, r=r, area=area, touches=touches_border(i),
                              fill=area / (math.pi * r ** 2 + 1e-9), bbox=cv2.boundingRect(c)))
    if not cands:
        raise SystemExit("No part found: the image has no separable foreground.")
    # Among regions at least a quarter the size of the largest, prefer one that stays inside the frame:
    # a table edge or shadow band touching the border must not beat the part just by being bigger,
    # but a tiny blob must not beat a part that happens to touch the border.
    biggest = max(c["area"] for c in cands)
    cands = [c for c in cands if c["area"] >= 0.25 * biggest]
    cands.sort(key=lambda c: (c["touches"], -c["area"]))
    pc = cands[0]
    c = pc["contour"]
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    mask_outer = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask_outer, [c], -1, 255, -1)
    mask_material = mask_outer.copy()
    if pc["inner"]:
        cv2.drawContours(mask_material, pc["inner"], -1, 0, -1)
    ctx.part = dict(contour=c, inner=pc["inner"], polarity=pc["pol"], x=pc["x"], y=pc["y"], r=pc["r"],
                    area=pc["area"], cx=cx, cy=cy, bbox=pc["bbox"], mask_outer=mask_outer,
                    mask_material=mask_material, otsu=float(otsu))
    ctx.data["part"] = dict(polarity=pc["pol"], otsu_threshold=float(otsu), mec_x=pc["x"], mec_y=pc["y"],
                            mec_r=pc["r"], mec_d=2 * pc["r"], fill_of_mec=pc["fill"], touches_frame=pc["touches"],
                            area_px2=pc["area"], centroid=[cx, cy], through_openings=len(pc["inner"]))
    if pc["touches"]:
        ctx.warn("The part outline touches the image border: the part may be cropped or the background not separated.")
    if pc["fill"] < 0.3:
        ctx.warn(f"The part outline fills only {pc['fill']:.0%} of its enclosing circle: very thin part or "
                 "segmentation error.")
    size = 2 * pc["r"]
    if size < C["image"]["low_resolution_px"]:
        ctx.warn(f"The part is only {size:.0f} px across: low resolution, expect +-1 px errors to matter.")
    if size < 0.3 * min(W, H):
        ctx.warn("The part fills a small part of the frame; a closer photo would be more accurate.")
    ctx.set_detection_scale(pc["r"])
    ctx.origin = (cx, cy, "part centroid")
    bx, by, bw, bh = pc["bbox"]
    m = 0.2 * max(bw, bh)
    ctx.view = (bx + bw / 2, by + bh / 2, bw / 2 + m, bh / 2 + m)
    ctx.say(f"image {W}x{H}  part: {pc['pol']}, enclosing circle d={size:.1f} px, "
            f"{len(pc['inner'])} opening(s) showing background")
    return dict(ok=True)
