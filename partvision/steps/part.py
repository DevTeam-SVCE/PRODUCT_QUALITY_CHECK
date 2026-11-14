"""STEP 1: separate the part from the background."""
import cv2
import numpy as np

from ..config import C
from ..registry import step
from .. import segment as seg


@step("part")
def part(ctx):
    """Separate the part from the background (method: config segmentation.method, or --segmentation)."""
    W, H, gray = ctx.W, ctx.H, ctx.gray
    pc, used, tried = seg.segment(gray, ctx.opt_get("segmentation"))
    if pc is None:
        raise SystemExit("No part found: the image has no separable foreground.")
    for m, why in tried:
        ctx.note(f"Segmentation '{m}' not used: {why}; used '{used}'.")
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
                    mask_material=mask_material, otsu=float(pc["threshold"]), method=used)
    ctx.data["part"] = dict(segmentation=used, polarity=pc["pol"], otsu_threshold=float(pc["threshold"]), mec_x=pc["x"], mec_y=pc["y"],
                            mec_r=pc["r"], mec_d=2 * pc["r"], fill_of_mec=pc["fill"], touches_frame=pc["touches"],
                            area_px2=pc["area"], centroid=[cx, cy], through_openings=len(pc["inner"]))
    if pc["touches"]:
        ctx.warn("The part outline touches the image border: the part may be cropped or the background not separated.")
    if pc["fill"] < C["image"]["low_fill_warn"]:
        ctx.warn(f"The part outline fills only {pc['fill']:.0%} of its enclosing circle: very thin part or "
                 "segmentation error.")
    size = 2 * pc["r"]
    if size < C["image"]["low_resolution_px"]:
        ctx.warn(f"The part is only {size:.0f} px across: low resolution, expect +-1 px errors to matter.")
    if size < C["image"]["small_in_frame_warn"] * min(W, H):
        ctx.warn("The part fills a small part of the frame; a closer photo would be more accurate.")
    ctx.set_detection_scale(pc["r"])
    ctx.origin = (cx, cy, "part centroid")
    bx, by, bw, bh = pc["bbox"]
    m = C["image"]["view_margin_frac"] * max(bw, bh)
    ctx.view = (bx + bw / 2, by + bh / 2, bw / 2 + m, bh / 2 + m)
    ctx.say(f"image {W}x{H}  part ({used}): {pc['pol']}, enclosing circle d={size:.1f} px, "
            f"{len(pc['inner'])} opening(s) showing background")
    return dict(ok=True)
