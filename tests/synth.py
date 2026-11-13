"""Virtual camera for testing the reference-frame capture: an exact pinhole projection of a textured plane
(background + reference sheet + optional part drawn from a DXF/JSON model), with blur, motion, noise, lighting."""
import math

import cv2
import numpy as np

WORLD_PPM = 4.0                # texture resolution, px per mm (above the camera's ~2.8 px/mm)
WORLD_MM = (700.0, 760.0)      # textured area (x, y) in mm
ORIGIN_MM = (245.0, 230.0)     # where the sheet's corner 0 lies inside the texture


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def make_world(sheet_mm=(210.0, 297.0), part=None, part_at=(105.0, 148.5), part_rot=0.0, seed=1,
               sheet_gray=232, bg_gray=150, part_gray=55, occluder=None):
    """Texture in mm coordinates (sheet corner 0 at the origin). part: drawing model (partvision.drawing.load).
    occluder: polygon in sheet mm drawn on top (a hand / object covering part of an edge)."""
    rng = np.random.default_rng(seed)
    W, H = int(WORLD_MM[0] * WORLD_PPM), int(WORLD_MM[1] * WORLD_PPM)
    low = cv2.resize(rng.normal(0, 1, (24, 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = bg_gray + 14 * low + rng.normal(0, 3, (H, W)).astype(np.float32)
    # wood-like streaks so the background is not blank
    img += 6 * np.sin(np.linspace(0, 90, W, dtype=np.float32))[None, :]

    def P(pts_mm):
        p = (np.asarray(pts_mm, np.float64) + ORIGIN_MM) * WORLD_PPM * 16
        return np.round(p).astype(np.int32).reshape(-1, 1, 2)

    sw, sh = sheet_mm
    cv2.fillPoly(img, [P([[0, 0], [sw, 0], [sw, sh], [0, sh]])], float(sheet_gray), cv2.LINE_AA, 4)
    if part is not None:
        c = np.array(part_at)
        R = rot("z", part_rot)[:2, :2]
        cen = part["outline"]["pts"].mean(0)

        def place(pts):
            return (np.asarray(pts, np.float64) - cen) @ R.T + c

        cv2.fillPoly(img, [P(place(part["outline"]["pts"]))], float(part_gray), cv2.LINE_AA, 4)
        for f in part["features"]:
            if f["kind"] not in ("hole", "opening"):
                continue
            if "pts" in f and f["pts"] is not None and len(f["pts"]) >= 3:
                cv2.fillPoly(img, [P(place(f["pts"]))], float(sheet_gray), cv2.LINE_AA, 4)
            else:
                ctr = place([[f["cx"], f["cy"]]])[0]
                cv2.circle(img, tuple(int(v) for v in P([ctr])[0, 0]), int(round(f["d"] / 2 * WORLD_PPM * 16)),
                           float(sheet_gray), -1, cv2.LINE_AA, 4)
    if occluder is not None:
        cv2.fillPoly(img, [P(occluder)], 90.0, cv2.LINE_AA, 4)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


class Camera:
    """Pinhole camera above the sheet. Pose: camera centre at (cx, cy, -Z) in sheet mm (looking along +z),
    then rotated: tilt_x / tilt_y (deg) about its own axes, yaw (deg) about the optical axis."""

    def __init__(self, size=(1920, 1080), fov_deg=75.0):
        self.size = size
        W, H = size
        f = 0.5 * math.hypot(W, H) / math.tan(math.radians(fov_deg) / 2)
        self.K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])

    def H_img_from_mm(self, cx=105.0, cy=148.5, Z=520.0, tilt_x=0.0, tilt_y=0.0, yaw=0.0):
        R = rot("z", yaw) @ rot("x", tilt_x) @ rot("y", tilt_y)
        t = -R @ np.array([cx, cy, -Z])
        return self.K @ np.c_[R[:, 0], R[:, 1], t]

    def far_side(self, sheet_mm=(210.0, 297.0), **pose):
        """Ground truth: the image-side name (top/right/bottom/left) of the sheet side farthest from the camera."""
        Hm = self.H_img_from_mm(**pose)
        sw, sh = sheet_mm
        mm = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], float)
        z = (np.c_[mm, np.ones(4)] @ (np.linalg.inv(self.K) @ Hm).T)[:, 2]
        img = self.corners(sheet_mm, **pose)
        i = int(np.argmax([(z[j] + z[(j + 1) % 4]) for j in range(4)]))
        d = (img[i] + img[(i + 1) % 4]) / 2 - img.mean(0)
        a = math.degrees(math.atan2(d[1], d[0])) % 360
        return {0: "right", 1: "bottom", 2: "left", 3: "top"}[int(((a + 45) % 360) // 90)]

    def corners(self, sheet_mm=(210.0, 297.0), **pose):
        Hm = self.H_img_from_mm(**pose)
        sw, sh = sheet_mm
        p = np.c_[np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], float), np.ones(4)] @ Hm.T
        return p[:, :2] / p[:, 2:]

    def render(self, world, blur=0.0, motion=None, gain=1.0, offset=0.0, noise=1.5, seed=0, **pose):
        S = np.array([[WORLD_PPM, 0, ORIGIN_MM[0] * WORLD_PPM], [0, WORLD_PPM, ORIGIN_MM[1] * WORLD_PPM], [0, 0, 1]])
        Hw = self.H_img_from_mm(**pose) @ np.linalg.inv(S)
        img = cv2.warpPerspective(world, Hw, self.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = cv2.GaussianBlur(img, (0, 0), 0.7 + blur)  # lens
        if motion:
            dx, dy = motion
            n = int(max(abs(dx), abs(dy))) * 2 + 1
            k = np.zeros((n, n), np.float32)
            cv2.line(k, (n // 2 - int(dx), n // 2 - int(dy)), (n // 2 + int(dx), n // 2 + int(dy)), 1.0, 1)
            img = cv2.filter2D(img, -1, k / k.sum())
        img = img * gain + offset + np.random.default_rng(seed).normal(0, noise, img.shape)
        g = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.merge([g, g, g])
