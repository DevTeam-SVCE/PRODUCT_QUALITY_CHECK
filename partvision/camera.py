"""Live acquisition with an A4 reference frame: camera reader, automatic capture logic, saving, and the hand-off
to the existing inspection pipeline.

    camera (thread) --latest frame--> CaptureController.process (processing thread) --capture job--> worker:
        finalize_capture: lens correction, full-resolution corner refinement, re-check, homography,
                          rectified image + calibration sidecar  -->  run_inspection (batch.py / cli.py)

Only the newest camera frame is processed (older ones are dropped), so the preview and the capture decision
never lag behind the camera. Detection runs at `processing_px`; the expensive full-resolution steps run only
when the cheap gates pass, and once more on the captured frame.
"""
import collections
import contextlib
import datetime as _dt
import json
import logging
import math
import os
import threading
import time

import cv2
import numpy as np

from . import refframe as rf

log = logging.getLogger("partvision.camera")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

MSG = dict(
    search="Position the reference frame inside the camera view.",
    partial="Ensure the complete reference frame is visible.",
    closer="Move the camera closer to the reference frame.",
    occluded="Keep the edges of the reference frame uncovered.",
    angle="Adjust camera angle.",
    aspect="This rectangle is not A4. Use the 210 x 297 mm reference frame.",
    light="Improve the lighting.",
    steady="Hold steady.",
    blur="Image not sharp. Hold steady or refocus.",
    ready="Alignment verified. Capturing automatically.",
    capturing="Capturing...",
    captured="Captured. Place the next part, or press Retake.",
    detecting="Reference frame found. Hold steady.",
)


# --------------------------------------------------------------------------------------------- camera

def parse_source(src):
    """Camera index ('0'), stream URL (IP-camera app on a phone: http://PHONE:8080/video, rtsp://...),
    video file, or a still image (repeated as a stream; for trying the app without a camera)."""
    if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
        return "device", int(src)
    if isinstance(src, str) and os.path.isfile(src) and src.lower().endswith(IMAGE_EXTS):
        return "image", src
    if isinstance(src, str) and os.path.isfile(src):
        return "video", src
    return "stream", src


class CameraSource(threading.Thread):
    """Reads frames continuously and keeps only the newest. Reconnects after a disconnect."""

    def __init__(self, source, width=1920, height=1080, fps=30):
        super().__init__(daemon=True, name="camera")
        self.kind, self.target = parse_source(source)
        self.desc = str(source)
        self.req = (int(width), int(height), float(fps))
        self.cond = threading.Condition()
        self.frame, self.t, self.seq = None, 0.0, 0
        self.connected, self.error = False, None
        self.size = None
        self.fps = 0.0
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _open(self):
        if self.kind == "image":
            img = cv2.imread(self.target)
            if img is None:
                raise IOError(f"cannot read image {self.target}")
            return img
        cap = cv2.VideoCapture(self.target)
        if not cap.isOpened():
            raise IOError(f"cannot open camera source {self.desc}")
        if self.kind == "device":
            w, h, fps = self.req
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _publish(self, frame):
        now = time.monotonic()
        with self.cond:
            if self.t:
                self.fps = 0.9 * self.fps + 0.1 / max(now - self.t, 1e-3) if self.fps else 1 / max(now - self.t, 1e-3)
            self.frame, self.t, self.seq = frame, now, self.seq + 1
            self.size = (frame.shape[1], frame.shape[0])
            self.connected, self.error = True, None
            self.cond.notify_all()

    def _disconnected(self, err):
        with self.cond:
            if self.connected or self.error != err:
                log.warning("camera: %s", err)
            self.connected, self.error = False, err
            self.cond.notify_all()

    def run(self):
        while not self._stop.is_set():
            try:
                cap = self._open()
            except Exception as e:  # noqa: BLE001
                self._disconnected(str(e))
                self._stop.wait(1.0)
                continue
            if self.kind == "image":
                period = 1.0 / max(self.req[2], 1)
                while not self._stop.is_set():
                    self._publish(cap.copy())
                    self._stop.wait(period)
                return
            fails = 0
            period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.kind == "video" else 0.0
            while not self._stop.is_set():
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.kind == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    fails += 1
                    if fails >= 10:
                        break
                    self._stop.wait(0.02)
                    continue
                fails = 0
                self._publish(frame)
                if period:
                    self._stop.wait(max(0.0, period - (time.monotonic() - t0)))
            cap.release()
            if not self._stop.is_set():
                self._disconnected("camera stopped sending frames; reconnecting")
                self._stop.wait(1.0)

    def wait_frame(self, after_seq, timeout=1.0):
        """Newest frame with seq > after_seq, or None on timeout / while disconnected."""
        with self.cond:
            if self.seq <= after_seq:
                self.cond.wait(timeout)
            if self.seq <= after_seq or not self.connected:
                return None
            return self.seq, self.t, self.frame


class PushSource(CameraSource):
    """The phone's own camera, read by its web browser (no app): the phone page (/phone) posts JPEG frames, which
    push() publishes like a camera frame. Disconnected when no frame arrived for `timeout` seconds."""

    def __init__(self, timeout=2.0):
        super().__init__("phone")
        self.kind, self.desc, self.timeout = "phone", "phone browser camera", timeout
        self.error = "Waiting for the phone: open the /phone page in the phone's browser."

    def push(self, data):
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("not a JPEG/PNG image")
        self._publish(frame)

    def run(self):
        while not self._stop.wait(0.5):
            if self.connected and time.monotonic() - self.t > self.timeout:
                self._disconnected("The phone stopped sending frames: keep the /phone page open and the screen on.")


# ---------------------------------------------------------------------------------------- capture logic

def _gate(ok, msg=None, detail=""):
    return dict(ok=bool(ok), msg=None if ok else msg, detail=detail)


class CaptureController:
    """Per-frame decision: detect/track the frame, evaluate every capture condition, decide to capture.

    Thread use: process() runs on one thread; retake() / request_manual() / capture_done() /
    capture_failed() may be called from others."""

    GROUPS = dict(frame=("detected", "corners", "visible", "size", "aspect"),
                  alignment=("alignment", "perspective"),
                  stability=("lighting", "stability", "sharpness", "calibration"))

    def __init__(self, cfg, intrinsics=None):
        self.cfg = dict(cfg)
        self.ref = tuple(float(v) for v in cfg["reference_mm"])
        self.target_aspect = min(self.ref) / max(self.ref)
        self.intr = intrinsics
        self._lock = threading.Lock()
        self._requests = set()
        self.busy = False            # a capture is being finalised
        self.last_sig = None         # (quad, thumb) of the last capture: the duplicate-capture lock
        self.last_error = None       # (message, until_t)
        self._full = None            # cached float32 full-res gray of the current frame
        self.reset()

    # -- requests from other threads
    def retake(self):
        with self._lock:
            self._requests.add("retake")

    def request_manual(self):
        with self._lock:
            self._requests.add("manual")

    def capture_done(self, sig):
        with self._lock:
            self.busy, self.last_sig = False, sig

    def capture_failed(self, msg, t=None):
        with self._lock:
            self.busy = False
            self.last_error = (msg, (t or time.monotonic()) + 3.0)
        self.reset()

    def reset(self):
        self.track = None
        self.hits = 0
        self.misses = 0
        self.hist = collections.deque()
        self.prev_thumb = None
        self.last_quad = None

    # -- helpers
    def _model(self, W, H):
        return rf.camera_model(self.intr, W, H, self.cfg["assumed_diagonal_fov_deg"])

    def _detect(self, small, sf):
        cfg = self.cfg
        partial = False
        if self.track is not None:
            r = rf.refine_quad(sf, self.track, 8.0, min_inliers=cfg["min_edge_inliers"])
            if r is not None and self._plausible(r["quad"], small.shape):
                return r, False
        cands, partial = rf.find_candidates(small, cfg, self.ref)
        if self.track is not None and cands:  # prefer the candidate we were tracking
            cands.sort(key=lambda c: (np.abs(c["quad"] - self.track).max() > cfg["track_jump_px"], -c["score"]))
        for c in cands[:3]:
            r = rf.refine_quad(sf, c["quad"], 6.0, min_inliers=cfg["min_edge_inliers"])
            r = r and rf.refine_quad(sf, r["quad"], 3.0, min_inliers=cfg["min_edge_inliers"])
            if r is not None and self._plausible(r["quad"], small.shape):
                r["touches"] = c["touches"]
                return r, partial
        return None, partial

    def _plausible(self, q, shape):
        g = rf.quad_geometry(q)
        H, W = shape[:2]
        return g["convex"] and g["area"] >= self.cfg["detect_min_area_frac"] * W * H and \
            abs(g["aspect"] / self.target_aspect - 1) <= self.cfg["detect_aspect_tol"] and \
            np.abs(g["angles"] - 90).max() <= self.cfg["detect_angle_tol_deg"]

    # -- main entry
    def process(self, frame, t):
        """Analyse one frame. Returns (status, job): job is a capture job when this frame is captured."""
        cfg = self.cfg
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = min(1.0, cfg["processing_px"] / max(W, H))
        small = cv2.resize(gray, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
        sf = cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)
        self._full = None
        with self._lock:
            req, self._requests = self._requests, set()
            busy = self.busy
            err = self.last_error if self.last_error and self.last_error[1] > t else None
        if "retake" in req:
            self.last_sig = None
            self.reset()

        r, partial = self._detect(small, sf)
        st = dict(state="search", message=MSG["search"], hint="", gates={}, metrics={}, quad=None, scale=s,
                  frame_wh=[W, H], t=t)
        gates = st["gates"]
        if r is None:
            self.misses += 1
            was_near_border = self.last_quad is not None and self._margin(self.last_quad, small.shape) < \
                cfg["min_margin_frac"] * min(small.shape[:2])
            if self.misses >= 2:
                self.track, self.hits, self.prev_thumb = None, 0, None
                self.hist.clear()
            gates["detected"] = _gate(False, MSG["search"])
            if partial or (was_near_border and self.misses < 15):
                st.update(state="partial", message=MSG["partial"])
            return self._finish(st, err, busy, req), None

        q = r["quad"]
        jump = float(np.abs(q - self.track).max()) if self.track is not None else 0.0
        if self.track is None or jump > cfg["track_jump_px"]:
            self.hits = 0
            self.hist.clear()
            self.prev_thumb = None
        self.hits += 1
        self.misses = 0
        self.track = q
        self.last_quad = q
        qfull = rf.to_full(q, s)
        st["quad"] = qfull.tolist()
        g = rf.quad_geometry(q)
        thumb = rf.sheet_thumb(small, q, g["landscape"])
        motion = rf.thumb_change(thumb, self.prev_thumb)
        self.prev_thumb = thumb
        self.hist.append((t, q, motion))
        span = cfg["stable_ms"] / 1000.0
        while self.hist and t - self.hist[0][0] > 3 * span:
            self.hist.popleft()

        # ---- frame: detected, corners, visible, size, aspect
        confirmed = self.hits >= cfg["confirm_frames"]
        gates["detected"] = _gate(confirmed, MSG["detecting"], "portrait" if not g["landscape"] else "landscape")
        occl = [i for i, o in enumerate(r["occluded"]) if o]
        gates["corners"] = _gate(not occl, MSG["occluded"],
                                 f"edge fit {max(r['rms']) / s:.2f} px" if not occl else f"{len(occl)} edge(s) covered")
        margin = self._margin(q, small.shape)
        visible = margin >= cfg["min_margin_frac"] * min(small.shape[:2]) and not r.get("touches")
        gates["visible"] = _gate(visible, MSG["partial"], f"margin {margin / s:.0f} px")
        K, dist, kind = self._model(W, H)
        qf = rf.undistort_points(qfull, K, dist)
        ppm = rf.native_px_per_mm(qf, self.ref)
        area_frac = g["area"] / (small.shape[0] * small.shape[1])
        gates["size"] = _gate(area_frac >= cfg["min_area_frac"] and ppm.min() >= cfg["min_px_per_mm"], MSG["closer"],
                              f"{area_frac:.0%} of view, {ppm.min():.2f} px/mm")
        al = rf.alignment(qf, K, self.ref, cfg)
        # A wrong aspect with square corners and no keystone is not tilt (PnP with the A4 model would explain it
        # as tilt): a different rectangle. With a visibly tilted view the aspect is judged once aligned.
        frontal = al["ratios"]["corner_angles"] <= 1 and al["ratios"]["keystone"] <= 1
        gates["aspect"] = _gate(al["aspect_ok"] or not frontal, MSG["aspect"],
                                f"aspect {al['aspect']:.4f} (A4 {self.target_aspect:.4f})")
        # ---- alignment
        hint = ""
        if not al["aligned"] and al["far_side"]:
            hint = f"The {al['far_side']} of the frame is farther from the camera ({al['tilt_deg']:.1f}° off)."
        gates["alignment"] = _gate(al["aligned"], MSG["angle"],
                                   f"tilt {al['tilt_deg']:.1f}°, corners ±{al['corner_angle_dev_deg']:.1f}°")
        gates["perspective"] = _gate(al["perspective_ok"], MSG["angle"], f"keystone {al['keystone']:.1%}")
        st["metrics"].update(tilt_deg=al["tilt_deg"], tilt_model=kind, corner_angle_dev_deg=al["corner_angle_dev_deg"],
                             keystone=al["keystone"], aspect=al["aspect"], alignment_score=al["score"],
                             area_frac=area_frac, px_per_mm=float(ppm.max()), mm_per_px=1.0 / float(ppm.max()),
                             distance_mm=al["distance_mm"], orientation="landscape" if g["landscape"] else "portrait")
        # ---- stability (cheap: corners + thumbnails)
        # frames needed: stable_frames at camera rates; a slow source (phone browser) needs at least 3, and the
        # window then reaches back as far as those frames go (so it always spans >= stable_ms)
        dts = np.diff([h[0] for h in self.hist])
        need = min(cfg["stable_frames"], max(3, int(span / float(np.median(dts))) + 1)) if len(dts) else \
            cfg["stable_frames"]
        win = [h for h in self.hist if t - h[0] <= span + 1e-6]
        if len(win) < need:
            win = list(self.hist)[-need:]
        covered = bool(self.hist) and t - self.hist[0][0] >= span - 1e-3
        disp = max((float(np.abs(h[1] - q).max()) for h in win), default=0.0)
        mot = [h[2] for h in win if h[2] is not None]
        mmean = max((m[0] for m in mot), default=0.0)
        mfrac = max((m[1] for m in mot), default=0.0)
        steady = covered and len(win) >= need and disp <= cfg["max_corner_disp_px"] and \
            mmean <= cfg["max_content_motion"] and mfrac <= cfg["max_motion_frac"]
        held = (t - self.hist[0][0]) if self.hist else 0.0
        gates["stability"] = _gate(steady, MSG["steady"],
                                   f"corners {disp:.2f} px, held {min(held, span) * 1000:.0f}/{span * 1000:.0f} ms")
        st["metrics"].update(corner_disp_px=disp, content_motion=mmean, motion_frac=mfrac, stable_frames=len(win))

        # ---- lighting + sharpness: full resolution, only when everything cheaper already passes
        cheap_ok = all(gates[k]["ok"] for k in ("detected", "corners", "visible", "size", "alignment", "perspective"))
        exp = rf.exposure(thumb)
        if cheap_ok:
            uniform = rf.illumination(thumb)[1]
            ew = rf.edge_width(self._gray_f(gray), qfull)
            lim = rf.edge_width_limit(cfg, W, H)
            contrast = ew["contrast"] if ew else 0.0
            light_ok = exp["median"] >= cfg["min_brightness"] and exp["clipped"] <= cfg["max_clipped_frac"] and \
                contrast >= cfg["min_edge_contrast"] and uniform >= cfg["min_uniformity"]
            why = "too dark" if exp["median"] < cfg["min_brightness"] else "glare / overexposed" \
                if exp["clipped"] > cfg["max_clipped_frac"] else "a shadow is falling on the sheet" \
                if uniform < cfg["min_uniformity"] else "low edge contrast" if not light_ok else ""
            gates["lighting"] = _gate(light_ok, MSG["light"], (f"{why}: " if why else "") +
                                      f"brightness {exp['median']:.0f}, edge contrast {contrast:.0f}")
            sharp_ok = ew is not None and ew["width_px"] <= lim
            gates["sharpness"] = _gate(sharp_ok, MSG["blur"],
                                       f"edge width {ew['width_px']:.1f} / {lim:.1f} px" if ew else "no edge profile")
            st["metrics"].update(edge_width_px=ew["width_px"] if ew else None, edge_width_limit_px=lim,
                                 sharpness_score=max(0.0, 1 - ew["width_px"] / lim) if ew else 0.0,
                                 edge_contrast=contrast, brightness=exp["median"], clipped=exp["clipped"],
                                 uniformity=uniform)
            Hm = cv2.getPerspectiveTransform(qf.astype(np.float32), rf.plane_points(g["landscape"], self.ref)[0]
                                             .astype(np.float32))
            gates["calibration"] = _gate(np.isfinite(Hm).all() and np.linalg.cond(Hm) < 1e9, MSG["search"],
                                         f"{1 / ppm.max():.4f} mm/px")
        else:
            for k in ("lighting", "sharpness", "calibration"):
                gates[k] = dict(ok=False, msg=None, detail="pending", pending=True)

        # ---- decision
        failing = [k for k in ("detected", "corners", "visible", "aspect", "alignment", "perspective", "size",
                               "lighting", "stability", "sharpness", "calibration") if not gates[k]["ok"]]
        manual = "manual" in req
        same = self._same_as_last(q, thumb)
        if not failing and not same and not busy:
            st.update(state="ready", message=MSG["ready"])
            return self._finish(st, err, busy, req), self._job(frame, q, s, st, thumb, manual=False, failing=[])
        if manual and not busy and all(gates[k]["ok"] for k in ("corners", "visible")):
            st.update(state="capturing", message=MSG["capturing"])
            return self._finish(st, err, busy, req), self._job(frame, q, s, st, thumb, manual=True, failing=failing)
        if not failing and busy:
            st.update(state="capturing", message=MSG["capturing"])
        elif not failing and same:
            st.update(state="captured", message=MSG["captured"])
        elif failing:
            k = failing[0]
            st.update(state={"detected": "detecting", "visible": "partial", "alignment": "adjust",
                             "perspective": "adjust", "stability": "steady", "sharpness": "steady"}.get(k, "adjust"),
                      message=gates[k]["msg"] or MSG["steady"], hint=hint if k in ("alignment", "perspective") else
                      gates[k]["detail"] if k in ("lighting", "size", "aspect") else "")
            if same and k in ("stability", "sharpness", "detected"):
                st.update(state="captured", message=MSG["captured"], hint="")
        return self._finish(st, err, busy, req), None

    def _gray_f(self, gray):
        if self._full is None:
            self._full = gray.astype(np.float32)
        return self._full

    @staticmethod
    def _margin(q, shape):
        H, W = shape[:2]
        return float(min(q[:, 0].min(), q[:, 1].min(), W - 1 - q[:, 0].max(), H - 1 - q[:, 1].max()))

    def _same_as_last(self, q, thumb):
        """The duplicate-capture lock: the scene still matches the last capture (frame not moved, inside of
        the frame unchanged). Moving the camera/frame or changing the part releases it."""
        if self.last_sig is None:
            return False
        if not self.cfg["auto_rearm"]:
            return True
        lq, lt = self.last_sig
        if lq.shape != q.shape or float(np.abs(lq - q).max()) > self.cfg["rearm_corner_px"]:
            return False
        ch = rf.thumb_change(thumb, lt, level=25.0)
        return ch is not None and ch[1] <= self.cfg["rearm_changed_frac"]

    def _job(self, frame, q, s, st, thumb, manual, failing):
        with self._lock:
            self.busy = True
        return dict(frame=frame, quad_small=q.copy(), scale=s, metrics=dict(st["metrics"]), t=st["t"],
                    sig=(q.copy(), thumb.copy()), manual=manual, failing=failing,
                    stability=dict(corner_disp_px=st["metrics"].get("corner_disp_px"),
                                   frames=st["metrics"].get("stable_frames"), window_ms=self.cfg["stable_ms"]))

    def _finish(self, st, err, busy, req):
        if busy and st.get("quad") is not None:
            st.update(state="capturing", message=MSG["capturing"], hint="")
        if err:
            st["error"] = err[0]
        groups = {}
        for name, keys in self.GROUPS.items():
            gs = [st["gates"][k] for k in keys if k in st["gates"]]
            if not gs or not st["gates"].get("detected", {}).get("ok", False) and name != "frame":
                groups[name] = dict(status="idle", detail="")
            elif all(x["ok"] for x in gs):
                groups[name] = dict(status="ok", detail=" · ".join(x["detail"] for x in gs[:2] if x["detail"]))
            else:
                bad = next(x for x in gs if not x["ok"])
                groups[name] = dict(status="pending" if bad.get("pending") else "fail", detail=bad["detail"])
        st["groups"] = groups
        return st


# ------------------------------------------------------------------------------------------- capture

def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def finalize_capture(job, cfg, intr, save_dir, source_desc=""):
    """Full-resolution processing of the captured frame. Writes, in save_dir/<id>/:
        <id>_original.png            the frame exactly as captured
        <id>_rectified.png           perspective-corrected inside of the frame (input to the pipeline)
        <id>_rectified.calib.json    calibration metadata (read automatically by analyse_part / inspect_all)
        <id>_detection.jpg           the frame with the detected corners
    Raises RuntimeError when the full-resolution re-check fails (the caller returns to live mode)."""
    t0 = time.monotonic()
    ref = tuple(float(v) for v in cfg["reference_mm"])
    frame = job["frame"]
    H, W = frame.shape[:2]
    K, dist, kind = rf.camera_model(intr, W, H, cfg["assumed_diagonal_fov_deg"])
    s = job["scale"]
    q_dist0 = rf.to_full(job["quad_small"], s)
    if dist is not None:
        frame_u = cv2.undistort(frame, K, dist)
        q0 = rf.undistort_points(q_dist0, K, dist)
    else:
        frame_u, q0 = frame, q_dist0
    gray_u = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
    gf = cv2.GaussianBlur(gray_u, (3, 3), 0).astype(np.float32)
    r = rf.refine_quad(gf, q0, 2.0 / s + 3, n=48, min_inliers=cfg["min_edge_inliers"])
    r = r and rf.refine_quad(gf, r["quad"], 3.0, n=64, min_inliers=cfg["min_edge_inliers"])
    if r is None:
        raise RuntimeError("full-resolution edge refinement failed")
    q = r["quad"]
    if np.abs(q - q0).max() > 2.0 / s + 3:
        raise RuntimeError("full-resolution corners disagree with the tracked frame")
    al = rf.alignment(q, K, ref, cfg)
    ew = rf.edge_width(gray_u.astype(np.float32), q)
    lim = rf.edge_width_limit(cfg, W, H)
    if not job["manual"]:
        bad = [n for n, ok in (("alignment", al["aligned"]), ("perspective", al["perspective_ok"]),
                               ("aspect", al["aspect_ok"]), ("sharpness", ew is not None and ew["width_px"] <= lim))
               if not ok]
        if bad:
            raise RuntimeError("full-resolution re-check failed: " + ", ".join(bad))
    rect = rf.rectify(frame_u, q, ref, cfg["rectify_px_per_mm"], cfg["rectify_inset_mm"])
    if rect["corner_check_px"] > 0.01 or rect["cond"] > 1e9:
        raise RuntimeError("homography is degenerate")
    # even out the paper's brightness (a shadow over the sheet otherwise merges with the part in the
    # inspection's threshold); brightness only, geometry untouched. The uncorrected image is kept too.
    raw_rect = rect["image"]
    surf, uniformity = rf.illumination(cv2.cvtColor(raw_rect, cv2.COLOR_BGR2GRAY))
    illum = dict(uniformity_before=uniformity, corrected=bool(cfg.get("flatten_illumination", True)))
    if illum["corrected"]:
        rect["image"] = rf.flatten(raw_rect, surf)
        illum["uniformity_after"] = rf.illumination(cv2.cvtColor(rect["image"], cv2.COLOR_BGR2GRAY))[1]
    ppm = rect["px_per_mm"]
    mm_plane = 1.0 / ppm
    h_part = float(cfg.get("part_height_mm") or 0.0)
    Z = al["distance_mm"]
    part_plane = None
    if h_part > 0 and Z and Z > h_part:
        f = (Z - h_part) / Z
        part_plane = dict(height_mm=h_part, camera_distance_mm=Z, scale_factor=f, mm_per_px=mm_plane * f,
                          distance_source="calibrated intrinsics" if dist is not None or intr else
                          "approximate (assumed field of view)")
    stamp = _dt.datetime.now().astimezone()
    cid = "cap_" + stamp.strftime("%Y%m%d_%H%M%S_") + f"{stamp.microsecond // 1000:03d}"
    d = os.path.join(save_dir, cid)
    os.makedirs(d, exist_ok=True)
    p_orig, p_rect = os.path.join(d, cid + "_original.png"), os.path.join(d, cid + "_rectified.png")
    p_meta, p_det = os.path.join(d, cid + "_rectified.calib.json"), os.path.join(d, cid + "_detection.jpg")
    if illum["corrected"]:
        p_raw = os.path.join(d, cid + "_rectified_uncorrected.png")
        cv2.imwrite(p_raw, raw_rect, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        illum["uncorrected_file"] = os.path.basename(p_raw)
    png = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # lossless; fast level (the capture is on the critical path)
    if not (cv2.imwrite(p_orig, frame, png) and cv2.imwrite(p_rect, rect["image"], png)):
        raise RuntimeError(f"cannot write images to {d}")
    q_orig = rf.distort_points(q, K, dist)
    det = frame.copy()
    cv2.polylines(det, [np.round(q_orig * 4).astype(np.int32)], True, (60, 200, 90), max(2, W // 640), cv2.LINE_AA, 2)
    for p in q_orig:
        cv2.circle(det, (int(round(p[0] * 4)), int(round(p[1] * 4))), 4 * max(6, W // 200), (60, 200, 90), max(2, W // 640),
                   cv2.LINE_AA, 2)
    cv2.imwrite(p_det, det, [cv2.IMWRITE_JPEG_QUALITY, 88])
    obj, (wmm, hmm) = rf.plane_points(al["landscape"], ref)
    native = rf.native_px_per_mm(q, ref)
    inset = float(cfg["rectify_inset_mm"])
    notes = ["Measurements are valid on the plane of the reference frame. A face above that plane appears larger "
             "by Z/(Z-h) (Z: camera distance, h: height); set camera.part_height_mm to correct the scale for the "
             "measured face. The homography does not remove parallax of features at other heights.",
             "Rectified pixels are square: the same mm per px horizontally and vertically."]
    if dist is None:
        notes.append("No lens calibration: lens distortion is not corrected and the tilt/distance estimates use an "
                     "assumed field of view (calibrate_camera.py makes both exact).")
    meta = dict(
        format="partvision.refframe/1",
        capture=dict(id=cid, timestamp=stamp.isoformat(timespec="milliseconds"), source=source_desc,
                     frame_size_wh=[W, H], manual=bool(job["manual"]), failing_checks_at_capture=job["failing"]),
        reference_frame=dict(width_mm=min(ref), height_mm=max(ref),
                             orientation="landscape" if al["landscape"] else "portrait",
                             plane_size_mm=[wmm, hmm], aspect_expected=min(ref) / max(ref),
                             aspect_measured=al["aspect"]),
        corners=dict(order="TL, TR, BR, BL (clockwise on screen)", mm=obj, original_px=q_orig, undistorted_px=q),
        homography=dict(mm_from_image=rect["H_mm_from_img"], rectified_from_image=rect["H_rect_from_img"],
                        usage="[x, y, w] = H @ [u, v, 1] with (u, v) in undistorted image px; result = (x/w, y/w)"),
        camera=dict(model=kind, camera_matrix=K, dist_coeffs=dist, intrinsics_file=intr["path"] if intr else None,
                    intrinsics_rms_px=intr.get("rms") if intr else None,
                    assumed_diagonal_fov_deg=None if intr else cfg["assumed_diagonal_fov_deg"]),
        rectified=dict(file=os.path.basename(p_rect), size_wh=list(rect["size_wh"]), px_per_mm=ppm, inset_mm=inset,
                       origin_mm=[inset, inset], covers_mm=[[inset, inset], [wmm - inset, hmm - inset]]),
        pixel_to_mm=dict(mm_per_px_x=mm_plane, mm_per_px_y=mm_plane, px_per_mm_x=ppm, px_per_mm_y=ppm,
                         native_px_per_mm_in_photo=dict(min=float(native.min()), max=float(native.max())),
                         plane="reference frame plane"),
        part_plane=part_plane,
        mm_per_px=part_plane["mm_per_px"] if part_plane else mm_plane,
        quality=dict(alignment_score=al["score"], tilt_deg=al["tilt_deg"], tilt_estimate=kind,
                     camera_distance_mm=Z, far_side=al["far_side"], corner_angles_deg=al["corner_angles_deg"],
                     corner_angle_max_dev_deg=al["corner_angle_dev_deg"], keystone=al["keystone"],
                     aspect_error=al["aspect_error"], pnp_reprojection_px=al["pnp_reprojection_px"],
                     sharpness_score=max(0.0, 1 - ew["width_px"] / lim) if ew else 0.0,
                     edge_width_px=ew["width_px"] if ew else None, edge_width_per_side_px=ew["per_side_px"] if ew else None,
                     edge_width_limit_px=lim, edge_contrast=ew["contrast"] if ew else None,
                     edge_fit_rms_px=r["rms"], edge_inliers=r["inliers"], stability=job["stability"],
                     thresholds={k: cfg[k] for k in ("max_tilt_deg", "max_corner_angle_dev_deg", "max_keystone",
                                                     "aspect_tol", "max_corner_disp_px", "stable_ms")}),
        illumination=illum,
        timing=dict(finalize_ms=None),
        notes=notes,
    )
    meta["timing"]["finalize_ms"] = round((time.monotonic() - t0) * 1000, 1)
    with open(p_meta, "w") as f:
        json.dump(meta, f, indent=1, default=_jsonable)
    return dict(id=cid, dir=d, original=p_orig, rectified=p_rect, calibration=p_meta, detection=p_det, meta=meta)


# ----------------------------------------------------------------------------------------- inspection

def run_inspection(rectified, out_dir, drawings=(), drawing=None, tol=None, config_path=None):
    """Hand the rectified image to the existing pipeline. Its calibration sidecar supplies --mm-per-px.
    drawing: check against one drawing; drawings: folders/files matched automatically (inspect_all);
    neither: drawing-free analysis. Returns a short JSON-safe summary."""
    from . import batch, cli
    os.makedirs(out_dir, exist_ok=True)
    extra = (["--tol", str(tol)] if tol is not None else []) + (["--config", config_path] if config_path else [])
    stem = os.path.splitext(os.path.basename(rectified))[0]
    logf = os.path.join(out_dir, "pipeline.log")
    t0 = time.monotonic()
    with open(logf, "w") as f, contextlib.redirect_stdout(f):
        try:
            if drawing:
                cli.main([rectified, out_dir, "--drawing", drawing, "--name", stem,
                          "--out-image", os.path.join(out_dir, stem + ".png")] + extra)
                mode = "drawing"
            elif drawings:
                batch.main(["--photos", rectified, "--drawings", *drawings, "--out", out_dir, "--keep"] + extra)
                mode = "batch"
            else:
                cli.main([rectified, out_dir, "--name", stem, "--out-image", os.path.join(out_dir, stem + ".png")]
                         + extra)
                mode = "analysis"
        except SystemExit as e:
            if e.code not in (0, None):
                return dict(ok=False, error=str(e), log=logf)
    res = dict(ok=True, mode=mode, image=os.path.join(out_dir, stem + ".png"), log=logf,
               seconds=round(time.monotonic() - t0, 2))
    if mode == "batch":
        with open(os.path.join(out_dir, "summary.json")) as f:
            s = json.load(f)[0]
        res.update(drawing=os.path.basename(s["drawing"]) if s.get("drawing") else None, score=s.get("score"),
                   status_counts=s.get("status_counts"), detected=s.get("detected"))
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            if s.get("drawing"):
                with open(os.path.join(out_dir, "data", f"{stem}_drawing_check_report.json")) as f:
                    res["size_check"] = json.load(f)["calibration"]["scale_check"]
        res["report"] = os.path.join(out_dir, "data", f"{stem}_drawing_check_report.json") if s.get("drawing") else \
            os.path.join(out_dir, "data", f"{stem}_analysis_report.json")
    else:
        res["report"] = os.path.join(out_dir, f"{stem}_report.json")
        with contextlib.suppress(OSError, ValueError, KeyError):
            with open(res["report"]) as f:
                rep = json.load(f)
            if "drawing" in rep:
                res.update(drawing=os.path.basename(rep["drawing"]["file"]), status_counts=rep["drawing"]["status_counts"])
            res["warnings"] = len(rep.get("warnings", []))
    if not os.path.exists(res["image"]):
        res["image"] = None
    return res


# ----------------------------------------------------------------------------------------- application

PALETTE = dict(neutral=(235, 235, 235), adjust=(40, 170, 255), ok=(110, 205, 70), captured=(110, 205, 70))


def render_preview(frame, st, preview_px):
    """Live preview: the frame, everything outside the reference frame dimmed, its boundary and corner marks.
    All text is in the web UI, not burnt into the video."""
    H, W = frame.shape[:2]
    k = min(1.0, preview_px / max(W, H))
    img = cv2.resize(frame, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
    if st.get("quad") is None:
        return img
    q = np.asarray(st["quad"]) * k
    col = PALETTE["ok"] if st["state"] in ("ready", "capturing", "captured") else \
        PALETTE["neutral"] if st["state"] in ("detecting", "search") else PALETTE["adjust"]
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(q * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
    dim = cv2.convertScaleAbs(img, alpha=0.55)
    img = np.where(mask[..., None] > 127, img, dim)
    sh = 4
    cv2.polylines(img, [np.round(q * (1 << sh)).astype(np.int32)], True, col, 2, cv2.LINE_AA, sh)
    L = max(14.0, 0.06 * min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])))
    for i in range(4):
        p = q[i]
        for nb in (q[i - 1], q[(i + 1) % 4]):
            e = p + (nb - p) / np.linalg.norm(nb - p) * L
            cv2.line(img, tuple(int(v) for v in np.round(p * (1 << sh))), tuple(int(v) for v in np.round(e * (1 << sh))),
                     col, 5, cv2.LINE_AA, sh)
    return img


def _placeholder(text, wh=(960, 540)):
    img = np.full((wh[1], wh[0], 3), 28, np.uint8)
    cv2.putText(img, text, (40, wh[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2, cv2.LINE_AA)
    return img


class CameraApp:
    """Wires camera -> controller -> preview/status, and capture jobs -> finalize -> inspection."""

    def __init__(self, source, cfg, intr=None, save_dir="captures", inspect=None):
        self.cfg = dict(cfg)
        self.intr = intr
        self.save_dir = os.path.abspath(save_dir)
        self.inspect = inspect  # None (capture only) or kwargs for run_inspection
        self.cam = PushSource() if str(source) == "phone" else CameraSource(source, cfg["width"], cfg["height"], cfg["fps"])
        self.ctl = CaptureController(cfg, intr)
        self.cond = threading.Condition()
        self.jpeg, self.jseq = None, 0
        self.processed_seq = 0  # camera seq of the newest processed frame (the phone waits for its own)
        self.status = dict(state="starting", message="Starting camera...")
        self.last_capture = None
        self.proc_ms = 0.0
        self._jobs = collections.deque()
        self._job_ev = threading.Event()
        self._insp_jobs = collections.deque()
        self._insp_ev = threading.Event()
        self._stop = threading.Event()

    def start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.cam.start()
        threading.Thread(target=self._process_loop, daemon=True, name="process").start()
        threading.Thread(target=self._worker, daemon=True, name="capture").start()
        threading.Thread(target=self._inspector, daemon=True, name="inspection").start()

    def stop(self):
        self._stop.set()
        self.cam.stop()
        self._job_ev.set()
        self._insp_ev.set()

    def _publish(self, img, st, seq=None):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["preview_jpeg_quality"])])
        with self.cond:
            if ok:
                self.jpeg, self.jseq = buf.tobytes(), self.jseq + 1
            self.status = st
            if seq is not None:
                self.processed_seq = seq
            self.cond.notify_all()

    def _process_loop(self):
        seq = 0
        while not self._stop.is_set():
            item = self.cam.wait_frame(seq, timeout=1.0)
            if item is None:
                if not self.cam.connected:
                    self.ctl.reset()
                    phone = self.cam.kind == "phone"
                    self._publish(_placeholder("Waiting for the phone camera..." if phone else
                                               "Camera not connected. Retrying..."),
                                  self._decorate(dict(state="disconnected", message="Waiting for the phone camera."
                                                      if phone else "Camera not connected.",
                                                      hint=self.cam.error or "", gates={}, groups={}, metrics={})))
                continue
            seq, t, frame = item
            t0 = time.monotonic()
            try:
                st, job = self.ctl.process(frame, t)
            except Exception as e:  # noqa: BLE001  (a bad frame must never stop the live view)
                log.exception("frame processing failed")
                self.ctl.reset()
                st, job = dict(state="search", message=MSG["search"], hint="", error=f"processing error: {e}",
                               gates={}, groups={}, metrics={}), None
            if job is not None:
                job["source"] = self.cam.desc
                self._jobs.append(job)
                self._job_ev.set()
            self.proc_ms = 0.9 * self.proc_ms + 0.1 * (time.monotonic() - t0) * 1000
            try:
                img = render_preview(frame, st, self.cfg["preview_px"])
            except Exception:  # noqa: BLE001
                img = frame
            self._publish(img, self._decorate(st), seq)

    def _decorate(self, st):
        st = dict(st)
        st.pop("t", None)
        st["camera"] = dict(connected=self.cam.connected, source=self.cam.desc, size=self.cam.size,
                            fps=round(self.cam.fps, 1), processing_ms=round(self.proc_ms, 1),
                            lens="calibrated" if self.intr else "uncalibrated")
        st["capture"] = self.last_capture
        return st

    def _worker(self):
        while not self._stop.is_set():
            self._job_ev.wait(1.0)
            self._job_ev.clear()
            while self._jobs:
                job = self._jobs.popleft()
                try:
                    res = finalize_capture(job, self.cfg, self.intr, self.save_dir, job.get("source", ""))
                except Exception as e:  # noqa: BLE001
                    log.warning("capture failed: %s", e)
                    self.ctl.capture_failed(f"Capture failed ({e}). Returning to live alignment.")
                    continue
                self.ctl.capture_done(job["sig"])
                m = res["meta"]
                cap = dict(id=res["id"], timestamp=m["capture"]["timestamp"], manual=m["capture"]["manual"],
                           files={k: os.path.relpath(res[k], self.save_dir) for k in
                                  ("original", "rectified", "calibration", "detection")},
                           mm_per_px=m["mm_per_px"], px_per_mm=m["pixel_to_mm"]["px_per_mm_x"],
                           rectified_wh=m["rectified"]["size_wh"], orientation=m["reference_frame"]["orientation"],
                           tilt_deg=m["quality"]["tilt_deg"], tilt_estimate=m["camera"]["model"],
                           alignment_score=m["quality"]["alignment_score"],
                           sharpness_score=m["quality"]["sharpness_score"], finalize_ms=m["timing"]["finalize_ms"],
                           latency_ms=round((time.monotonic() - job["t"]) * 1000), inspection=None)
                log.info("captured %s (%.4f mm/px) in %d ms", res["id"], cap["mm_per_px"], cap["latency_ms"])
                if self.inspect is not None:
                    cap["inspection"] = dict(state="queued" if self._insp_jobs else "running")
                    self._insp_jobs.append((cap, res))
                    self._insp_ev.set()
                self.last_capture = cap

    def _inspector(self):
        """Inspections run here, one at a time, so a slow inspection (tens of seconds) never delays the next
        capture: captures are saved at once and their inspections queue."""
        while not self._stop.is_set():
            self._insp_ev.wait(1.0)
            self._insp_ev.clear()
            while self._insp_jobs:
                cap, res = self._insp_jobs.popleft()
                if self.last_capture is not None and self.last_capture["id"] == cap["id"]:
                    self.last_capture = dict(self.last_capture, inspection=dict(state="running"))
                try:
                    r = run_inspection(res["rectified"], os.path.join(res["dir"], "inspection"), **self.inspect)
                except Exception as e:  # noqa: BLE001
                    log.exception("inspection failed")
                    r = dict(ok=False, error=str(e))
                for k in ("image", "report", "log"):
                    if r.get(k):
                        r[k] = os.path.relpath(r[k], self.save_dir)
                r["state"] = "done" if r.get("ok") else "failed"
                log.info("inspection of %s: %s", cap["id"], r.get("drawing") or r.get("error") or r["state"])
                if self.last_capture is not None and self.last_capture["id"] == cap["id"]:
                    self.last_capture = dict(self.last_capture, inspection=r)

    def retake(self):
        self.last_capture = None
        self.ctl.retake()

    def manual_capture(self):
        self.ctl.request_manual()
