"""Scenario tests for the A4 reference-frame capture (python3 -m unittest discover -s tests -v).

Frames come from tests/synth.py: an exact pinhole projection, so corners, tilt, scale and the far side are
known. Timestamps are simulated at 30 fps, so latencies are in camera time, independent of this machine."""
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import synth  # noqa: E402
from partvision import camera, config, refframe as rf  # noqa: E402

FPS = 30.0
CAM = synth.Camera()
WORLD = synth.make_world()
_cache = {}


def frame(world=None, key=None, **kw):
    k = (key, tuple(sorted(kw.items()))) if key or world is None else None
    if k and k in _cache:
        return _cache[k]
    f = CAM.render(WORLD if world is None else world, **kw)
    if k:
        _cache[k] = f
    return f


def cfg(**over):
    c = dict(config.load()["camera"])
    c.update(over)
    return c


def run(ctl, frames, t0=0.0, stop_on_job=True):
    """Feed frames (list, or callable i -> frame, with n) at 30 fps. Returns (statuses, jobs, index of 1st job)."""
    sts, jobs, first = [], [], None
    for i, f in enumerate(frames):
        st, job = ctl.process(f, t0 + i / FPS)
        sts.append(st)
        if job:
            jobs.append(job)
            first = i if first is None else first
            if stop_on_job:
                break
    return sts, jobs, first


def static(n, **kw):
    f = frame(key="static", **kw)
    return [f] * n


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pqc_cam_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def capture(self, ctl, frames):
        sts, jobs, i = run(ctl, frames)
        self.assertTrue(jobs, f"no capture; last state {sts[-1]['state']}: {sts[-1]['message']} {sts[-1]['hint']}")
        return sts, jobs[0], i


class TestCaptureScenarios(Base):

    def test_aligned_capture_fast_and_accurate(self):
        c = cfg()
        ctl = camera.CaptureController(c)
        sts, job, i = self.capture(ctl, static(30))
        self.assertLessEqual(i / FPS, 0.35, "capture should follow ~stable_ms after the frame is first seen")
        self.assertEqual(sts[i]["state"], "ready")
        self.assertEqual(sts[i]["message"], camera.MSG["ready"])
        res = camera.finalize_capture(job, c, None, self.tmp)
        m = res["meta"]
        truth = CAM.corners()
        err = np.abs(np.array(m["corners"]["undistorted_px"]) - truth).max()
        self.assertLess(err, 0.5, "full-resolution corners")
        true_ppm = np.linalg.norm(truth[1] - truth[0]) / 210.0
        self.assertAlmostEqual(m["pixel_to_mm"]["px_per_mm_x"] / true_ppm, 1.0, delta=0.003)
        self.assertEqual(m["pixel_to_mm"]["mm_per_px_x"], m["pixel_to_mm"]["mm_per_px_y"])
        self.assertLess(m["quality"]["tilt_deg"], 0.2)
        for k in ("original", "rectified", "calibration", "detection"):
            self.assertTrue(os.path.isfile(res[k]), k)
        rect = cv2.imread(res["rectified"])
        self.assertEqual([rect.shape[1], rect.shape[0]], m["rectified"]["size_wh"])
        self.assertAlmostEqual(rect.shape[0] / rect.shape[1], (297 - 4) / (210 - 4), delta=0.01)
        for key in ("homography", "corners", "camera", "capture", "quality"):
            self.assertIn(key, m)
        self.assertIn("timestamp", m["capture"])

    def test_slight_tilt_captures_with_correct_estimate(self):
        pose = dict(tilt_x=1.2, tilt_y=0.6)
        c = cfg()
        sts, job, i = self.capture(camera.CaptureController(c), static(30, **pose))
        m = camera.finalize_capture(job, c, None, self.tmp)["meta"]
        self.assertAlmostEqual(m["quality"]["tilt_deg"], math.hypot(1.2, 0.6), delta=0.15)

    def test_significant_tilt_rejected_with_direction_hint(self):
        for pose in (dict(tilt_x=6, cy=148.5 - 55), dict(tilt_y=-5, cx=105 - 45)):
            ctl = camera.CaptureController(cfg())
            sts, jobs, _ = run(ctl, static(40, **pose))
            self.assertFalse(jobs, pose)
            st = sts[-1]
            self.assertEqual(st["message"], camera.MSG["angle"], pose)
            self.assertIn(CAM.far_side(**pose), st["hint"], pose)
            self.assertEqual(st["groups"]["alignment"]["status"], "fail")

    def test_tilt_just_over_threshold_rejected(self):
        ctl = camera.CaptureController(cfg())
        sts, jobs, _ = run(ctl, static(40, tilt_x=2.6, cy=148.5 - 24))
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["message"], camera.MSG["angle"])

    def test_partially_outside_view(self):
        ctl = camera.CaptureController(cfg())
        sts, jobs, _ = run(ctl, static(20, cx=105 + 260))
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["state"], "partial")
        self.assertEqual(sts[-1]["message"], camera.MSG["partial"])
        # touching the border: detected but not fully visible
        sts, jobs, _ = run(camera.CaptureController(cfg()), static(20, cy=148.5 + 40))
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["message"], camera.MSG["partial"])

    def test_no_reference_frame(self):
        w = synth.make_world(sheet_gray=150)
        sts, jobs, _ = run(camera.CaptureController(cfg()), [CAM.render(w)] * 10)
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["message"], camera.MSG["search"])

    def test_moving_frame_never_captured(self):
        frames = [CAM.render(WORLD, cx=105 + 0.8 * i, seed=i) for i in range(24)]
        sts, jobs, _ = run(camera.CaptureController(cfg()), frames)
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["message"], camera.MSG["steady"])

    def test_camera_shake_never_captured(self):
        rng = np.random.default_rng(3)
        # hand-held shake: ~1 mm / 0.2 deg frame to frame (about 3 px at 1080p)
        frames = [CAM.render(WORLD, cx=105 + rng.normal(0, 1.0), cy=148.5 + rng.normal(0, 1.0),
                             yaw=rng.normal(0, 0.2), seed=i) for i in range(24)]
        sts, jobs, _ = run(camera.CaptureController(cfg()), frames)
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["message"], camera.MSG["steady"])

    def test_part_being_placed_waits_until_still(self):
        model = _model("circular_top")
        moving = [CAM.render(synth.make_world(part=model, part_at=(60 + 4 * i, 150)), seed=i) for i in range(12)]
        still = [CAM.render(synth.make_world(part=model, part_at=(104, 150)))] * 20
        sts, jobs, i = run(camera.CaptureController(cfg()), moving + still)
        self.assertTrue(jobs)
        self.assertGreaterEqual(i, len(moving), "captured while the part was still moving")
        self.assertTrue(any(s["message"] == camera.MSG["steady"] for s in sts[3:len(moving)]))

    def test_poor_lighting(self):
        for kw, why in ((dict(gain=0.16), "too dark"), (dict(gain=1.3, offset=10), "glare")):
            sts, jobs, _ = run(camera.CaptureController(cfg()), [CAM.render(WORLD, **kw)] * 30)
            self.assertFalse(jobs, kw)
            self.assertEqual(sts[-1]["message"], camera.MSG["light"], kw)
            self.assertIn(why, sts[-1]["hint"])

    def test_shadow_across_sheet(self):
        """A soft shadow over part of the sheet: corrected in the rectified image; a deep one is not captured."""
        img = CAM.render(synth.make_world(part=_model("circular_top"), part_at=(105, 170)))
        H, W = img.shape[:2]

        def shade(depth):
            ramp = np.clip((np.arange(H, dtype=np.float32) - 0.2 * H) / (0.35 * H), 0, 1)  # dark top -> lit
            k = (1 - depth) + depth * ramp
            return np.clip(img * k[:, None, None], 0, 255).astype(np.uint8)

        c = cfg()
        sts, job, _ = self.capture(camera.CaptureController(c), [shade(0.35)] * 30)
        m = camera.finalize_capture(job, c, None, self.tmp)
        il = m["meta"]["illumination"]
        self.assertLess(il["uniformity_before"], 0.8)
        self.assertGreater(il["uniformity_after"], 0.9)
        rect = cv2.cvtColor(cv2.imread(m["rectified"]), cv2.COLOR_BGR2GRAY)
        self.assertLess(np.percentile(rect, 3), 120, "the part must stay dark after the correction")
        self.assertTrue(os.path.isfile(os.path.join(m["dir"], il["uncorrected_file"])))
        sts, jobs, _ = run(camera.CaptureController(cfg()), [shade(0.65)] * 30)
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["message"], camera.MSG["light"])
        self.assertIn("shadow", sts[-1]["hint"])

    def test_blurred_image(self):
        for kw in (dict(blur=3.5), dict(motion=(7, 0))):
            sts, jobs, _ = run(camera.CaptureController(cfg()), [CAM.render(WORLD, **kw)] * 30)
            self.assertFalse(jobs, kw)
            self.assertEqual(sts[-1]["message"], camera.MSG["blur"], kw)

    def test_distances(self):
        expect = {380: camera.MSG["partial"], 470: None, 650: None, 950: camera.MSG["closer"]}
        for Z, msg in expect.items():
            sts, jobs, _ = run(camera.CaptureController(cfg()), [CAM.render(WORLD, Z=Z)] * 30)
            if msg is None:
                self.assertTrue(jobs, f"Z={Z}: {sts[-1]['message']}")
            else:
                self.assertFalse(jobs, f"Z={Z}")
                self.assertEqual(sts[-1]["message"], msg, f"Z={Z}")

    def test_portrait_landscape_and_rotated(self):
        for yaw, orient in ((0, "portrait"), (90, "landscape"), (-90, "landscape"), (18, "portrait"), (75, "landscape")):
            c = cfg()
            sts, job, _ = self.capture(camera.CaptureController(c), [CAM.render(WORLD, yaw=yaw, Z=560)] * 30)
            m = camera.finalize_capture(job, c, None, self.tmp)["meta"]
            self.assertEqual(m["reference_frame"]["orientation"], orient, yaw)
            w, h = m["rectified"]["size_wh"]
            self.assertEqual(w > h, orient == "landscape", yaw)
            truth = CAM.corners(yaw=yaw, Z=560)
            got = np.array(m["corners"]["undistorted_px"])
            self.assertLess(min(np.abs(np.roll(got, k, 0) - truth).max() for k in range(4)), 0.6, yaw)

    def test_letter_sheet_is_not_a4(self):
        w = synth.make_world(sheet_mm=(215.9, 279.4))
        sts, jobs, _ = run(camera.CaptureController(cfg()), [CAM.render(w, cx=108, cy=140)] * 30)
        self.assertFalse(jobs)
        self.assertEqual(sts[-1]["message"], camera.MSG["aspect"])

    def test_occluded_edge(self):
        # a finger over ~20% of the left edge: robust line fit ignores it, corners stay accurate
        small = [[-15, 120], [25, 120], [25, 180], [-15, 180]]
        c = cfg()
        sts, job, _ = self.capture(camera.CaptureController(c), [CAM.render(synth.make_world(occluder=small))] * 30)
        m = camera.finalize_capture(job, c, None, self.tmp)["meta"]
        self.assertLess(np.abs(np.array(m["corners"]["undistorted_px"]) - CAM.corners()).max(), 0.6)
        # most of the left edge covered: no capture
        big = [[-15, 40], [25, 40], [25, 250], [-15, 250]]
        sts, jobs, _ = run(camera.CaptureController(cfg()), [CAM.render(synth.make_world(occluder=big))] * 30)
        self.assertFalse(jobs)


class TestPhoneSource(Base):

    def test_slow_phone_browser_frames_capture(self):
        """The phone's browser sends ~4-10 fps: capture still needs >= 3 steady frames over >= stable_ms."""
        f = frame(key="static")
        for dt in (0.1, 0.14, 0.25):
            ctl = camera.CaptureController(cfg())
            job, i = None, 0
            for i in range(20):
                st, job = ctl.process(f, i * dt)
                if job:
                    break
            self.assertIsNotNone(job, dt)
            self.assertGreaterEqual(i * dt, cfg()["stable_ms"] / 1000 - 1e-6, dt)
            self.assertLessEqual(i * dt, 0.55, dt)

    def test_push_source(self):
        src = camera.PushSource(timeout=0.5)
        src.start()
        try:
            self.assertIsNone(src.wait_frame(0, timeout=0.1))
            ok, buf = cv2.imencode(".jpg", frame(key="static"))
            src.push(buf.tobytes())
            seq, t, img = src.wait_frame(0, timeout=1.0)
            self.assertEqual(img.shape[:2], (1080, 1920))
            with self.assertRaises(ValueError):
                src.push(b"not an image")
            time.sleep(1.2)
            self.assertFalse(src.connected)
        finally:
            src.stop()


class TestCaptureLock(Base):

    def test_duplicate_prevention_retake_and_rearm(self):
        c = cfg()
        ctl = camera.CaptureController(c)
        f0 = frame(key="static")
        sts, jobs, i = run(ctl, [f0] * 30)
        self.assertEqual(len(jobs), 1)
        ctl.capture_done(jobs[0]["sig"])
        # same scene for 3 s: no second capture
        sts, jobs, _ = run(ctl, [f0] * 90, t0=1.0, stop_on_job=False)
        self.assertEqual(jobs, [])
        self.assertEqual(sts[-1]["state"], "captured")
        self.assertEqual(sts[-1]["message"], camera.MSG["captured"])
        # retake: captures again
        ctl.retake()
        sts, jobs, i = run(ctl, [f0] * 30, t0=5.0)
        self.assertEqual(len(jobs), 1)
        self.assertLessEqual(i / FPS, 0.35)
        ctl.capture_done(jobs[0]["sig"])
        # camera moved: the lock releases by itself, capture after the new position is stable
        f1 = CAM.render(WORLD, cx=105 + 6)
        sts, jobs, _ = run(ctl, [f1] * 30, t0=8.0)
        self.assertEqual(len(jobs), 1, sts[-1]["message"])
        ctl.capture_done(jobs[0]["sig"])
        # a new part placed (frame unchanged): captured as a new scene
        f2 = CAM.render(synth.make_world(part=_model("box_front")), cx=105 + 6)
        sts, jobs, _ = run(ctl, [f2] * 30, t0=11.0)
        self.assertEqual(len(jobs), 1, sts[-1]["message"])

    def test_no_capture_while_busy_and_failure_returns_to_live(self):
        c = cfg()
        ctl = camera.CaptureController(c)
        f0 = frame(key="static")
        sts, jobs, _ = run(ctl, [f0] * 30)
        self.assertEqual(len(jobs), 1)
        sts, more, _ = run(ctl, [f0] * 20, t0=2.0, stop_on_job=False)  # finalising: no second job
        self.assertEqual(more, [])
        self.assertEqual(sts[-1]["state"], "capturing")
        bad = dict(jobs[0], frame=np.full_like(f0, 128))  # e.g. a corrupted frame
        with self.assertRaises(RuntimeError):
            camera.finalize_capture(bad, c, None, self.tmp)
        ctl.capture_failed("Capture failed (test). Returning to live alignment.", t=3.0)
        sts, jobs, _ = run(ctl, [f0] * 30, t0=3.0)
        self.assertTrue(jobs, "back in live mode, captures again")
        self.assertIn("Capture failed", sts[0].get("error", ""))

    def test_manual_capture_records_failing_checks(self):
        ctl = camera.CaptureController(cfg())
        f = CAM.render(WORLD, tilt_x=5, cy=148.5 - 45)
        run(ctl, [f] * 10)
        ctl.request_manual()
        st, job = ctl.process(f, 1.0)
        self.assertIsNotNone(job)
        self.assertTrue(job["manual"])
        self.assertIn("alignment", job["failing"])


class TestGeometry(unittest.TestCase):

    def test_tilt_estimate_is_stable_near_frontal(self):
        """Regression: IPPE gave ~1.3 deg of spurious tilt for 0.1 px corner noise near a frontal view."""
        rng = np.random.default_rng(0)
        q0 = CAM.corners()
        t = [rf.pose(q0 + rng.normal(0, 0.1, (4, 2)), CAM.K, (210, 297))["tilt_deg"] for _ in range(200)]
        self.assertLess(np.percentile(t, 95), 0.2)

    def test_assumed_fov_error_scales_tilt(self):
        """Without lens calibration the tilt is estimated with an assumed FOV; a 15% wrong focal length gives a
        proportionally wrong tilt (documented limitation), still the right order of magnitude."""
        q = CAM.corners(tilt_x=2.0)
        K = CAM.K.copy()
        K[0, 0] *= 0.85
        K[1, 1] *= 0.85
        self.assertAlmostEqual(rf.pose(q, K, (210, 297))["tilt_deg"], 2.0, delta=0.5)

    def test_lens_distortion_corrected_with_intrinsics(self):
        dist = np.array([-0.12, 0.05, 0, 0, 0], float)
        und = CAM.render(WORLD)
        H, W = und.shape[:2]
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        p = cv2.undistortPoints(np.c_[xs.ravel(), ys.ravel()].reshape(-1, 1, 2), CAM.K, dist, P=CAM.K).reshape(H, W, 2)
        distorted = cv2.remap(und, p[..., 0], p[..., 1], cv2.INTER_LINEAR)
        intr = dict(K=CAM.K, dist=dist, size=(W, H), rms=0.1, path="synthetic")
        c = cfg()
        ctl = camera.CaptureController(c, intr)
        sts, jobs, _ = run(ctl, [distorted] * 30)
        self.assertTrue(jobs, sts[-1]["message"])
        with tempfile.TemporaryDirectory() as d:
            m = camera.finalize_capture(jobs[0], c, intr, d)["meta"]
        self.assertLess(np.abs(np.array(m["corners"]["undistorted_px"]) - CAM.corners()).max(), 0.6)
        self.assertEqual(m["camera"]["model"], "calibrated")


class TestRealPhoto(unittest.TestCase):
    PHOTO = os.environ.get("PQC_REAL_PHOTO", "")

    @unittest.skipUnless(os.path.isfile(PHOTO), "set PQC_REAL_PHOTO to a photo of the A4 frame")
    def test_real_photo_detected(self):
        img = cv2.imread(self.PHOTO)
        sts, jobs, _ = run(camera.CaptureController(cfg()), [img] * 20, stop_on_job=False)
        self.assertTrue(sts[-1]["gates"]["detected"]["ok"])
        print("\nreal photo:", sts[-1]["message"], sts[-1]["hint"], f"tilt {sts[-1]['metrics']['tilt_deg']:.1f} deg")


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def _model(name):
    from partvision import drawing
    config.load()
    return drawing.load(os.path.join(ROOT, "dxf_files", name + ".dxf"), 0.5)[0]


class TestEndToEnd(Base):

    def test_camera_to_inspection(self):
        """Still-image source -> live app (headless) -> automatic capture -> rectified + calibration sidecar ->
        existing drawing matcher (inspect_all) with the calibrated absolute scale."""
        model = _model("circular_top")
        img = CAM.render(synth.make_world(part=model, part_at=(105, 150), part_rot=17), Z=480)
        src = os.path.join(self.tmp, "scene.png")
        cv2.imwrite(src, img)
        from partvision import camera_ui
        save = os.path.join(self.tmp, "caps")
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            t0 = time.monotonic()
            rc = camera_ui.main(["--source", src, "--headless", "--save-dir", save, "--drawings", "dxf_files",
                                 "--timeout", "120"])
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0)
        (cid,) = os.listdir(save)
        d = os.path.join(save, cid)
        meta = json.load(open(os.path.join(d, cid + "_rectified.calib.json")))
        summ = json.load(open(os.path.join(d, "inspection", "summary.json")))[0]
        self.assertEqual(os.path.basename(summ["drawing"] or ""), "circular_top.dxf")
        rep = json.load(open(os.path.join(d, "inspection", "data", cid + "_rectified_drawing_check_report.json")))
        self.assertEqual(rep["calibration"]["mm_per_px"], meta["mm_per_px"])
        chk = rep["calibration"]["scale_check"]  # independent scale vs the drawing's fitted scale
        self.assertLess(abs(chk["deviation_pct"]), 0.4, chk)
        self.assertEqual(chk["status"], "UNVERIFIED")  # no circular_top.part.json: height unknown
        self.assertTrue(chk["plausible"])
        self.assertLess(abs(chk["implied_height_mm"]), 3.0, chk)  # rendered flat on the sheet
        counts = summ["status_counts"]
        self.assertEqual(counts.get("DEVIATES", 0), 0, counts)
        self.assertLessEqual(counts.get("MISSING", 0), 1, counts)  # one hole the synthetic render does not show
        # drawing-free analysis of the same capture reports mm from the calibration
        ana = json.load(open(os.path.join(d, "inspection", "data", cid + "_rectified_analysis_report.json")))
        self.assertEqual(ana["mm_per_px"], meta["mm_per_px"])
        print(f"\ne2e: {counts}, scale check {chk['deviation_pct']:+.2f}% in {time.monotonic() - t0:.1f} s")


class TestNewPart(Base):

    @staticmethod
    def spec(k=1.0):
        """A part that is not in dxf_files: a plate with an asymmetric hole pattern and a slot (k scales it)."""
        return {"outline": {"type": "rect", "length": 120 * k, "width": 70 * k},
                "holes": [{"id": "H1", "x": 15 * k, "y": 15 * k, "d": 8 * k}, {"id": "H2", "x": 105 * k, "y": 15 * k, "d": 8 * k},
                          {"id": "H3", "x": 15 * k, "y": 55 * k, "d": 6 * k}, {"id": "H4", "x": 100 * k, "y": 52 * k, "d": 12 * k}],
                "openings": [{"id": "S1", "type": "slot", "x": 55 * k, "y": 40 * k, "length": 30 * k, "width": 8 * k,
                              "angle": 0}]}

    def test_new_part_added_by_dropping_files_only(self):
        """Adding a part = dropping its drawing (+ optional NAME.part.json) in the drawings folder. A look-alike
        drawing at 80% size is ruled out by the reference frame's absolute scale."""
        from partvision import drawing
        dd = os.path.join(self.tmp, "drawings")
        os.makedirs(dd)
        _write_json(os.path.join(dd, "bracket.json"), self.spec())
        _write_json(os.path.join(dd, "bracket_small.json"), self.spec(0.8))
        _write_json(os.path.join(dd, "bracket.part.json"), {"height_mm": 0.0})  # flat in this render
        shutil.copy(os.path.join(ROOT, "dxf_files", "circular_top.dxf"), dd)
        config.load()
        model = drawing.load(os.path.join(dd, "bracket.json"), 0.5)[0]
        img = CAM.render(synth.make_world(part=model, part_at=(100, 160), part_rot=-12), Z=470)
        c = cfg()
        sts, job, _ = self.capture(camera.CaptureController(c), [img] * 30)
        res = camera.finalize_capture(job, c, None, self.tmp)
        r = camera.run_inspection(res["rectified"], os.path.join(res["dir"], "inspection"), drawings=[dd])
        log = open(r["log"]).read()
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["drawing"], "bracket.json", log)
        self.assertIn("bracket_small.json", log)
        self.assertRegex(log, r"bracket_small\.json\s+score")  # same geometry: matched scale-free ...
        self.assertIn("size tie-break: bracket.json", log)   # ... and the calibrated size picks the right one
        self.assertNotIn("bracket.part.json", log.split("drawing ")[0] + "".join(
            ln for ln in log.splitlines() if ln.startswith("drawing ")))  # metadata is not loaded as a drawing
        chk = r["size_check"]
        self.assertEqual(chk["status"], "OK", chk)
        self.assertEqual(chk["height_source"], "drawing metadata")
        self.assertLess(abs(chk["deviation_pct"]), 0.5, chk)
        self.assertEqual(r["status_counts"].get("DEVIATES", 0), 0, r["status_counts"])
        self.assertEqual(r["status_counts"].get("MISSING", 0), 0, r["status_counts"])


class TestPartMeta(unittest.TestCase):

    def test_size_check_bounds(self):
        from partvision import partmeta
        config.load()
        calib = {"mm_per_px": 0.4, "pixel_to_mm": {"mm_per_px_x": 0.4}, "quality": {"camera_distance_mm": 500.0}}
        ok = partmeta.size_check(calib, 0.4 / 1.06, None, 100, 0.5)  # looks 6% larger: a ~28 mm high face
        self.assertTrue(ok["plausible"])
        self.assertEqual(ok["status"], "UNVERIFIED")
        self.assertAlmostEqual(ok["implied_height_mm"], 500 * (1 - 1 / 1.06), places=6)
        self.assertFalse(partmeta.size_check(calib, 0.4 / 0.9, None, 100, 0.5)["plausible"])   # 10% smaller
        self.assertFalse(partmeta.size_check(calib, 0.4 / 1.3, None, 100, 0.5)["plausible"])   # 30% larger
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.dxf")
            _write_json(os.path.join(d, "x.part.json"), {"height_mm": 500 * (1 - 1 / 1.06)})
            known = partmeta.size_check(calib, 0.4 / 1.06, p, 100, 0.5)
            self.assertEqual(known["status"], "OK")
            self.assertLess(abs(known["deviation_over_part_mm"]), 1e-6)


class TestCameraSource(unittest.TestCase):

    def test_disconnected_camera_reported_not_crashing(self):
        app = camera.CameraApp("/nonexistent/stream.mjpg", cfg(), None, tempfile.mkdtemp(prefix="pqc_cam_"), None)
        app.start()
        try:
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline and app.status.get("state") != "disconnected":
                time.sleep(0.1)
            self.assertEqual(app.status["state"], "disconnected")
            self.assertFalse(app.status["camera"]["connected"])
        finally:
            app.stop()


if __name__ == "__main__":
    unittest.main()
