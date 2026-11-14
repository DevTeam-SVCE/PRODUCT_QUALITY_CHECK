"""Part/background segmentation (python3 -m unittest discover -s tests -v).

Synthetic scenes: a textured metal plate with a square through-window on white paper, plus a soft cast shadow
darker than the plate. The shadow must stay background with the 'edges' method; 'otsu' merges it (the failure
this method exists for)."""
import os
import sys
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from partvision import config, segment  # noqa: E402

W, H = 860, 1220
PLATE = np.array([[300, 350], [560, 350], [560, 800], [300, 800]], np.int32)
WINDOW = np.array([[380, 480], [480, 480], [480, 580], [380, 580]], np.int32)


def scene(shadow=True, plate_gray=125, shadow_gray=110, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 222, np.float32)
    if shadow:  # offset copy of the plate, heavily blurred: a soft penumbra ~20 px wide
        sh = np.zeros((H, W), np.float32)
        cv2.fillPoly(sh, [PLATE - [90, -60]], 1.0)
        sh = cv2.GaussianBlur(sh, (0, 0), 9)
        img = img * (1 - sh) + shadow_gray * sh
    plate = np.zeros((H, W), np.uint8)
    cv2.fillPoly(plate, [PLATE], 255)
    cv2.fillPoly(plate, [WINDOW], 0)
    tex = plate_gray + cv2.GaussianBlur(rng.normal(0, 6, (H, W)).astype(np.float32), (0, 0), 1.2) * 3
    img = np.where(plate > 0, tex, img)
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 1.5, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8), plate


def iou(cand, truth):
    m = np.zeros((H, W), np.uint8)
    cv2.drawContours(m, [cand["contour"]], -1, 255, -1)
    t = np.zeros((H, W), np.uint8)
    cv2.fillPoly(t, [PLATE], 255)
    return np.logical_and(m, t).sum() / np.logical_or(m, t).sum()


class SegmentTest(unittest.TestCase):
    def setUp(self):
        config.load()

    def test_edges_excludes_soft_shadow(self):
        img, _ = scene()
        cand, used, _ = segment.segment(img, "edges")
        self.assertEqual(used, "edges")
        self.assertGreater(iou(cand, None), 0.97)
        self.assertTrue(cand["pol"].startswith("dark"))

    def test_otsu_merges_shadow(self):
        img, _ = scene()
        cand, _, _ = segment.segment(img, "otsu")
        self.assertLess(iou(cand, None), 0.9)  # documents why 'edges' is the default

    def test_auto_prefers_edges(self):
        img, _ = scene()
        cand, used, tried = segment.segment(img, "auto")
        self.assertEqual(used, "edges")
        self.assertGreater(iou(cand, None), 0.97)

    def test_no_shadow_both_agree(self):
        img, _ = scene(shadow=False)
        for m in ("edges", "otsu"):
            cand, _, _ = segment.segment(img, m)
            self.assertGreater(iou(cand, None), 0.97, m)

    def test_window_reported_as_opening(self):
        img, _ = scene(shadow=False)
        cand, _, _ = segment.segment(img, "edges")
        self.assertEqual(len(cand["inner"]), 1)
        x, y, w, h = cv2.boundingRect(cand["inner"][0])
        self.assertAlmostEqual(x + w / 2, 430, delta=4)
        self.assertAlmostEqual(y + h / 2, 530, delta=4)

    def test_config_selects_method(self):
        img, _ = scene()
        config.C["segmentation"]["method"] = "otsu"
        _, used, _ = segment.segment(img)
        self.assertEqual(used, "otsu")

    def test_unknown_method(self):
        img, _ = scene(shadow=False)
        with self.assertRaises(SystemExit):
            segment.segment(img, "nope")


if __name__ == "__main__":
    unittest.main()
