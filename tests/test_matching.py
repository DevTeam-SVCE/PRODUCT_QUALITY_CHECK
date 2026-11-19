"""Alignment evidence weights and hole coverage (python3 -m unittest discover -s tests -v)."""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from partvision import config  # noqa: E402
from partvision.drawing_check import align as dm  # noqa: E402


def hole(dx=0.0, r=7.0, **kw):
    return dict(exp_x=100.0, exp_y=100.0, exp_r=r, meas_x=100.0 + dx, meas_y=100.0, **kw)


class AgreementTest(unittest.TestCase):
    def setUp(self):
        config.load()

    def test_on_target_good_hole_is_full_credit(self):
        self.assertAlmostEqual(dm._hole_agree(hole(quality_ok=True)), 1.0)

    def test_missing_hole_is_zero(self):
        self.assertEqual(dm._hole_agree(dict(exp_x=1, exp_y=1, exp_r=5, status="MISSING")), 0.0)

    def test_weak_wall_counts_partly(self):
        w = config.C["matching"]["weak_hole_weight"]
        self.assertAlmostEqual(dm._hole_agree(hole(quality_ok=False)), w)

    def test_round_edge_search_is_not_position_evidence(self):
        # a round edge searched at the expected place sits there by construction: low weight even when exact
        w = config.C["matching"]["round_edge_weight"]
        self.assertAlmostEqual(dm._hole_agree(hole(quality_ok=True, seen="round edge")), w)
        self.assertLess(w, config.C["matching"]["weak_hole_weight"])

    def test_falloff_in_radii(self):
        k = config.C["matching"]["agreement_falloff_radii"]
        self.assertAlmostEqual(dm._hole_agree(hole(dx=k * 7.0, quality_ok=True)), math.exp(-1))
        # a small hole a few px off (parallax of a raised face) still earns credit
        self.assertGreater(dm._hole_agree(hole(dx=4.0, quality_ok=True)), 0.5)

    def test_config_changes_weights(self):
        config.C["matching"]["weak_hole_weight"] = 0.0
        self.assertEqual(dm._hole_agree(hole(quality_ok=False)), 0.0)


if __name__ == "__main__":
    unittest.main()
