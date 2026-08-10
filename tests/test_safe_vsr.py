import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_vsr import (
    SAFE_VSR_MAX_EDGE,
    assert_channel_integrity,
    plan_dimensions,
)


class DimensionPlanningTests(unittest.TestCase):
    def test_15360_target_uses_direct_vsr(self):
        plan = plan_dimensions(
            input_width=3840,
            input_height=3840,
            requested_width=15360,
            requested_height=15360,
        )
        self.assertFalse(plan.uses_safe_fallback)
        self.assertEqual((plan.vsr_width, plan.vsr_height), (15360, 15360))

    def test_16384_target_uses_verified_safe_intermediate(self):
        plan = plan_dimensions(
            input_width=4096,
            input_height=4096,
            requested_width=16384,
            requested_height=16384,
        )
        self.assertTrue(plan.uses_safe_fallback)
        self.assertEqual((plan.output_width, plan.output_height), (16384, 16384))
        self.assertEqual((plan.vsr_width, plan.vsr_height), (SAFE_VSR_MAX_EDGE,) * 2)

    def test_non_square_target_preserves_safe_edge_and_aspect(self):
        plan = plan_dimensions(
            input_width=4096,
            input_height=2048,
            requested_width=16384,
            requested_height=8192,
        )
        self.assertEqual((plan.vsr_width, plan.vsr_height), (15360, 7680))

    def test_downscale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports upscaling"):
            plan_dimensions(
                input_width=4096,
                input_height=4096,
                requested_width=2048,
                requested_height=2048,
            )


class ChannelIntegrityTests(unittest.TestCase):
    def test_normal_rgb_is_accepted(self):
        source = torch.full((3, 4, 4), 0.4)
        output = torch.full((3, 8, 8), 0.42)
        assert_channel_integrity(source, output)

    def test_blue_channel_collapse_is_rejected(self):
        source = torch.full((3, 4, 4), 0.4)
        output = torch.full((3, 8, 8), 0.4)
        output[2].zero_()
        with self.assertRaisesRegex(RuntimeError, "blue channel collapsed"):
            assert_channel_integrity(source, output)

    def test_nan_is_rejected(self):
        source = torch.full((3, 4, 4), 0.4)
        output = torch.full((3, 8, 8), 0.4)
        output[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(RuntimeError, "NaN or infinite"):
            assert_channel_integrity(source, output)


if __name__ == "__main__":
    unittest.main()
