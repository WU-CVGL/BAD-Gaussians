import unittest
from pathlib import Path

import pypose as pp
import torch

from bad_gaussians.bad_utils import TrajectoryIO
from bad_gaussians.spline import SplineConfig

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=5, sci_mode=False)


class TestSpline(unittest.TestCase):

    def test_spline_basic(self):
        print(f"\n{'='*10}{self.id()}{'='*10}")
        linear_spline_config = SplineConfig(degree=1)
        cubic_spline_config = SplineConfig(degree=3)
        linear_spline = linear_spline_config.setup()
        cubic_spline = cubic_spline_config.setup()

        p0 = pp.identity_SE3(1)
        p1 = pp.randn_SE3(1)
        p2 = pp.randn_SE3(1)
        p3 = pp.randn_SE3(1)

        p0.translation()[0] = 1
        p1.translation()[0] = 2
        p2.translation()[0] = 3
        p3.translation()[0] = 4

        for spline in [linear_spline, cubic_spline]:
            spline.insert(p0)
            spline.insert(p1)
            spline.insert(p2)
            spline.insert(p3)

        pose0 = linear_spline(torch.tensor([0.15001]))
        print(f"linear: {pose0}")
        self.assertTrue(torch.isclose(pose0.translation(), torch.tensor([2.5001, 2.5001, 2.5001]), atol=1e-3).all())
        pose1 = cubic_spline(torch.tensor([0.15001]))
        print(f"cubic: {pose1}")
        self.assertTrue(torch.isclose(pose1.translation(), torch.tensor([2.5001, 2.5001, 2.5001]), atol=1e-3).all())

    def test_spline_tum(self):
        print(f"\n{'='*10}{self.id()}{'='*10}")
        timestamps, tum_trajectory = TrajectoryIO.load_tum_trajectory(Path("data/traj.txt"))

        linear_spline_config = SplineConfig(
            degree=1,
            sampling_interval=(timestamps[1] - timestamps[0]),
            start_time=timestamps[0]
        )
        cubic_spline_config = SplineConfig(
            degree=3,
            sampling_interval=(timestamps[1] - timestamps[0]),
            start_time=timestamps[0]
        )
        linear_spline = linear_spline_config.setup()
        cubic_spline = cubic_spline_config.setup()

        linear_spline.set_data(tum_trajectory)
        cubic_spline.set_data(tum_trajectory)

        poses0 = linear_spline(torch.tensor([timestamps[20], timestamps[30]]))
        print(f"linear: {poses0}")
        self.assertTrue(torch.isclose(poses0[0], tum_trajectory[20], atol=1e-3).all())
        self.assertTrue(torch.isclose(poses0[1], tum_trajectory[30], atol=1e-3).all())
        poses1 = cubic_spline(torch.tensor([timestamps[20], timestamps[30]]))
        print(f"cubic: {poses1}")
        self.assertTrue(torch.isclose(poses1[0], tum_trajectory[20], atol=1e-3).all())
        self.assertTrue(torch.isclose(poses1[1], tum_trajectory[30], atol=1e-3).all())


if __name__ == "__main__":
    unittest.main()
