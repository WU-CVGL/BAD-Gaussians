"""
SE(3) B-spline trajectory

Created by lzzhao on 2023.09.29
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Type

import pypose as pp
import torch
from jaxtyping import Float
from pypose import LieTensor
from torch import nn, Tensor
from typing_extensions import assert_never

from nerfstudio.configs.base_config import InstantiateConfig

from bad_gaussians.spline_functor import linear_interpolation, cubic_bspline_interpolation


@dataclass
class SplineConfig(InstantiateConfig):
    """Configuration for spline instantiation."""

    _target: Type = field(default_factory=lambda: Spline)
    """Target class to instantiate."""

    degree: int = 1
    """Degree of the spline. 1 for linear spline, 3 for cubic spline."""

    sampling_interval: float = 0.1
    """Sampling interval of the control knots."""

    start_time: float = 0.0
    """Starting timestamp of the spline."""


class Spline(nn.Module):
    """SE(3) spline trajectory.

    Args:
        config: the SplineConfig used to instantiate class
    """

    config: SplineConfig
    data: Float[LieTensor, "num_knots 7"]
    start_time: float
    end_time: float
    t_lower_bound: float
    t_upper_bound: float

    def __init__(self, config: SplineConfig):
        super().__init__()
        self.config = config
        self.data = pp.identity_SE3(0)
        self.order = self.config.degree + 1
        """Order of the spline, i.e. control knots per segment, 2 for linear, 4 for cubic"""

        self.set_start_time(config.start_time)
        self.update_end_time()

    def __len__(self):
        return self.data.shape[0]

    def forward(self, timestamps: Float[Tensor, "*batch_size"]) -> Float[LieTensor, "*batch_size 7"]:
        """Interpolate the spline at the given timestamps.

        Args:
            timestamps: Timestamps to interpolate the spline at. Range: [t_lower_bound, t_upper_bound].

        Returns:
            poses: The interpolated pose.
        """
        segment, u = self.get_segment(timestamps)
        u = u[..., None]  # (*batch_size) to (*batch_size, interpolations=1)
        if self.config.degree == 1:
            poses = linear_interpolation(segment, u)
        elif self.config.degree == 3:
            poses = cubic_bspline_interpolation(segment, u)
        else:
            assert_never(self.config.degree)
        return poses.squeeze()

    def get_segment(
            self,
            timestamps: Float[Tensor, "*batch_size"]
    ) -> Tuple[
        Float[LieTensor, "*batch_size self.order 7"],
        Float[Tensor, "*batch_size"]
    ]:
        """Get the spline segment and normalized position on segment at the given timestamp.

        Args:
            timestamps: Timestamps to get the spline segment and normalized position at.

        Returns:
            segment: The spline segment.
            u: The normalized position on the segment.
        """
        assert torch.all(timestamps >= self.t_lower_bound)
        assert torch.all(timestamps <= self.t_upper_bound)
        batch_size = timestamps.shape
        relative_time = timestamps - self.start_time
        normalized_time = relative_time / self.config.sampling_interval
        start_index = torch.floor(normalized_time).int()
        u = normalized_time - start_index
        if self.config.degree == 3:
            start_index -= 1

        indices = (start_index.tile((self.order, 1)).T +
                   torch.arange(self.order).tile((*batch_size, 1)).to(start_index.device))
        indices = indices[..., None].tile(7)
        segment = pp.SE3(torch.gather(self.data.expand(*batch_size, -1, -1), 1, indices))

        return segment, u

    def insert(self, pose: Float[LieTensor, "1 7"]):
        """Insert a control knot"""
        self.data = pp.SE3(torch.cat([self.data, pose]))
        self.update_end_time()

    def set_data(self, data: Float[LieTensor, "num_knots 7"] | pp.Parameter):
        """Set the spline data."""
        self.data = data
        self.update_end_time()

    def set_start_time(self, start_time: float):
        """Set the starting timestamp of the spline."""
        self.start_time = start_time
        if self.config.degree == 1:
            self.t_lower_bound = self.start_time
        elif self.config.degree == 3:
            self.t_lower_bound = self.start_time + self.config.sampling_interval
        else:
            assert_never(self.config.degree)

    def update_end_time(self):
        """Update the ending timestamp of the spline."""
        self.end_time = self.start_time + self.config.sampling_interval * (len(self) - 1)
        if self.config.degree == 1:
            self.t_upper_bound = self.end_time
        elif self.config.degree == 3:
            self.t_upper_bound = self.end_time - self.config.sampling_interval
        else:
            assert_never(self.config.degree)
