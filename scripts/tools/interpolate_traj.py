"""
Interpolate TUM trajectory with cubic B-spline given timestamps.
"""

import argparse
from pathlib import Path

import torch

from bad_gaussians.bad_utils import TrajectoryIO
from bad_gaussians.spline import SplineConfig

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)


def main():
    parser = argparse.ArgumentParser(description="Interpolate TUM trajectory with cubic B-spline given timestamps.")
    parser.add_argument("--input", type=str, required=True, help="Path to the TUM trajectory.")
    parser.add_argument("--times", type=str, required=True, help="Path to the timestamps.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output TUM trajectory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    times_path = Path(args.times)
    if not times_path.exists():
        raise FileNotFoundError(f"File not found: {times_path}")

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"File already exists: {output_path}")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    print(f"Interpolating TUM trajectory from {input_path} to {output_path} using timestamps from {times_path}")

    timestamps, tum_trajectory = TrajectoryIO.load_tum_trajectory(input_path)

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

    # read timestamps from times_path
    timestamps = []
    with open(times_path, 'r') as f:
        for line in f:
            timestamps.append(float(line.strip()))

    timestamps = torch.tensor(timestamps, dtype=torch.float64)
    linear_poses = linear_spline(timestamps)
    cubic_poses = cubic_spline(timestamps)

    TrajectoryIO.write_tum_trajectory(output_path, timestamps, cubic_poses)


main()
