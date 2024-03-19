"""
Export poses from poses_bounds.npy
"""
import argparse
from pathlib import Path

import numpy as np
import pypose as pp
import torch

from bad_gaussians.bad_utils import TrajectoryIO

DEVICE = 'cpu'


def main():
    parser = argparse.ArgumentParser(description="Export poses from poses_bounds.npy.")
    parser.add_argument("--input", type=str, required=True, help="Path to the npy file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output TUM trajectory.")
    args = parser.parse_args()

    npy_path = Path(args.input)
    if not npy_path.exists():
        raise FileNotFoundError(f"File not found: {npy_path}")

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"File already exists: {output_path}")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    print(f"Exporting poses from {npy_path} to {output_path}")

    # Load data from npy file, shape -1, 17
    pose_bounds = np.load(npy_path)
    # (N, 17)
    assert pose_bounds.shape[-1] == 17
    # extract (N, 15) from (N, 17), reshape to (N, 3, 5)
    matrices = np.reshape(pose_bounds[:, :-2], (-1, 3, 5))

    # pop every 8th pose
    matrices = np.delete(matrices, np.arange(0, matrices.shape[0], 8), axis=0)

    poses = pp.from_matrix(torch.tensor(matrices[:, :, :4]).to(DEVICE), check=False, ltype=pp.SE3_type)

    num_cameras = poses.shape[0]
    timestamps = [float(x) for x in range(num_cameras)]
    TrajectoryIO.write_tum_trajectory(output_path, torch.tensor(timestamps), poses)


main()
