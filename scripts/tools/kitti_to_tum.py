"""
Convert KITTI trajectory to TUM format.

Usage:
python tools/kitti_to_tum.py \
    --input=data/Replica/office0/traj.txt \
    --output=data/Replica/office0/traj_tum.txt
"""
from __future__ import annotations

import argparse

import torch

from bad_gaussians.bad_utils import TrajectoryIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="KITTI trajectory file")
    parser.add_argument("--output", type=str, help="TUM trajectory file")
    args = parser.parse_args()

    poses = TrajectoryIO.load_kitti_trajectory(args.input)
    timestamps = torch.arange(0, len(poses))
    TrajectoryIO.write_tum_trajectory(args.output, timestamps, poses)
