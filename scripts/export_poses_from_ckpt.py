"""
Export optimized poses from a checkpoint.
"""
import argparse
from pathlib import Path

import pypose as pp
import torch
from typing_extensions import assert_never

from bad_gaussians.bad_utils import TrajectoryIO
from bad_gaussians.deblur_nerf_dataparser import DeblurNerfDataParserConfig
from bad_gaussians.spline_functor import linear_interpolation_mid, cubic_bspline_interpolation

# DEVICE = 'cuda:0'
DEVICE = 'cpu'

def main():
    parser = argparse.ArgumentParser(description="Export optimized poses from a checkpoint.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output TUM trajectory.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Exporting optimized poses from {ckpt_path} using data from {data_dir}")
    export_poses(ckpt_path, data_dir, output_path)


def export_poses(ckpt_path, data_dir, output_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    pose_adjustment = ckpt['pipeline']['_model.camera_optimizer.pose_adjustment']

    # parser = BadNerfColmapDataParserConfig(data=data_dir, colmap_path="sparse/0").setup()
    # parser = DeblurNerfDataParserConfig(data=data_dir, downscale_factor=1).setup()
    parser = DeblurNerfDataParserConfig(
        data=data_dir,
        downscale_factor=1,
        images_path="images_1",
        eval_mode="all",
    ).setup()
    parser_outputs = parser.get_dataparser_outputs(split="train")

    print(parser_outputs)

    print(parser_outputs.cameras.camera_to_worlds.shape)

    poses = pp.mat2SE3(parser_outputs.cameras.camera_to_worlds.to(DEVICE))

    num_cameras, num_ctrl_knots, _ = pose_adjustment.shape

    if num_ctrl_knots == 2:
        poses_delta = linear_interpolation_mid(pose_adjustment)
    elif num_ctrl_knots == 4:
        poses_delta = cubic_bspline_interpolation(
            pose_adjustment,
            torch.tensor([0.5], device=pose_adjustment.device)
        ).squeeze(1)
    else:
        assert_never(num_ctrl_knots)

    print(poses.shape)
    print(poses_delta.shape)
    poses_optimized = poses @ poses_delta
    # poses_optimized = poses

    timestamps = [float(x) for x in range(num_cameras)]

    if output_path.is_dir():
        output_path = output_path / "BAD-Gaussians.txt"

    if output_path.exists():
        raise FileExistsError(f"File already exists: {output_path}")

    TrajectoryIO.write_tum_trajectory(output_path, torch.tensor(timestamps), poses_optimized)

    print(f"Exported optimized poses to {output_path}")


main()
