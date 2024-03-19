"""
Export poses from colmap images.txt
"""

import argparse
from pathlib import Path

import pypose as pp
import torch

from bad_gaussians.bad_utils import TrajectoryIO
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

DEVICE = 'cpu'


def main():
    parser = argparse.ArgumentParser(description="Export poses from colmap images.txt.")
    parser.add_argument("--input", type=str, required=True, help="Path to colmap files. E.g. ./sparse/0")
    parser.add_argument("--output", type=str, required=True, help="Path to the output TUM trajectory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"File already exists: {output_path}")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    print(f"Exporting poses from {input_path} to {output_path}")

    dataparser = ColmapDataParserConfig(data=input_path, colmap_path=".").setup()
    frames = dataparser._get_all_images_and_cameras(input_path)["frames"]

    names = [frame["file_path"] for frame in frames]
    poses = [frame["transform_matrix"] for frame in frames]
    poses = [x for _, x in sorted(zip(names, poses))]

    poses = pp.mat2SE3(torch.tensor(poses))
    num_cameras = poses.shape[0]
    timestamps = [float(x) for x in range(num_cameras)]
    TrajectoryIO.write_tum_trajectory(output_path, torch.tensor(timestamps), poses)


main()
