#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.data.utils import (
    closest_point_2_lines,
    qvec2rotmat,
    run_colmap,
    run_ffmpeg,
    sharpness,
)

ROOT_DIR = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = ROOT_DIR / "scripts"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place."
    )

    parser.add_argument(
        "--video_in",
        default="",
        help="Run ffmpeg first to convert a provided video file into a set of images. Uses the video_fps parameter also.",
    )
    parser.add_argument("--video_fps", default=2)
    parser.add_argument(
        "--time_slice",
        default="",
        help="Time (in seconds) in the format t1,t2 within which the images should be generated from the video. E.g.: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video.",
    )
    parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
    parser.add_argument(
        "--colmap_matcher",
        default="sequential",
        choices=["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"],
        help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.",
    )
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
    parser.add_argument(
        "--colmap_camera_model",
        default="OPENCV",
        choices=[
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
            "OPENCV_FISHEYE",
        ],
        help="Camera model",
    )
    parser.add_argument(
        "--colmap_camera_params",
        default="",
        help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist",
    )
    parser.add_argument(
        "--colmap_fix_camera_params", action="store_true", help="Fix camera parameters to the ones given."
    )
    parser.add_argument("--images", default="images", help="Input path to the images.")
    parser.add_argument(
        "--text",
        default="colmap_text",
        help="Input path to the colmap text files (set automatically if --run_colmap is used).",
    )
    parser.add_argument(
        "--aabb_scale",
        default=32,
        choices=["1", "2", "4", "8", "16", "32", "64", "128"],
        help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128",
    )
    parser.add_argument("--skip_early", default=0, help="Skip this many images from the start.")
    parser.add_argument(
        "--keep_colmap_coords",
        action="store_true",
        help="Keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering).",
    )
    parser.add_argument("--out", default="transforms.json", help="Output JSON file path.")
    parser.add_argument("--vocab_path", default="", help="Vocabulary tree path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Do not ask for confirmation for overwriting existing images and COLMAP data.",
    )
    parser.add_argument(
        "--mask_categories",
        nargs="*",
        type=str,
        default=[],
        help="Object categories that should be masked out from the training images. See `scripts/category2id.json` for supported categories.",
    )
    args = parser.parse_args()
    return args


class ColmapProcessor:
    @staticmethod
    def parse_cameras(cameras_txt_path: Path) -> Dict[int, dict]:
        cameras = {}
        with open(cameras_txt_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                elements = line.strip().split()
                camera_id = int(elements[0])
                model = elements[1]

                camera = {
                    "w": float(elements[2]),
                    "h": float(elements[3]),
                    "fl_x": float(elements[4]),
                    "fl_y": float(elements[4]),
                    "cx": float(elements[2]) / 2,
                    "cy": float(elements[3]) / 2,
                    "k1": 0,
                    "k2": 0,
                    "k3": 0,
                    "k4": 0,
                    "p1": 0,
                    "p2": 0,
                    "is_fisheye": False,
                }

                parser = ColmapProcessor.CAMERA_PARSERS.get(model)
                if not parser:
                    raise ValueError(f"Unknown camera model: {model}")
                parser(elements, camera)

                camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
                camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
                camera["fovx"] = math.degrees(camera["camera_angle_x"])
                camera["fovy"] = math.degrees(camera["camera_angle_y"])

                cameras[camera_id] = camera
        return cameras

    @staticmethod
    def parse_images(images_txt_path: Path, image_dir: Path, skip_early: int) -> Tuple[List[dict], np.ndarray]:
        frames = []
        up = np.zeros(3)
        with open(images_txt_path, "r") as f:
            for i, line in enumerate(f):
                if line.startswith("#") or i < skip_early * 2 or i % 2 == 0:
                    continue

                elements = line.strip().split()
                frame = ColmapProcessor.process_image_line(elements, image_dir)
                frames.append(frame)
                up += frame["transform_matrix"][0:3, 1]
        return frames, up

    @staticmethod
    def process_image_line(elements: List[str], image_dir: Path) -> dict:
        image_path = image_dir / "_".join(elements[9:])
        sharpness_val = sharpness(str(image_path))

        qvec = np.array(list(map(float, elements[1:5])))
        tvec = np.array(list(map(float, elements[5:8])))
        camera_id = int(elements[8])

        rotation = qvec2rotmat(-qvec)
        translation = tvec.reshape(3, 1)
        transform = np.vstack([np.hstack([rotation, translation]), [0, 0, 0, 1]])
        c2w = np.linalg.inv(transform)

        return {
            "file_path": str(image_path.relative_to(image_dir.parent)),
            "sharpness": sharpness_val,
            "transform_matrix": c2w,
            "camera_id": camera_id,
        }

    # Camera model parsing functions
    @staticmethod
    def _parse_simple_pinhole(elements: List[str], camera: dict):
        camera.update(
            {
                "cx": float(elements[5]),
                "cy": float(elements[6]),
            }
        )

    @staticmethod
    def _parse_pinhole(elements: List[str], camera: dict):
        camera.update(
            {
                "fl_y": float(elements[5]),
                "cx": float(elements[6]),
                "cy": float(elements[7]),
            }
        )

    # Define other camera model parsers similarly...

    CAMERA_PARSERS = {
        "SIMPLE_PINHOLE": _parse_simple_pinhole,
        "PINHOLE": _parse_pinhole,
        # Add other camera models here...
    }


class SceneProcesser:
    @staticmethod
    def adjust_scene(frames: List[dict], keep_colmap_coords: bool, aabb_scale: int) -> None:
        if not keep_colmap_coords:
            SceneProcesser.reorient_scene(frames)
            SceneProcesser.center_and_scale_scene(frames, aabb_scale)
        else:
            raise NotImplementedError("Keeping COLMAP coordinates is not yet supported.")

    @staticmethod
    def reorient_scene(frames: List[dict]) -> None:
        up = sum(frame["transform_matrix"][0:3, 1] for frame in frames)
        up /= np.linalg.norm(up)
        R = rotmat(up, [0, 0, 1])
        for frame in frames:
            frame["transform_matrix"] = R @ frame["transform_matrix"]

    @staticmethod
    def center_and_scale_scene(frames: List[dict], aabb_scale: int) -> None:
        center = SceneProcesser.compute_scene_center(frames)
        for frame in frames:
            frame["transform_matrix"][0:3, 3] -= center

        avg_distance = np.mean([np.linalg.norm(f["transform_matrix"][0:3, 3]) for f in frames])
        scale_factor = 4.0 / avg_distance
        for frame in frames:
            frame["transform_matrix"][0:3, 3] *= scale_factor

    @staticmethod
    def compute_scene_center(frames: List[dict]) -> np.ndarray:
        total_weight = 0.0
        center = np.zeros(3)
        for f in frames:
            for g in frames:
                point, weight = closest_point_2_lines(
                    f["transform_matrix"][0:3, 3],
                    f["transform_matrix"][0:3, 2],
                    g["transform_matrix"][0:3, 3],
                    g["transform_matrix"][0:3, 2],
                )
                if weight > 1e-5:
                    center += point * weight
                    total_weight += weight
        return center / total_weight if total_weight > 0 else np.zeros(3)


def main():
    args = parse_args()

    if args.video_in:
        run_ffmpeg(args, SCRIPTS_DIR)

    if args.run_colmap:
        run_colmap(args, SCRIPTS_DIR)

    cameras = ColmapProcessor.parse_cameras(Path(args.text) / "cameras.txt")
    frames, up_vector = ColmapProcessor.parse_images(Path(args.text) / "images.txt", Path(args.images), args.skip_early)

    output_data = {"frames": frames, "aabb_scale": args.aabb_scale, **cameras.get(1, {})}

    if not args.keep_colmap_coords:
        SceneProcesser.adjust_scene(frames, args.keep_colmap_coords, args.aabb_scale)

    with open(args.out, "w") as f:
        json.dump(output_data, f, indent=2)

    if len(args.mask_categories) > 0:
        raise NotImplementedError("Masking categories is not yet supported.")
        # apply_masks(args, frames, args.mask_categories)


if __name__ == "__main__":
    main()
