"""Data parser for Blender dataset compatible with BAD-Gaussians"""

import json
import subprocess
from pathlib import Path
from typing import List, Literal, Dict

import imageio
import numpy as np
from rich.console import Console

from datasets.colmap_utils import auto_orient_and_center_poses

CONSOLE = Console(width=120)


class BlenderParser:
    """Parser for Blender dataset, structured similarly to ColmapDataParser."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        factor: int = 1,
        normalize: bool = False,
        scale_factor: float = 1.0,
        orientation_method: Literal["pca", "up", "vertical", "none"] = "up",
        center_method: Literal["poses", "focus", "none"] = "poses",
        auto_scale_poses: bool = True,
        test_every: int = 8,
        downscale_rounding_mode: Literal["floor", "round", "ceil"] = "floor",
    ):
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.split = split
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.scale_factor = scale_factor
        self.downscale_rounding_mode = downscale_rounding_mode
        self.auto_scale_poses = auto_scale_poses

        # Load metadata from JSON
        json_path = self.data_dir / f"transforms_{self.split}.json"
        with open(json_path, 'r') as f:
            meta = json.load(f)

        # Collect image paths and poses from metadata
        image_paths = []
        poses = []
        blur_folder = self.data_dir / self.split / "blur_data"
        for frame in meta['frames']:
            file_path = Path(frame['file_path'])
            # print(f"blur_folder: {blur_folder}")
            # print(f"file_path: {file_path}")
            # image_name = file_path.name
        
            # The file_path is some how formatted as f"{split}\\_r0{image_name}"
            # extract the image name from the file_path
            image_name = file_path.name.split("\\")[-1][3:]
            img_path = blur_folder / image_name
            if not img_path.exists():
                # print(f"Image not found: {img_path}")
                continue  # Skip missing images
            image_paths.append(img_path)
            pose = np.array(frame['transform_matrix'], dtype=np.float32)
            poses.append(pose)

        # Handle downscaling if factor >1
        if self.factor > 1:
            original_image_dir = blur_folder
            downscaled_dir = original_image_dir.parent / f"blur_data_{self.factor}"
            if not downscaled_dir.exists():
                self._downscale_images(original_image_dir, downscaled_dir)
            image_paths = [downscaled_dir / img.name for img in image_paths]

        # Read image dimensions and setup camera parameters
        try:
            assert len(image_paths) > 0, f"No images found in {blur_folder}"
        except AssertionError as e:
            print(f"len(meta['frames']): {len(meta['frames'])}")
        img = imageio.v2.imread(image_paths[0])
        original_h, original_w = img.shape[:2]
        h, w = original_h // self.factor, original_w // self.factor
        self.imsize_dict = {0: (w, h)}

        # Focal length and intrinsic matrix
        camera_angle_x = meta.get('camera_angle_x', 0.7854)  # Default to ~45 degrees if missing
        focal = 0.5 * original_w / np.tan(0.5 * camera_angle_x) / self.factor
        cx, cy = (original_w / 2) / self.factor, (original_h / 2) / self.factor
        self.Ks_dict = {0: np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float32)}
        self.params_dict = {0: np.array([], dtype=np.float32)}  # No distortion

        # Convert poses to OpenCV convention (Y down, Z forward)
        flip_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        poses = np.array([pose @ flip_mat.T for pose in poses])

        # Convert to 4x4 camtoworld matrices
        camtoworlds = np.eye(4, dtype=np.float32)[None].repeat(len(poses), 0)
        camtoworlds[:, :3, :4] = poses[:, :3, :4]

        # Orient and center poses
        camtoworlds, transform_matrix = auto_orient_and_center_poses(
            camtoworlds, orientation_method, center_method
        )

        # Scale poses
        if self.auto_scale_poses:
            max_dist = np.max(np.linalg.norm(camtoworlds[:, :3, 3], axis=1))
            scale_factor = 1.0 / max_dist
        else:
            scale_factor = 1.0
        scale_factor *= self.scale_factor
        camtoworlds[:, :3, 3] *= scale_factor

        # Store parsed data
        self.image_names = [img.name for img in image_paths]
        self.image_paths = [str(img) for img in image_paths]
        self.camtoworlds = camtoworlds
        self.camera_ids = [0] * len(image_paths)
        self.transform = transform_matrix

        # Scene scale calculation
        camera_locs = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locs, axis=0)
        self.scene_scale = np.max(np.linalg.norm(camera_locs - scene_center, axis=1))

        # Placeholders for compatibility with ColmapDataParser
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.points_err = np.zeros(0, dtype=np.float32)
        self.points_rgb = np.zeros((0, 3), dtype=np.uint8)
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}

    def _downscale_images(self, src_dir: Path, dst_dir: Path):
        """Downscale images using ffmpeg."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_dir.glob("*.png"):
            cmd = [
                'ffmpeg', '-y', '-i', str(img_path),
                '-vf', f'scale=iw/{self.factor}:ih/{self.factor}',
                '-q:v', '2', str(dst_dir / img_path.name)
            ]
            subprocess.run(cmd, check=True)
        CONSOLE.log(f"Downscaled images saved to {dst_dir}")
