# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import subprocess
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Literal, Optional

import cv2
import os
import numpy as np
import datasets.colmap_parsing_utils as colmap_utils
from datasets.colmap_utils import (
    parse_colmap_camera_params,
    auto_orient_and_center_poses,
)
from rich.console import Console

MAX_AUTO_RESOLUTION = 1600
CONSOLE = Console(width=120)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


def run_command(cmd: str, verbose=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        CONSOLE.rule(
            "[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ",
            style="red",
        )
        CONSOLE.print(f"[bold red]Error running command: {cmd}")
        CONSOLE.rule(style="red")
        CONSOLE.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


class ColmapParser:
    """an adapted version of nerfstudio ColmapDataParser"""

    def __init__(
        self,
        data_dir: str,
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

        self.data_dir = Path(data_dir)
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.scale_factor = scale_factor
        self.downscale_rounding_mode = downscale_rounding_mode

        colmap_dir = data_dir / "sparse/0/"
        if not os.path.exists(colmap_dir):
            colmap_dir = data_dir / "sparse"
        assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

        if (colmap_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(colmap_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(colmap_dir / "images.txt")
        elif (colmap_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(colmap_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(colmap_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {colmap_dir}")

        # read in GT poses if exists
        if (colmap_dir / "images_gt.bin").exists():
            im_id_to_image_gt = colmap_utils.read_images_binary(colmap_dir / "images_gt.bin")
        elif (colmap_dir / "images_gt.txt").exists():
            im_id_to_image_gt = colmap_utils.read_images_text(colmap_dir / "images_gt.txt")
        else:
            im_id_to_image_gt = None

        cameras = {}
        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Parse frames
        # we want to sort all images based on im_id
        ordered_im_id = sorted(im_id_to_image.keys())

        # Extract extrinsic matrices in world-to-camera format.
        # imdata = manager.images
        w2c_mats = []
        w2c_mats_gt = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for im_id in ordered_im_id:
            im = im_id_to_image[im_id]
            rot = colmap_utils.qvec2rotmat(im.qvec)
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            if im_id_to_image_gt is not None:
                im_gt = im_id_to_image_gt[im_id + 1]  # NOTE: +1 hacks for now
                rot_gt = colmap_utils.qvec2rotmat(im_gt.qvec)
                trans_gt = im_gt.tvec.reshape(3, 1)
                w2c_gt = np.concatenate([np.concatenate([rot_gt, trans_gt], 1), bottom], axis=0)
                w2c_mats_gt.append(w2c_gt)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras[camera_id]
            # fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            fx, fy, cx, cy = cam["fl_x"], cam["fl_y"], cam["cx"], cam["cy"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam["model"]
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam["k1"]], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam["k1"], cam["k2"], 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam["k1"], cam["k2"], cam["p1"], cam["p2"]], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam["k1"], cam["k2"], cam["k3"], cam["k4"]], dtype=np.float32)
                camtype = "fisheye"
            assert camtype == "perspective", f"Only support perspective camera model, got {type_}"

            params_dict[camera_id] = params

            # image size
            # imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            imsize_dict[camera_id] = (cam["w"] // factor, cam["h"] // factor)
        print(f"[Parser] {len(im_id_to_image)} images, taken by {len(set(camera_ids))} cameras.")

        if len(im_id_to_image) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ in ["PINHOLE", "SIMPLE_PINHOLE", 0, 1]):
            print(f"Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [im_id_to_image[k].name for k in ordered_im_id]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
            self.downscale_factor = factor
            self._downscale_factor = None
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        if not os.path.exists(image_dir):
            self._setup_downscale_factor(
                Path(colmap_image_dir), [Path(colmap_image_dir) / f for f in colmap_files], [], []
            )
        images_test_dir = os.path.join(data_dir, "images_test")
        images_test_downscale_dir = os.path.join(data_dir, "images_test" + image_dir_suffix)
        if os.path.exists(images_test_dir) and not os.path.exists(images_test_downscale_dir):
            test_files = sorted(_get_rel_paths(images_test_dir))
            self._setup_downscale_factor(Path(images_test_dir), [Path(images_test_dir) / f for f in test_files], [], [])
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # construct masks path
        mask_dir = os.path.join(data_dir, "masks")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in colmap_files]

        # get masks path
        if os.path.exists(mask_dir):
            mask_paths = [os.path.join(mask_dir, f) for f in image_names]
            self.mask_paths = mask_paths
            print("Got masks path!")

        camtoworlds, transform_matrix = auto_orient_and_center_poses(
            camtoworlds, method=orientation_method, center_method=center_method
        )

        scale_factor = 1.0  # NOTE:
        if auto_scale_poses:
            scale_factor /= float(np.max(np.abs(camtoworlds[:, :3, 3])))
        scale_factor *= self.scale_factor
        camtoworlds[:, :3, 3] *= scale_factor
        N = camtoworlds.shape[0]
        bottoms = np.repeat(bottom[np.newaxis, :], N, axis=0)
        camtoworlds = np.concatenate((camtoworlds, bottoms), axis=1)

        if im_id_to_image_gt is not None:
            w2c_mats_gt = np.stack(w2c_mats_gt, axis=0)
            camtoworlds_gt = np.linalg.inv(w2c_mats_gt)
            camtoworlds_gt = camtoworlds_gt[inds]
            camtoworlds_gt, _ = auto_orient_and_center_poses(
                camtoworlds_gt, method=orientation_method, center_method=center_method
            )
            scale_factor_gt = 1.0
            if auto_scale_poses:
                scale_factor_gt /= float(np.max(np.abs(camtoworlds_gt[:, :3, 3])))
            scale_factor_gt *= self.scale_factor
            camtoworlds_gt[:, :3, 3] *= scale_factor_gt
            camtoworlds_gt = np.concatenate((camtoworlds_gt, bottoms), axis=1)
            self.camtoworlds_gt = camtoworlds_gt

        # load in 3D points
        if (colmap_dir / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_dir / "points3D.bin")
        elif (colmap_dir / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_dir / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_dir}")
        points = np.array([p.xyz for p in colmap_points.values()], dtype=np.float32)
        points = (
            np.concatenate(
                (
                    points,
                    np.ones_like(points[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points *= scale_factor

        points_rgb = np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8)
        points_err = np.array([p.error for p in colmap_points.values()], dtype=np.float32)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.transform = transform_matrix  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert camera_id in self.params_dict, f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(K, params, (width, height), 0)
            mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist, (width, height), cv2.CV_32FC1)
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # BAD-Gaussians: Check if the colmap outputs are estimated on downscaled data.
        # If so, correct the camera parameters. E.g., ball sequence in Deblur-NeRF dataset.
        image = cv2.imread(image_paths[0])  # load the first image to get the image size
        h, w = image.shape[:2]
        # check if the cx and cy are in the correct range
        ideal_cx = w / 2.0
        ideal_cy = h / 2.0

        def find_int_scale_factor(scaled):
            if scaled < 1:
                scale = round(1 / scaled)
            else:
                scale = 1 / round(scaled)
            return scale

        cx_0 = list(Ks_dict.values())[0][0, 2]
        cy_0 = list(Ks_dict.values())[0][1, 2]
        if not abs(cx_0 - ideal_cx) / ideal_cx < 0.3:
            x_scale = cx_0 / ideal_cx
            print(f"[WARN] cx is away from the center of the image, correcting... cx scale: {x_scale}")
            scale = find_int_scale_factor(x_scale)
            for cam_id in Ks_dict.keys():
                Ks_dict[cam_id][0, 0] *= scale  # fx
                Ks_dict[cam_id][0, 2] *= scale  # cx
                imsize_dict[cam_id] = (
                    imsize_dict[cam_id][0] * scale,
                    imsize_dict[cam_id][1],
                )
        if not abs(cy_0 - ideal_cy) / ideal_cy < 0.3:
            y_scale = cy_0 / ideal_cy
            print(f"[WARN] cy is away from the center of the image, correcting... cy scale: {y_scale}")
            scale = find_int_scale_factor(y_scale)
            for cam_id in Ks_dict.keys():
                Ks_dict[cam_id][1, 1] *= scale  # fy
                Ks_dict[cam_id][1, 2] *= scale  # cy
                imsize_dict[cam_id] = (
                    imsize_dict[cam_id][0],
                    imsize_dict[cam_id][1] * scale,
                )

    def _downscale_images(
        self,
        paths,
        get_fname,
        downscale_factor: int,
        downscale_rounding_mode: str = "floor",
        nearest_neighbor: bool = False,
    ):
        def calculate_scaled_size(original_width, original_height, downscale_factor, mode="floor"):
            if mode == "floor":
                return math.floor(original_width / downscale_factor), math.floor(original_height / downscale_factor)
            elif mode == "round":
                return round(original_width / downscale_factor), round(original_height / downscale_factor)
            elif mode == "ceil":
                return math.ceil(original_width / downscale_factor), math.ceil(original_height / downscale_factor)
            else:
                raise ValueError("Invalid mode. Choose from 'floor', 'round', or 'ceil'.")

        with status(msg="[bold yellow]Downscaling images...", spinner="growVertical"):
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            # Using %05d ffmpeg commands appears to be unreliable (skips images).
            for path in paths:
                # Compute image-wise rescaled width/height.
                img = Image.open(path)
                w, h = img.size
                w_scaled, h_scaled = calculate_scaled_size(w, h, downscale_factor, downscale_rounding_mode)
                # Downscale images using ffmpeg.
                nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
                path_out = get_fname(path)
                path_out.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg_cmd = [
                    f'ffmpeg -y -noautorotate -i "{path}" ',
                    f"-q:v 2 -vf scale={w_scaled}:{h_scaled}{nn_flag} ",
                    f'"{path_out}"',
                ]
                ffmpeg_cmd = " ".join(ffmpeg_cmd)
                run_command(ffmpeg_cmd)

        CONSOLE.log("[bold green]:tada: Done downscaling images.")

    def _setup_downscale_factor(
        self,
        images_parent_dir: Path,
        image_filenames: List[Path],
        mask_filenames: List[Path],
        depth_filenames: List[Path],
    ):
        """
        Setup the downscale factor for the dataset. This is used to downscale the images and cameras.
        """

        def get_fname(parent: Path, filepath: Path) -> Path:
            """Returns transformed file name when downscale factor is applied"""
            rel_part = filepath.relative_to(parent)
            base_part = parent.parent / (str(parent.name) + f"_{self.downscale_factor}")
            return base_part / rel_part

        filepath = next(iter(image_filenames))
        if self._downscale_factor is None:
            if self.downscale_factor is None:
                test_img = Image.open(filepath)
                w, h = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    df += 1

                self._downscale_factor = 2**df
                CONSOLE.log(f"Using image downscale factor of {self._downscale_factor}")
            else:
                self._downscale_factor = self.downscale_factor
            if self._downscale_factor > 1 and not all(
                get_fname(images_parent_dir, fp).parent.exists() for fp in image_filenames
            ):
                # Downscaled images not found
                # Ask if user wants to downscale the images automatically here
                CONSOLE.log(
                    f"[bold red]Downscaled images do not exist for factor of {self._downscale_factor}.[/bold red]"
                )
                CONSOLE.log(
                    f"[green]Downscaling the images using '{self.downscale_rounding_mode}' rounding mode now.[/green]"
                )
                # Install the method
                self._downscale_images(
                    image_filenames,
                    partial(get_fname, images_parent_dir),
                    self._downscale_factor,
                    self.downscale_rounding_mode,
                    nearest_neighbor=False,
                )
                if len(mask_filenames) > 0:
                    raise NotImplementedError
                if len(depth_filenames) > 0:
                    raise NotImplementedError
            else:
                sys.exit(1)

        # Return transformed filenames
        if self._downscale_factor > 1:
            image_filenames = [get_fname(images_parent_dir, fp) for fp in image_filenames]
            if len(mask_filenames) > 0:
                raise NotImplementedError
            if len(depth_filenames) > 0:
                raise NotImplementedError
        assert isinstance(self._downscale_factor, int)
        self._downscale_factor = None
        return image_filenames, mask_filenames, depth_filenames, self._downscale_factor

if __name__ == "__main__":
    import argparse
    import imageio.v2 as imageio

    from datasets.colmap import Dataset


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = ColmapParser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
