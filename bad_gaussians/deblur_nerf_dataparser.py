"""
Data parser for Deblur-NeRF COLMAP datasets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import cv2
import torch

from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser, ColmapDataParserConfig

from bad_gaussians.image_restoration_dataparser import _find_files


@dataclass
class DeblurNerfDataParserConfig(ColmapDataParserConfig):
    """Deblur-NeRF dataset config"""

    _target: Type = field(default_factory=lambda: DeblurNerfDataParser)
    """target class to instantiate"""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    All uses all the images for any split.
    """
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    downscale_factor: Optional[int] = 1
    """The downscale factor for the images. Default: 1."""
    poses_bounds_path: Path = Path("poses_bounds.npy")
    """Path to the poses bounds file relative to the data path."""
    colmap_path: Path = Path("sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    drop_distortion: bool = False
    """Whether to drop the camera distortion parameters. Default: False."""
    scale_factor: float = 0.25
    """[IMPORTANT] How much to scale the camera origins by.
    Default: 0.25 suggested for LLFF datasets with COLMAP.
    """


@dataclass
class DeblurNerfDataParser(ColmapDataParser):
    """Deblur-NeRF COLMAP dataset parser"""

    config: DeblurNerfDataParserConfig
    _downscale_factor: Optional[int] = None

    def _get_all_images_and_cameras(self, recon_dir: Path):
        out = super()._get_all_images_and_cameras(recon_dir)
        out["frames"] = sorted(out["frames"], key=lambda x: x["file_path"])
        return out

    def _check_outputs(self, outputs):
        """
        Check if the colmap outputs are estimated on downscaled data. If so, correct the camera parameters.
        """
        # load the first image to get the image size
        image = cv2.imread(str(self.config.data / self.config.images_path / outputs.image_filenames[0]))
        # get the image size
        h, w = image.shape[:2]
        # check if the cx and cy are in the correct range
        cx = outputs.cameras.cx[0]
        cy = outputs.cameras.cy[0]
        ideal_cx = torch.tensor(w / 2)
        ideal_cy = torch.tensor(h / 2)
        if not torch.allclose(cx, ideal_cx, rtol=0.3):
            x_scale = cx / ideal_cx
            print(f"[WARN] cx is not at the center of the image, correcting... cx scale: {x_scale}")
            if x_scale < 1:
                outputs.cameras.fx *= round(1 / x_scale.item())
                outputs.cameras.cx *= round(1 / x_scale.item())
                outputs.cameras.width *= round(1 / x_scale.item())
            else:
                outputs.cameras.fx /= round(x_scale.item())
                outputs.cameras.cx /= round(x_scale.item())
                outputs.cameras.width //= round(x_scale.item())

        if not torch.allclose(cy, ideal_cy, rtol=0.3):
            y_scale = cy / ideal_cy
            print(f"[WARN] cy is not at the center of the image, correcting... cy scale: {y_scale}")
            if y_scale < 1:
                outputs.cameras.fy *= round(1 / y_scale.item())
                outputs.cameras.cy *= round(1 / y_scale.item())
                outputs.cameras.height *= round(1 / y_scale.item())
            else:
                outputs.cameras.fy /= round(y_scale.item())
                outputs.cameras.cy /= round(y_scale.item())
                outputs.cameras.height //= round(y_scale.item())

        return outputs

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.eval_mode == "interval":
            # find the file named `hold=n` , n is the eval_interval to be recognized
            hold_file = [f for f in os.listdir(self.config.data) if f.startswith('hold=')]
            if len(hold_file) == 0:
                print(f"[INFO] defaulting hold={self.config.eval_interval}")
            else:
                self.config.eval_interval = int(hold_file[0].split('=')[-1])

        gt_folder_path = self.config.data / "images_test"
        if gt_folder_path.exists():
            outputs = super()._generate_dataparser_outputs("train")
            if split != "train":
                gt_image_filenames = _find_files(gt_folder_path, exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])
                num_gt_images = len(gt_image_filenames)
                print(f"[INFO] Found {num_gt_images} ground truth sharp images.")
                # number of GT sharp testing images should be equal to the number of degraded training images
                assert num_gt_images == len(outputs.image_filenames)
                outputs.image_filenames = gt_image_filenames
        else:
            print("[INFO] No ground truth sharp images found.")
            outputs = super()._generate_dataparser_outputs(split)

        if self.config.drop_distortion:
            for camera in outputs.cameras:
                camera.distortion_params = None
        # outputs.image_filenames = [f.with_suffix('.png') for f in outputs.image_filenames]
        outputs = self._check_outputs(outputs)

        return outputs
