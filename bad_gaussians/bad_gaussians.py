"""
BAD-Gaussians model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch import Tensor

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.model_components import renderers
from nerfstudio.utils import colormaps

from bad_gaussians.bad_camera_optimizer import (
    BadCameraOptimizer,
    BadCameraOptimizerConfig,
    TrajSamplingMode,
)
from bad_gaussians.bad_losses import EdgeAwareVariationLoss


@dataclass
class BadGaussiansModelConfig(SplatfactoModelConfig):
    """BAD-Gaussians Model config"""

    _target: Type = field(default_factory=lambda: BadGaussiansModel)
    """The target class to be instantiated."""

    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    Refs:
    1. https://github.com/nerfstudio-project/gsplat/pull/117
    2. https://github.com/nerfstudio-project/nerfstudio/pull/2888
    3. Yu, Zehao, et al. "Mip-Splatting: Alias-free 3D Gaussian Splatting." arXiv preprint arXiv:2311.16493 (2023).
    """

    camera_optimizer: BadCameraOptimizerConfig = field(default_factory=BadCameraOptimizerConfig)
    """Config of the camera optimizer to use"""

    cull_alpha_thresh: float = 0.005
    """Threshold for alpha to cull gaussians. Default: 0.1 in splatfacto, 0.005 in splatfacto-big."""

    densify_grad_thresh: float = 4e-4
    """[IMPORTANT] Threshold for gradient to densify gaussians. Default: 4e-4. Tune it smaller with complex scenes."""

    continue_cull_post_densification: bool = False
    """Whether to continue culling after densification. Default: True in splatfacto, False in splatfacto-big."""

    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled.
    Default: 250. Use 3000 with high resolution images (e.g. higher than 1920x1080).
    """

    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number. Default: 0. Use 2 with high resolution images."""

    tv_loss_lambda: Optional[float] = None
    """weight of total variation loss"""


class BadGaussiansModel(SplatfactoModel):
    """BAD-Gaussians Model

    Args:
        config: configuration to instantiate model
    """

    config: BadGaussiansModelConfig
    camera_optimizer: BadCameraOptimizer

    def __init__(self, config: BadGaussiansModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        # Scale densify_grad_thresh by the number of virtual views
        self.config.densify_grad_thresh /= self.config.camera_optimizer.num_virtual_views
        # (Experimental) Total variation loss
        self.tv_loss = EdgeAwareVariationLoss(in1_nc=3)

    def populate_modules(self) -> None:
        super().populate_modules()
        self.camera_optimizer: BadCameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    def forward(
            self,
            camera: Cameras,
            mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(camera, mode)

    def get_outputs(
            self, camera: Cameras,
            mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs.

        Args:
            camera: Input camera. This camera should have all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        is_training = self.training and torch.is_grad_enabled()

        # BAD-Gaussians: get virtual cameras
        virtual_cameras = self.camera_optimizer.apply_to_camera(camera, mode)

        if is_training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)
        if self.crop_box is not None and not is_training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None

        camera_downscale = self._get_downscale_factor()

        for cam in virtual_cameras:
            cam.rescale_output_resolution(1 / camera_downscale)

        # BAD-Gaussians: render virtual views
        virtual_views_rgb = []
        virtual_views_alpha = []
        for cam in virtual_cameras:
            # shift the camera to center of scene looking at center
            R = cam.camera_to_worlds[0, :3, :3]  # 3 x 3
            T = cam.camera_to_worlds[0, :3, 3:4]  # 3 x 1
            # flip the z axis to align with gsplat conventions
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
            R = R @ R_edit
            # analytic matrix inverse to get world2camera matrix
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv
            # update last_size
            W, H = int(cam.width.item()), int(cam.height.item())
            self.last_size = (H, W)

            if crop_ids is not None:
                opacities_crop = self.opacities[crop_ids]
                means_crop = self.means[crop_ids]
                features_dc_crop = self.features_dc[crop_ids]
                features_rest_crop = self.features_rest[crop_ids]
                scales_crop = self.scales[crop_ids]
                quats_crop = self.quats[crop_ids]
            else:
                opacities_crop = self.opacities
                means_crop = self.means
                features_dc_crop = self.features_dc
                features_rest_crop = self.features_rest
                scales_crop = self.scales
                quats_crop = self.quats

            colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
            BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
            self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                None,  # Deprecated projmat
                cam.fx.item(),
                cam.fy.item(),
                cam.cx.item(),
                cam.cy.item(),
                H,
                W,
                BLOCK_WIDTH,
            )  # type: ignore

            # rescale the camera back to original dimensions before returning
            cam.rescale_output_resolution(camera_downscale)

            if (self.radii).sum() == 0:
                rgb = background.repeat(H, W, 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)

                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

            # Important to allow xys grads to populate properly
            if is_training:
                self.xys.retain_grad()

            if self.config.sh_degree > 0:
                viewdirs = means_crop.detach() - cam.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
                viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
                rgbs = spherical_harmonics(n, viewdirs, colors_crop)
                rgbs = torch.clamp(rgbs + 0.5, min=0.0) # type: ignore
            else:
                rgbs = torch.sigmoid(colors_crop[:, 0, :])

            # rescale the camera back to original dimensions
            # cam.rescale_output_resolution(camera_downscale)
            assert (num_tiles_hit > 0).any()  # type: ignore

            # apply the compensation of screen space blurring to gaussians
            if self.config.rasterize_mode == "antialiased":
                alphas = torch.sigmoid(opacities_crop) * comp[:, None]
            elif self.config.rasterize_mode == "classic":
                alphas = torch.sigmoid(opacities_crop)
            rgb, alpha = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                rgbs,
                alphas,
                H,
                W,
                BLOCK_WIDTH,
                background=background,
                return_alpha=True,
            )  # type: ignore
            alpha = alpha[..., None]
            rgb = torch.clamp(rgb, max=1.0)  # type: ignore
            virtual_views_rgb.append(rgb)
            virtual_views_alpha.append(alpha)
        depth_im = None
        rgb = torch.stack(virtual_views_rgb, dim=0).mean(dim=0)
        alpha = torch.stack(virtual_views_alpha, dim=0).mean(dim=0)

        # eval
        if not is_training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": background}  # type: ignore

    @torch.no_grad()
    def get_outputs_for_camera(
            self,
            camera: Cameras,
            obb_box: Optional[OrientedBox] = None,
            mode: TrajSamplingMode = "mid",
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        # BAD-Gaussians: camera.to(device) will drop metadata
        metadata = camera.metadata
        camera = camera.to(self.device)
        camera.metadata = metadata
        outs = self.get_outputs(camera, mode=mode)
        return outs  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Add total variation loss
        rgb = outputs["rgb"].permute(2, 0, 1).unsqueeze(0)  # H, W, 3 to 1, 3, H, W
        if self.config.tv_loss_lambda is not None:
            loss_dict["tv_loss"] = self.tv_loss(rgb) * self.config.tv_loss_lambda
        # Add loss from camera optimizer
        self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict
