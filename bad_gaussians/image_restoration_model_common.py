from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from nerfstudio.models.base_model import Model
from nerfstudio.utils import colormaps


@torch.no_grad()
def get_restoration_eval_image_metrics_and_images(
        model: Model,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
    """Parse the evaluation outputs.
    Args:
        batch: Batch of data.
        outputs: Outputs of the model.

    Returns:
        A dictionary of metrics.
    """
    gt = batch["image"][:, :, :3].to(model.device)
    degraded = batch["degraded"][:, :, :3].to(model.device)
    rgb = outputs["rgb"]
    if "accumulation" in outputs:
        accumulation = outputs["accumulation"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        combined_acc = torch.cat([acc], dim=1)
    else:
        accumulation = None

    depth = colormaps.apply_depth_colormap(
        outputs["depth"],
        accumulation=accumulation,
    )
    combined_rgb = torch.cat([degraded, rgb, gt], dim=1)
    combined_depth = torch.cat([depth], dim=1)

    # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
    gt = torch.moveaxis(gt, -1, 0)[None, ...]
    rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

    psnr = model.psnr(gt, rgb)
    ssim = model.ssim(gt, rgb)
    lpips = model.lpips(gt, rgb)

    # all of these metrics will be logged as scalars
    metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
    images_dict = {"img": combined_rgb, "depth": combined_depth}
    if "accumulation" in outputs:
        images_dict["accumulation"] = combined_acc

    return metrics_dict, images_dict
