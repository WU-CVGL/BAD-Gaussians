"""
Full image datamanager for image restoration.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Type, Union, cast, Tuple, Dict

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import variable_res_collate
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.rich_utils import CONSOLE

from bad_gaussians.image_restoration_dataloader import ImageRestorationFixedIndicesEvalDataloader, ImageRestorationRandIndicesEvalDataloader


@dataclass
class ImageRestorationFullImageDataManagerConfig(FullImageDatamanagerConfig):
    """Datamanager for image restoration"""

    _target: Type = field(default_factory=lambda: ImageRestorationFullImageDataManager)
    """Target class to instantiate."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""


class ImageRestorationFullImageDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Data manager implementation for image restoration
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ImageRestorationFullImageDataManagerConfig

    def __init__(
            self,
            config: ImageRestorationFullImageDataManagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break

        self._fixed_indices_eval_dataloader = ImageRestorationFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = ImageRestorationRandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    @property
    def fixed_indices_eval_dataloader(self):
        """Returns the fixed indices eval dataloader"""
        return self._fixed_indices_eval_dataloader

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch. Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        # BAD-Gaussians: pass camera index to BadNerfCameraOptimizer
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")
