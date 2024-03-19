'''Viewer of BAD-Gaussianss'''
import numpy as np
import torch
import viser.transforms as vtf

from nerfstudio.viewer.viewer import Viewer, VISER_NERFSTUDIO_SCALE_RATIO

from bad_gaussians.bad_camera_optimizer import BadCameraOptimizer


class BadViewer(Viewer):
    # BAD-Gaussianss: Overriding original update_camera_poses because BadNerfCameraOptimizer returns LieTensor
    def update_camera_poses(self):
        # TODO this fn accounts for like ~5% of total train time
        # Update the train camera locations based on optimization
        assert self.camera_handles is not None
        if hasattr(self.pipeline.datamanager, "train_camera_optimizer"):
            camera_optimizer = self.pipeline.datamanager.train_camera_optimizer
        elif hasattr(self.pipeline.model, "camera_optimizer"):
            camera_optimizer = self.pipeline.model.camera_optimizer
        else:
            return
        idxs = list(self.camera_handles.keys())
        with torch.no_grad():
            assert isinstance(camera_optimizer, BadCameraOptimizer)
            c2ws_delta = camera_optimizer(torch.tensor(idxs, device=camera_optimizer.device))
        for i, key in enumerate(idxs):
            # both are numpy arrays
            c2w_orig = self.original_c2w[key]
            c2w_delta = c2ws_delta[i, ...]
            c2w = c2w_orig @ c2w_delta.matrix().cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
            R = R @ vtf.SO3.from_x_radians(np.pi)
            self.camera_handles[key].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
            self.camera_handles[key].wxyz = R.wxyz
