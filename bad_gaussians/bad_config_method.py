"""
BAD-Gaussians configs.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from bad_gaussians.bad_camera_optimizer import BadCameraOptimizerConfig
from bad_gaussians.image_restoration_dataparser import ImageRestorationDataParserConfig
from bad_gaussians.image_restoration_full_image_datamanager import ImageRestorationFullImageDataManagerConfig
from bad_gaussians.image_restoration_trainer import ImageRestorationTrainerConfig
from bad_gaussians.bad_gaussians import BadGaussiansModelConfig
from bad_gaussians.image_restoration_pipeline import ImageRestorationPipelineConfig


bad_gaussians = MethodSpecification(
    config=ImageRestorationTrainerConfig(
        method_name="bad-gaussians",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=500,
        max_num_iterations=30001,
        mixed_precision=False,
        use_grad_scaler=False,
        gradient_accumulation_steps={"camera_opt": 25},
        pipeline=ImageRestorationPipelineConfig(
            eval_render_start_end=True,
            eval_render_estimated=True,
            datamanager=ImageRestorationFullImageDataManagerConfig(
                cache_images="gpu",  # reduce CPU usage, caused by pin_memory()?
                dataparser=ImageRestorationDataParserConfig(
                    load_3D_points=True,
                    eval_mode="interval",
                    eval_interval=8,
                ),
            ),
            model=BadGaussiansModelConfig(
                camera_optimizer=BadCameraOptimizerConfig(mode="linear", num_virtual_views=10),
                use_scale_regularization=True,
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    max_steps=30000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of BAD-Gaussians",
)
