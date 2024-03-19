"""
Image restoration trainer.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Literal, Type
from typing_extensions import assert_never

import torch
from dataclasses import dataclass, field
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter

from bad_gaussians.image_restoration_pipeline import ImageRestorationPipeline, ImageRestorationPipelineConfig
from bad_gaussians.bad_viewer import BadViewer


@dataclass
class ImageRestorationTrainerConfig(TrainerConfig):
    """Configuration for image restoration training"""
    _target: Type = field(default_factory=lambda: ImageRestorationTrainer)
    """The target class to be instantiated."""

    pipeline: ImageRestorationPipelineConfig = field(default_factory=ImageRestorationPipelineConfig)
    """Image restoration pipeline configuration"""


class ImageRestorationTrainer(Trainer):
    """Image restoration Trainer class"""
    config: ImageRestorationTrainerConfig
    pipeline: ImageRestorationPipeline

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Set up the trainer.

        Args:
            test_mode: The test mode to use.
        """
        # BAD-Gaussianss: Overriding original setup since we want to use our BadNerfViewer
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            assert_never(self.config.vis)
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = BadViewer(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

        # BAD-Gaussianss: disable eval if no eval images
        if self.pipeline.datamanager.eval_dataset.cameras is None:
            self.config.steps_per_eval_all_images = int(9e9)
            self.config.steps_per_eval_batch = int(9e9)
            self.config.steps_per_eval_image = int(9e9)

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.
        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)
        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            # BAD-Gaussianss: pass output_path to save rendered images
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step, output_path=self.base_dir)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
