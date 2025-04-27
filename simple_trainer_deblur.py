import json
import math
import os
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import List
from typing_extensions import assert_never

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from dataclasses import dataclass, field
from fused_ssim import fused_ssim
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfview.viewer import Viewer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from bad_gaussians.bad_camera_optimizer import BadCameraOptimizer, BadCameraOptimizerConfig
from datasets.blender_dataperser import BlenderParser
from datasets.colmap import Dataset
from datasets.colmap_dataparser import ColmapParser
from datasets.deblur_nerf import DeblurNerfDataset
from pose_viewer import PoseViewer
from simple_trainer import Config, Runner, create_splats_with_optimizers
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)
from utils import (
    AppearanceOptModule,
    CameraOptModuleSE3,
    set_random_seed,
)


@dataclass
class DeblurConfig(Config):
    # Path to the Mip-NeRF 360 dataset
    # data_dir: str = "data/360_v2/garden"
    # data_dir: str = "/datasets/bad-gaussian/data/bad-nerf-gtK-colmap-nvs/blurtanabata"
    data_dir: str = "/home/lzzhao/ws/BAD-Gaussians2/data/blurtanabata"
    # Downsample factor for the dataset
    data_factor: int = 1
    # How much to scale the camera origins by. 0.25 is suggested for LLFF scenes.
    scale_factor: float = 1.0
    # Directory to save results
    # result_dir: str = "results/garden"
    result_dir: str = "results/tanabata_fused-ssim"
    # Every N images there is a test image
    test_every: int = 8

    ########### Viewer ###############

    # Visualize cameras in the viewer
    visualize_cameras: bool = True

    ########### Training ###############

    # Number of training steps
    max_steps: int = 20_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 10_000, 15_000, 20_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 10_000, 15_000, 20_000, 30_000])

    # Use fused SSIM from Taming 3DGS (https://github.com/nerfstudio-project/gsplat/pull/396)
    fused_ssim = False

    # Whether to pin memory for DataLoader. Disable if you run out of memory.
    pin_memory: bool = False

    ########### Background ###############

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    ########### Motion Deblur ###############

    # BAD-Gaussians: Bundle adjusted deblur camera optimizer
    camera_optimizer: BadCameraOptimizerConfig = field(
        default_factory=lambda: BadCameraOptimizerConfig(
            mode="linear",
            num_virtual_views=10,
        )
    )

    ########### Camera Opt ###############

    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-3
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Learning rate decay rate of camera optimization
    pose_opt_lr_decay: float = 1e-3
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 1e-5
    # Pose gradient accumulation steps
    pose_gradient_accumulation_steps: int = 25

    ########### Novel View Eval ###############

    # Enable novel view synthesis evaluation during training
    nvs_eval_enable_during_training: bool = True

    ########### Novel View Eval Camera Opt ###############

    # Steps per image to evaluate the novel view synthesis during training
    nvs_steps: int = 200
    # Steps per image to evaluate the novel view synthesis in final evaluation
    nvs_steps_final: int = 1000
    # Novel view synthesis evaluation pose learning rate
    nvs_pose_lr: float = 1e-3
    # Novel view synthesis evaluation pose regularization
    nvs_pose_reg: float = 0.0
    # Novel view synthesis evaluation pose learning rate decay
    nvs_pose_lr_decay: float = 1e-2

    ########### Deblurring Eval ###############

    # Enable deblurring evaluation during training
    deblur_eval_enable_during_training: bool = True
    # Enable pose optimization in deblurring evaluation, helpful when GT sharp images are not captured exactly from the mid of the trajectory.
    deblur_eval_enable_pose_opt: bool = False

    ########### Regularizations ###############

    # If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians.
    enable_phys_scale_reg: bool = False
    # threshold of ratio of gaussian max to min scale before applying regularization loss from the PhysGaussian paper
    max_gauss_ratio: float = 10.0

    # Regularizations from 3dgs-mcmc
    enable_mcmc_opacity_reg: bool = True
    enable_mcmc_scale_reg: bool = True
    opacity_reg: float = 0.01
    scale_reg: float = 0.01

    ######################################

    # Avoid multiple initialization
    bad_gaussians_post_init_complete: bool = False

    def __post_init__(self):
        if not self.bad_gaussians_post_init_complete:
            self.bad_gaussians_post_init_complete = True
            timestr = time.strftime("%Y%m%d-%H%M%S")
            self.result_dir = Path(self.result_dir) / timestr
            if isinstance(self.strategy, DefaultStrategy):
                self.strategy.grow_grad2d = self.strategy.grow_grad2d / self.camera_optimizer.num_virtual_views


class DeblurRunner(Runner):
    """Engine for training and testing."""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: DeblurConfig) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = ColmapParser(
        # self.parser = BlenderParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            scale_factor=cfg.scale_factor,
        )

        self.trainset = DeblurNerfDataset(self.parser, split="train")
        self.valset = DeblurNerfDataset(self.parser, split="val")
        self.testset = DeblurNerfDataset(self.parser, split="test")
        if len(self.valset.indices) == 0:
            self.valset = None
        if len(self.testset.indices) == 0:
            self.testset = None
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self.pose_optimizers = []
        self.camera_optimizer: BadCameraOptimizer = self.cfg.camera_optimizer.setup(
            # valset poses are not being optimized in training, but in NVS
            num_cameras=len(self.trainset) + (len(self.valset) if self.valset else 0),
            device=self.device,
        )
        camera_optimizer_param_groups = {}
        self.camera_optimizer.get_param_groups(camera_optimizer_param_groups)
        self.pose_optimizers = [
            torch.optim.Adam(
                camera_optimizer_param_groups["camera_opt"],
                lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.pose_opt_reg,
            )
        ]
        if world_size > 1:
            self.camera_optimizer = DDP(self.camera_optimizer)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree).to(
                self.device
            )
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            if cfg.visualize_cameras:
                self.viewer = PoseViewer(
                    server=self.server,
                    render_fn=self._viewer_render_fn,
                    mode="training",
                )
            else:
                self.viewer = Viewer(
                    server=self.server,
                    render_fn=self._viewer_render_fn,
                    mode="training",
                )

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
        ]

        # pose optimization has a learning rate schedule
        pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.pose_optimizers[0], gamma=cfg.pose_opt_lr_decay ** (1.0 / max_steps)
        )
        schedulers.append(pose_scheduler)

        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=cfg.pin_memory,
        )
        trainloader_iter = iter(trainloader)

        if cfg.visualize_cameras:
            self._init_viewer_state()

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            assert camtoworlds.shape[0] == 1
            camtoworlds = self.camera_optimizer.apply_to_cameras(camtoworlds, image_ids, "uniform")[
                0
            ]  # [num_virt_views, 4, 4]
            assert camtoworlds.shape[0] == cfg.camera_optimizer.num_virtual_views
            Ks = Ks.tile((camtoworlds.shape[0], 1, 1))

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # BAD-Gaussians: average the virtual views
            colors = colors.mean(0)[None]

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            if self.cfg.fused_ssim:
                ssimloss = 1.0 - fused_ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid")
            else:
                ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            if cfg.enable_mcmc_opacity_reg:
                loss = loss + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()

            if cfg.enable_mcmc_scale_reg:
                loss = loss + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()

            if cfg.enable_phys_scale_reg and step % 10 == 0:
                scale_exp = torch.exp(self.splats["scales"])
                scale_reg = (
                    torch.maximum(
                        scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                        torch.tensor(cfg.max_gauss_ratio),
                    )
                    - cfg.max_gauss_ratio
                )
                scale_reg = 0.1 * scale_reg.mean()
                loss += scale_reg

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)

                # monitor camera pose optimization
                metrics_dict = {}
                self.camera_optimizer.get_metrics_dict(metrics_dict)
                for k, v in metrics_dict.items():
                    self.writer.add_scalar(f"train/{k}", v, step)

                # monitor pose learning rate
                self.writer.add_scalar("train/poseLR", pose_scheduler.get_last_lr()[0], step)

                # monitor ATE
                #     self.visualize_traj(step)

                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if world_size > 1:
                    data["camera_opt"] = self.camera_optimizer.module.state_dict()
                else:
                    data["camera_opt"] = self.camera_optimizer.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                if step % cfg.pose_gradient_accumulation_steps == cfg.pose_gradient_accumulation_steps - 1:
                    optimizer.step()
                if step % cfg.pose_gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                if cfg.deblur_eval_enable_during_training and self.testset is not None:
                    if cfg.deblur_eval_enable_pose_opt:
                        self.eval_with_pose_opt(step, "deblur", self.testset)
                    else:
                        self.eval_deblur(step, "deblur", self.testset)
                if cfg.nvs_eval_enable_during_training and self.valset is not None:
                    self.eval_with_pose_opt(step, "nvs", self.valset)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval_deblur(self, step: int, stage: str, dataset: Dataset):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(testloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)

            # Apply learned mid-virtual-view pose optimizations
            camtoworlds = self.camera_optimizer.apply_to_cameras(camtoworlds, image_ids, "mid")

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            if world_rank == 0:
                # write images
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(f"{self.render_dir}/{step:04d}_{stage}_{i:04d}.png", (canvas * 255).astype(np.uint8))

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))
                    # write images
                    canvas = torch.cat([pixels, cc_colors], dim=2).squeeze(0).cpu().numpy()
                    imageio.imwrite(
                        f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_corrected.png", (canvas * 255).astype(np.uint8)
                    )

        if world_rank == 0:
            ellipse_time /= len(testloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            if cfg.use_bilateral_grid:
                print(
                    f"Corrected PSNR: {stats['cc_psnr']:.3f}, SSIM: {stats['cc_ssim']:.4f}, LPIPS: {stats['cc_lpips']:.3f} "
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    def eval_with_pose_opt(self, step: int, stage: str, dataset: Dataset):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        # Freeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = False

        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)

            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]

            eval_pose_adjust = CameraOptModuleSE3(1).to(self.device)
            eval_pose_adjust.random_init(cfg.pose_noise)
            eval_pose_optimizer = torch.optim.Adam(
                eval_pose_adjust.parameters(),
                lr=cfg.nvs_pose_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.nvs_pose_reg,
                eps=1e-15,
            )

            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                eval_pose_optimizer, gamma=cfg.pose_opt_lr_decay ** (1.0 / cfg.max_steps)
            )

            NVS_STEPS = cfg.nvs_steps_final if step == cfg.max_steps - 1 else cfg.nvs_steps
            for j in range(NVS_STEPS):
                camtoworlds_new = eval_pose_adjust(camtoworlds, torch.tensor([0]).to(self.device))
                colors, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds_new,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB",
                )
                # clamping here should be fine since we are only optimizing the camera
                colors = torch.clamp(colors, 0.0, 1.0)
                colors_p = colors.permute(0, 3, 1, 2).detach()  # [1, 3, H, W]

                # loss
                l1loss = F.l1_loss(colors, pixels)
                loss = l1loss

                loss.backward()

                eval_pose_optimizer.step()
                eval_pose_optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                with torch.no_grad():
                    if j % 20 == 0:
                        psnr = self.psnr(colors_p, pixels_p)
                        ssim = self.ssim(colors_p, pixels_p)
                        lpips = self.lpips(colors_p, pixels_p)
                        print(
                            f"Stage {stage} at Step_{step:04d}:"
                            f"NVS_IMG_#{i:04d}_step_{j:04d}:"
                            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                        )
                        if cfg.use_bilateral_grid:
                            cc_colors = color_correct(colors, pixels)
                            cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                            cc_psnr = self.psnr(cc_colors_p, pixels_p)
                            cc_ssim = self.ssim(cc_colors_p, pixels_p)
                            cc_lpips = self.lpips(cc_colors_p, pixels_p)
                            print(
                                f"Corrected PSNR: {cc_psnr.item():.3f}, SSIM: {cc_ssim.item():.4f}, LPIPS: {cc_lpips.item():.3f} "
                            )
                        # # NVS Debugging
                        # stats = {
                        #     "psnr": psnr.item(),
                        #     "ssim": ssim.item(),
                        #     "lpips": lpips.item(),
                        # }
                        # for k, v in stats.items():
                        #     self.writer.add_scalar(f"nvs/{step}/{i}/{k}", v, j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/pose_lr", scheduler.get_last_lr()[0], j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/camera_opt_translation", eval_pose_adjust.poses_opt[:, :3].mean(), j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/camera_opt_rotation", eval_pose_adjust.poses_opt[:, 3:].mean(), j)
                        # self.writer.flush()
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["lpips"].append(lpips)
            if cfg.use_bilateral_grid:
                metrics["cc_psnr"].append(cc_psnr)
                metrics["cc_ssim"].append(cc_ssim)
                metrics["cc_lpips"].append(cc_lpips)

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).detach().cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_{j:04d}.png", (canvas * 255).astype(np.uint8)
            )
            if cfg.use_bilateral_grid:
                canvas = torch.cat([pixels, cc_colors], dim=2).squeeze(0).detach().cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_{j:04d}_corrected.png",
                    (canvas * 255).astype(np.uint8),
                )
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        # save stats as json
        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
            json.dump(stats, f)

        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"{stage}/{k}", v, step)
        self.writer.flush()

        # Unfreeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = True

    @torch.no_grad()
    def eval_traj(self, step: int):
        # TODO: add gt trajectory

        # Get estimated trajectory
        camtoworlds = self.camera_optimizer.get_cameras()

        raise NotImplementedError

    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        if not self.cfg.disable_viewer and isinstance(self.viewer, PoseViewer):
            assert self.viewer and self.trainset
            self.viewer.init_scene(train_dataset=self.trainset, train_state="training")


def main(local_rank: int, world_rank, world_size: int, cfg: DeblurConfig):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_size > 1:
            print("Viewer is disabled in distributed training.")

    runner = DeblurRunner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [torch.load(file, map_location=runner.device, weights_only=False) for file in cfg.ckpt]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k].detach().to(runner.device) for ckpt in ckpts])
        runner.camera_optimizer.load_state_dict(ckpts[0]["camera_opt"])
        step = ckpts[0]["step"]
        if runner.testset is not None:
            if cfg.deblur_eval_enable_pose_opt:
                runner.eval_with_pose_opt(step=step, stage="deblur", dataset=runner.testset)
            else:
                runner.eval_deblur(step=step, stage="deblur", dataset=runner.testset)
        if runner.valset is not None:
            runner.eval_with_pose_opt(step=step, stage="nvs", dataset=runner.valset)

        runner.render_traj(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:
    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default
    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25
    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            DeblurConfig(
                strategy=DefaultStrategy(
                    verbose=True,
                    grow_grad2d=3e-3,
                    absgrad=True,
                    refine_start_iter=1000,
                ),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            DeblurConfig(
                init_opa=0.5,
                init_scale=0.1,
                strategy=MCMCStrategy(verbose=True, cap_max=500_000),
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
