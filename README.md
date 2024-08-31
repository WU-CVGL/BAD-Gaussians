<h1 align=center font-weight:100> 😈<strong><i>BAD-Gaussians</i></strong>: <strong><i>B</i></strong>undle-<strong><i>A</i></strong>djusted <strong><i>D</i></strong>eblur Gaussian Splatting</h1>

<a href="https://arxiv.org/abs/2403.11831"><img src="https://img.shields.io/badge/arXiv-2403.11831-b31b1b.svg"></a>
<a href="https://lingzhezhao.github.io/BAD-Gaussians/"><img src="https://img.shields.io/badge/Project-Page-green.svg"/></a>

This as an official implementation of our arXiv 2024 paper 
[**BAD-Gaussians**: Bundle Adjusted Deblur Gaussian Splatting](https://lingzhezhao.github.io/BAD-Gaussians/), based on the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework.

## Demo

Deblurring & novel-view synthesis results on [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF/)'s real-world motion-blurred data:

<video src="https://github.com/WU-CVGL/BAD-Gaussians/assets/43722188/703fbf8d-adb8-4472-b685-6dbe45bb0057"></video>

> Left: BAD-Gaussians deblured novel-view renderings;
>
> Right: Input images.


## Quickstart

### 1. Installation

You may check out the original [`nerfstudio`](https://github.com/nerfstudio-project/nerfstudio) repo for prerequisites and dependencies. 
Currently, our codebase is tested with nerfstudio v1.0.3.

TL;DR: You can install `nerfstudio` with:

```bash
# (Optional) create a fresh conda env
conda create --name nerfstudio -y "python<3.11"
conda activate nerfstudio

# install dependencies
pip install --upgrade pip setuptools
pip install "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install nerfstudio!
pip install nerfstudio==1.0.3
```

Then you can install this repo as a Python package with:

```bash
pip install git+https://github.com/WU-CVGL/BAD-Gaussians
```

### 2. Prepare the dataset

#### Deblur-NeRF Synthetic Dataset (Re-rendered)

As described in the previous BAD-NeRF paper, we re-rendered Deblur-NeRF's synthetic dataset with 51 interpolations per blurry image.

Additionally, in the previous BAD-NeRF paper, we directly run COLMAP on blurry images only, with neither ground-truth 
camera intrinsics nor sharp novel-view images. We find this is quite challenging for COLMAP - it may fail to 
reconstruct the scene and we need to re-run COLMAP for serval times. To this end, we provided a new set of data, 
where we ran COLMAP with ground-truth camera intrinsics over both blurry and sharp novel-view images, 
named `bad-nerf-gtK-colmap-nvs`:

[Download link](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EoCe3vaC9V5Fl74DjbGriwcBKj1nbB0HQFSWnVTLX7qT9A)

#### Deblur-NeRF Real Dataset

You can directly download the `real_camera_motion_blur` folder from [Deblur-NeRF](https://limacv.github.io/deblurnerf/).

#### Your Custom Dataset

1. Use the [`ns-process-data` tool from Nerfstudio](https://docs.nerf.studio/reference/cli/ns_process_data.html)
    to process deblur-nerf training images. 

    For example, if the
    [dataset from BAD-NeRF](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EsgdW2cRic5JqerhNbTsxtkBqy9m6cbnb2ugYZtvaib3qA?e=bjK7op)
    is in `llff_data`, execute:

    ```
    ns-process-data images \
        --data llff_data/blurtanabata/images \
        --output-dir data/my_data/blurtanabata
    ```

2. The folder `data/my_data/blurtanabata` is ready.

> Note: Although nerfstudio does not model the NDC scene contraction for LLFF data, 
> we found that `scale_factor = 0.25` works well on LLFF datasets.
> If your data is captured in a [LLFF fashion](https://github.com/Fyusion/LLFF#using-your-own-input-images-for-view-synthesis) (i.e. forward-facing), 
> instead of object-centric like Mip-NeRF 360, 
> you can pass the `scale_factor = 0.25` parameter to the nerfstudio dataparser (which is already set to default in our `DeblurNerfDataParser`),
> e.g., `ns-train bad-gaussians --data data/my_data/my_seq --vis viewer+tensorboard nerfstudio-data --scale_factor 0.25`

### 3. Training

1. For `Deblur-NeRF synthetic` dataset, train with:

    ```bash
    ns-train bad-gaussians \
        --data data/bad-nerf-gtK-colmap-nvs/blurtanabata \
        --pipeline.model.camera-optimizer.mode "linear" \
        --vis viewer+tensorboard \
        deblur-nerf-data
    ```
    where
    - `--data data/bad-nerf-gtK-colmap-nvs/blurtanabata` is the relative path of the data sequence;
    - `--pipeline.model.camera-optimizer.mode "linear"` enables linear camera pose interpolation
    - `--vis viewer+tensorboard` enables both the viewer and the tensorboard metrics saving
    - `deblur-nerf-data` chooses the DeblurNerfDataparser

2. For `Deblur-NeRF real` dataset with `downscale_factor=4`, train with:
    ```bash
    ns-train bad-gaussians \
        --data data/real_camera_motion_blur/blurdecoration \
        --pipeline.model.camera-optimizer.mode "cubic" \
        --vis viewer+tensorboard \
        deblur-nerf-data \
        --downscale_factor 4
    ```
    where
    - `--pipeline.model.camera-optimizer.mode "cubic"` enables cubic B-spline;
    - `--downscale_factor 4` after the `deblur-nerf-data` tells the DeblurNerfDataparser to downscale the images' width and height to `1/4` of its originals.

3. For `Deblur-NeRF real` dataset with *full resolution*, train with:
    ```bash
    ns-train bad-gaussians \
        --data data/real_camera_motion_blur/blurdecoration \
        --pipeline.model.camera-optimizer.mode "cubic" \
        --pipeline.model.camera-optimizer.num_virtual_views 15 \
        --pipeline.model.num_downscales 2 \
        --pipeline.model.resolution_schedule 3000 \
        --vis viewer+tensorboard \
        deblur-nerf-data
    ```
    where
    - `--pipeline.model.camera-optimizer.mode "cubic"` enables cubic B-spline;
    - `--pipeline.model.camera-optimizer.num_virtual_views 15` increases the number of virtual cameras to 15;
    - `--pipeline.model.num_downscales 2` and `--pipeline.model.resolution_schedule 3000` enables coarse-to-fine training.

4. For custom data processed with `ns-process-data`, train with:

    ```bash
    ns-train bad-gaussians \
        --data data/my_data/blurtanabata \
        --vis viewer+tensorboard \
        nerfstudio-data --eval_mode "all"
    ```

    > Note: To improve reconstruction quality on your custom dataset, you may need to add 
    some of the parameters to enable *cubic B-spline*, *more virtual cameras* and
    *coarse-to-fine training*, as shown in the examples above.

### 4. Render videos

This command will generate a trajectory with the camera poses of the training images, keeping their original order, interplate 10 frames between adjacent images with a frame rate of 30. It will load the `config.yml` and save the video to `renders/<your_filename>.mp4`.

```bash
ns-render interpolate \
  --load-config outputs/blurtanabata/bad-gaussians/<your_experiment_date_time>/config.yml \
  --pose-source train \
  --frame-rate 30 \
  --interpolation-steps 10 \
  --output-path renders/<your_filename>.mp4
```

> Note1: You can add the `--render-nearest-camera True` option to compare with the blurry inputs, but it will slow down the rendering process significantly.
>
> Note2: The working directory when executing this command must be the parent of `outputs`, i.e. the same directory when training.
>
> Note3: You can find more information of this command in the [nerfstudio docs](https://docs.nerf.studio/reference/cli/ns_render.html#ns-render).

### 5. Export the 3D Gaussians

This command will load the `config.yml` and export a `splat.ply` into the same folder:

```bash
ns-export gaussian-splat \
    --load-config outputs/blurtanabata/bad-gaussians/<your_experiment_date_time>/config.yml \
    --output-dir outputs/blurtanabata/bad-gaussians/<your_experiment_date_time>
```

> Note1: We use `rasterize_mode = antialiased` by default. However, if you want to export the 3D gaussians, since the `antialiased` mode (i.e. *Mip-Splatting*) is not supported by most 3D-GS viewers, it is better to turn if off during training using: `--pipeline.model.rasterize_mode "classic"`
>
> Note2: The working directory when executing this command must be the parent of `outputs`, i.e. the same directory when training.

Then you can visualize this file with any viewer, for example the [WebGL Viewer](https://antimatter15.com/splat/).

### 6. Debug with your IDE

Open this repo with your IDE, create a configuration, and set the executing python script path to
`<nerfstudio_path>/nerfstudio/scripts/train.py`, with the parameters above.


## Citation

If you find this useful, please consider citing:

```bibtex
@inproceedings{zhao2024badgaussians,
    author = {Zhao, Lingzhe and Wang, Peng and Liu, Peidong},
    title = {Bad-gaussians: Bundle adjusted deblur gaussian splatting},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2024}
} 
```

## Acknowledgments

- Kudos to the [Nerfstudio](https://github.com/nerfstudio-project/) and [gsplat](https://github.com/nerfstudio-project/gsplat) contributors for their amazing works:

    ```bibtex
    @inproceedings{nerfstudio,
        title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
        author       = {
            Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
            and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
            Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
            Angjoo
        },
        year         = 2023,
        booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
        series       = {SIGGRAPH '23}
    }

    @software{Ye_gsplat,
        author  = {Ye, Vickie and Turkulainen, Matias, and the Nerfstudio team},
        title   = {{gsplat}},
        url     = {https://github.com/nerfstudio-project/gsplat}
    }

    @misc{ye2023mathematical,
        title={Mathematical Supplement for the $\texttt{gsplat}$ Library}, 
        author={Vickie Ye and Angjoo Kanazawa},
        year={2023},
        eprint={2312.02121},
        archivePrefix={arXiv},
        primaryClass={cs.MS}
    }
    ```

- Kudos to the [pypose](https://github.com/pypose/pypose) contributors for their amazing library:

    ```bibtex
    @inproceedings{wang2023pypose,
    title = {{PyPose}: A Library for Robot Learning with Physics-based Optimization},
    author = {Wang, Chen and Gao, Dasong and Xu, Kuan and Geng, Junyi and Hu, Yaoyu and Qiu, Yuheng and Li, Bowen and Yang, Fan and Moon, Brady and Pandey, Abhinav and Aryan and Xu, Jiahe and Wu, Tianhao and He, Haonan and Huang, Daning and Ren, Zhongqiang and Zhao, Shibo and Fu, Taimeng and Reddy, Pranay and Lin, Xiao and Wang, Wenshan and Shi, Jingnan and Talak, Rajat and Cao, Kun and Du, Yi and Wang, Han and Yu, Huai and Wang, Shanzhao and Chen, Siyu and Kashyap, Ananth  and Bandaru, Rohan and Dantu, Karthik and Wu, Jiajun and Xie, Lihua and Carlone, Luca and Hutter, Marco and Scherer, Sebastian},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023}
    }
    ```
