import json
import os
import shutil
import sys
from pathlib import Path
from glob import glob

import cv2
import numpy as np


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def run_ffmpeg(args, scripts_folder):
    ffmpeg_binary = "ffmpeg"

    # On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
    if os.name == "nt" and os.system(f"where {ffmpeg_binary} >nul 2>nul") != 0:
        ffmpeg_glob = os.path.join(ROOT_DIR, "external", "ffmpeg", "*", "bin", "ffmpeg.exe")
        candidates = glob(ffmpeg_glob)
        if not candidates:
            print("FFmpeg not found. Attempting to download FFmpeg from the internet.")
            do_system(os.path.join(scripts_folder, "download_ffmpeg.bat"))
            candidates = glob(ffmpeg_glob)

        if candidates:
            ffmpeg_binary = candidates[0]

    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)

    images = '"' + args.images + '"'
    video = '"' + args.video_in + '"'
    fps = float(args.video_fps) or 1.0
    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (
        not args.overwrite
        and (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip() + "y")[:1]
        != "y"
    ):
        sys.exit(1)
    try:
        # Passing Images' Path Without Double Quotes
        shutil.rmtree(args.images)
    except:
        pass
    do_system(f"mkdir {images}")

    time_slice_value = ""
    time_slice = args.time_slice
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"
    do_system(f'{ffmpeg_binary} -i {video} -qscale:v 1 -qmin 1 -vf "fps={fps}{time_slice_value}" {images}/%04d.jpg')


def run_colmap(args, scripts_folder):
    colmap_binary = "colmap"

    # On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
    if os.name == "nt" and os.system(f"where {colmap_binary} >nul 2>nul") != 0:
        colmap_glob = os.path.join(ROOT_DIR, "external", "colmap", "*", "COLMAP.bat")
        candidates = glob(colmap_glob)
        if not candidates:
            print("COLMAP not found. Attempting to download COLMAP from the internet.")
            do_system(os.path.join(scripts_folder, "download_colmap.bat"))
            candidates = glob(colmap_glob)

        if candidates:
            colmap_binary = candidates[0]

    db = args.colmap_db
    images = '"' + args.images + '"'
    db_noext = str(Path(db).with_suffix(""))

    # if args.text=="text":
    # 	args.text=db_noext+"_text"
    text = args.text
    sparse = "sparse"
    # sparse=db_noext+"_sparse"
    print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    if (
        not args.overwrite
        and (
            input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()
            + "y"
        )[:1]
        != "y"
    ):
        sys.exit(1)
    if os.path.exists(db):
        os.remove(db)
    do_system(
        f'{colmap_binary} feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params "{args.colmap_camera_params}" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}'
    )
    match_cmd = (
        f"{colmap_binary} {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
    )
    if args.vocab_path:
        match_cmd += f" --VocabTreeMatching.vocab_tree_path {args.vocab_path}"
    do_system(match_cmd)
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")

    mapper_args = []
    if args.colmap_fix_camera_params:
        mapper_args.append("--Mapper.ba_refine_focal_length 0")
        mapper_args.append("--Mapper.ba_refine_principal_point 0")
        mapper_args.append("--Mapper.ba_refine_extra_params 0")
    else:
        mapper_args.append("--Mapper.ba_refine_focal_length 1")
        mapper_args.append("--Mapper.ba_refine_principal_point 1")
        mapper_args.append("--Mapper.ba_refine_extra_params 1")

    mapper_args_str = " ".join(mapper_args)

    bundle_adjuster_args = []
    if args.colmap_fix_camera_params:
        # bundle_adjuster_args.append("--Mapper.ba_refine_focal_length 0")
        # bundle_adjuster_args.append("--Mapper.ba_refine_principal_point 0")
        # bundle_adjuster_args.append("--Mapper.ba_refine_extra_params 0")
        bundle_adjuster_args.append("--BundleAdjustment.refine_focal_length 0")
        bundle_adjuster_args.append("--BundleAdjustment.refine_principal_point 0")
        bundle_adjuster_args.append("--BundleAdjustment.refine_extra_params 0")
    else:
        # bundle_adjuster_args.append("--Mapper.ba_refine_focal_length 1")
        # bundle_adjuster_args.append("--Mapper.ba_refine_principal_point 1")
        # bundle_adjuster_args.append("--Mapper.ba_refine_extra_params 1")
        bundle_adjuster_args.append("--BundleAdjustment.refine_focal_length 1")
        bundle_adjuster_args.append("--BundleAdjustment.refine_principal_point 1")
        bundle_adjuster_args.append("--BundleAdjustment.refine_extra_params 1")

    bundle_adjuster_args_str = " ".join(bundle_adjuster_args)

    do_system(
        f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {sparse} {mapper_args_str}"
    )
    do_system(
        f"{colmap_binary} bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 {bundle_adjuster_args_str}"
    )
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    do_system(f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def sharpness(image_path: str) -> float:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # variance of laplacian
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def closest_point_2_lines(
    oa, da, ob, db
):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom
