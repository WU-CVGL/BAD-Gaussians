import sys
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from datasets.colmap_parsing_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_text,
    write_images_text,
    write_points3D_text,
)



if __name__ == "__main__":
    # Usage: python colmap_bin2text.py path_to_data_dir
    if len(sys.argv) != 2:
        print("Usage: python colmap_bin2text.py path_to_data_dir")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    cameras = read_cameras_binary(data_dir / 'cameras.bin')
    images = read_images_binary(data_dir / 'images.bin')
    points3D = read_points3D_binary(data_dir / 'points3D.bin')

    # Write to human-readable text files
    write_cameras_text(cameras, data_dir / 'cameras.txt')
    write_images_text(images, data_dir / 'images.txt')
    write_points3D_text(points3D, data_dir / 'points3D.txt')

    print("Done.")