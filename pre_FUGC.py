from pathlib import Path

import numpy as np
from skimage import transform
from PIL import Image
from tqdm import tqdm

fugc_paths = {
    "labeled": {
        "images": Path.cwd() / "data" / "FUGC2025" / "labeled_data" / "images",
        "labels": Path.cwd() / "data" / "FUGC2025" / "labeled_data" / "labels",
    },
    "unlabeled": {
        "images": Path.cwd() / "data" / "FUGC2025" / "unlabeled_data" / "images",
        "labels": None,
    }
}

npy_paths = {
    "labeled": {
        "images": Path.cwd() / "data" / "npy" / "FUGC2025" / "labeled_data" / "imgs",
        "labels": Path.cwd() / "data" / "npy" / "FUGC2025" / "labeled_data" / "gts",
    },
    "unlabeled": {
        "images": Path.cwd() / "data" / "npy" / "FUGC2025" / "unlabeled_data" / "imgs",
        "labels": None,
    }
}

for t in npy_paths:
    for path in npy_paths[t].values():
        if path is None:
            continue
        path.mkdir(parents=True, exist_ok=True)

image_size = 1024

for t in fugc_paths:
    for image_path in tqdm(fugc_paths[t]["images"].glob("*.png")):
        image_name = image_path.stem
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        resized_image_np = transform.resize(
            image_np,
            (image_size, image_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )

        resize_image_np_01 = (resized_image_np - resized_image_np.min()) / np.clip(
            resized_image_np.max() -resized_image_np.min(),
            a_min=1e-8,
            a_max=None
        )

        np.save(
            npy_paths[t]["images"] / f"{image_name}.npy",
            resize_image_np_01
        )

        if fugc_paths[t]["labels"] is None:
            continue

        gt_image = Image.open(fugc_paths[t]["labels"] / image_path.name)
        gt_np = np.array(gt_image)

        resized_gt_np = transform.resize(
            gt_np,
            (image_size, image_size),
            order=0,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )

        np.save(
            npy_paths[t]["labels"] / f"{image_name}.npy",
            resized_gt_np
        )

