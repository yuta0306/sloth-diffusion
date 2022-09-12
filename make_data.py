import json
import os
from typing import List, Tuple

import einops
import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm


def get_filenames(path: str) -> List[str]:
    for _, _, filenames in os.walk(path):
        return filenames


def split(filenames: List[str], val_size: int = 10000) -> Tuple[List[str], List[str]]:
    train_files = filenames[:-val_size]
    valid_files = filenames[-val_size:]
    return train_files, valid_files


def save_image_as_pt(file: str, save_to: str):
    try:
        image = Image.open(file)
        image = image.convert("RGB")
        image = np.array(image)
        image = torch.from_numpy(image)
        image = einops.rearrange(image, "h w c -> c h w")
        filename = file.split("/")[-1].removesuffix(".png")
        filename = filename.removesuffix(".jpg")
        torch.save(image, os.path.join(save_to, f"{filename}.pt"))
    except Exception:
        print("invalid image:", file)


if __name__ == "__main__":
    # filenames = get_filenames("./images")
    # print("Full data size:", len(filenames))

    # train_files, valid_files = split(filenames)

    # Parallel(n_jobs=-1)(
    #     delayed(save_image_as_pt)(os.path.join("images", filename), "./train")
    #     for filename in tqdm(train_files)
    # )
    # Parallel(n_jobs=-1)(
    #     delayed(save_image_as_pt)(os.path.join("images", filename), "./valid")
    #     for filename in tqdm(valid_files)
    # )
    for _, _, train_files in os.walk("train"):
        print(len(train_files))
    for _, _, valid_files in os.walk("valid"):
        print(len(valid_files))

    json.dump(train_files, open("train.json", "w"))
    json.dump(valid_files, open("valid.json", "w"))
