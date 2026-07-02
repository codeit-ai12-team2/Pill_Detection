from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
import albumentations as A

YOLO_DIR = Path(__file__).parent


def load_config(config_path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> int | str:
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return 0
    if torch.backends.mps.is_available():
        print("MPS is available. Using MPS.")
        return "mps"
    print("CUDA or MPS is not available. Using CPU.")
    return "cpu"


def get_augmentations() -> list:
    return [
        # Blur variations
        # A.OneOf(
        #     [
        #         A.MotionBlur(blur_limit=7, p=1.0),
        #         A.MedianBlur(blur_limit=7, p=1.0),
        #         A.GaussianBlur(blur_limit=7, p=1.0),
        #     ],
        #     p=0.3,
        # ),
        # Noise variations
        # A.OneOf(
        #     [
        #         A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        #         A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        #     ],
        #     p=0.2,
        # ),
        # Color and contrast adjustments
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        # # Simulate occlusions
        # A.CoarseDropout(
        #     max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2
        # ),
    ]


def main(model_name: str = "yolo11s"):
    device = get_device()

    train_config = load_config(YOLO_DIR / "train.yaml")
    model_config = load_config(YOLO_DIR / f"{model_name}.yaml")

    config = {**train_config, **model_config}

    model_pt = YOLO_DIR / config.pop("model")
    config["data"] = str((YOLO_DIR / config["data"]).resolve())

    model = YOLO(model_pt)
    model.train(
        **config,
        device=device,
        project=str(YOLO_DIR / "runs/detect"),
        augmentations=get_augmentations()
    )


if __name__ == "__main__":
    main()