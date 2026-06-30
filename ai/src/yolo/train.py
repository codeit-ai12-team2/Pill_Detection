from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일을 읽어옵니다.

    Args:
        config_path: YAML 파일 경로

    Returns:
        : 설정 정보를 담은 Dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    if torch.cuda.is_available():
        device = 0
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS is available. Using MPS.")
    else:
        print("CUDA or MPS is not available. Using CPU.")
        device = "cpu"

    train_config = load_config("train.yaml")
    model_config = load_config("yolo11s.yaml")

    config = {**train_config, **model_config}

    model = YOLO(config.pop("model"))
    model.train(**config, device=device)

if __name__ == "__main__":
    main()