import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def load_sample_from_json(annot_folder_path: str, image_dir: str) -> tuple:
    """
    annotations 폴더에서 해당 이미지 및 바운딩 박스 정도 다운

    Args:
        annot_folder_path (str): annotations 폴더 경로
        image_dir (str): 이미지 파일 폴더 경로

    Returns:
        tuple: (image, target)
            - image (PIL.Image): RGB
            - target(dict): {
                "boxes": xyxy 형식,
                "labels": category id,
                "label_names": 클래스명 리스트
                "image_path": 이미지 파일 경로
                }
    """
    boxes, labels, label_names = [], [], []
    img_filename = None

    for sub_folder in os.listdir(annot_folder_path):
        sub_path = os.path.join(annot_folder_path, sub_folder)
        if not os.path.isdir(sub_path):
            continue

        for json_file in os.listdir(sub_path):
            if not json_file.endswith(".json"):
                continue

            json_path = os.path.join(sub_path, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if img_filename is None:
                img_filename = data["images"][0]["file_name"]

            categories = {cat["id"]: cat["name"] for cat in data["categories"]}

            for ann in data["annotations"]:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
                label_names.append(categories[ann["category_id"]])

    img_path = os.path.join(image_dir, img_filename)
    image = Image.open(img_path).convert("RGB")

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "label_names": label_names,
        "image_path": img_path,
    }

    return image, target


def build_color_palette(num_classes: int) -> list[tuple[float, float, float]]:
    """
    클래스 수에 맞게 색상 구분

    Args:
        num_classes (int): 생성할 색상 수(클래스 수)

    Returns:
        list[tuple[float, float, float]]: RGB 색상 (0~1범위)
    """
    colors = []
    for i in range(num_classes):
        hue = (i * 137.508) % 360 / 360
        s, v = 0.75, 0.95
        h = hue * 6
        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors.append((r + m, g + m, b + m))
    return colors


def visualize_sample(
    image,
    target: dict,
    ax: plt.Axes | None = None,
    show_labels: bool = True,
    line_width: float = 2.0,
    font_size: float = 9.0,
    num_classes: int = 56,
    palette: list[tuple[float, float, float]] | None = None,
) -> plt.Axes:
    """
    이미지 위에 바운딩 박스와 클래스 라벨 시각화

    Args:
        image : 시각화 이미지(PIL.Image, numpy.array)
        target (dict): 바운딩 박스 정보
                       - "boxes": (N, 4) 형태, xyxy 좌표 (Tensor 또는 ndarray)
                       - "labels": (N,) 형태, 카테고리 ID (Tensor 또는 list)
                       - "label_names": 클래스명 리스트
                       - "image_path": 이미지 파일 경로 (타이틀 표시용, 선택)
        ax (plt.Axes | None): 그릴 Axes 객체. None이면 새로 생성.
        show_labels (bool): 클래스명 텍스트화 여부(기본값: True)
        line_width (float):  바운딩 박스 선 두께(기본값: 2.0)
        font_size (float): 라벨 텍스트 글씨 크기(기본값: 9.0)
        num_classes (int): 색상 팔레트 사용할 때의 클래스 수(기본값: 56)
        palette (list[tuple[float, float, float]] | None): 사전 정의 함수 색상 팔레트(기본값: None)

    Returns:
        plt.Axes: 바운딩 박스가 그려진 Axes 객체
    """
    img_np = np.array(image)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(img_np)
    ax.axis("off")

    boxes = target.get("boxes")
    labels = target.get("labels")
    label_names = target.get("label_names", [])

    if boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0):
        return ax

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().tolist()

    colors = palette if palette is not None else build_color_palette(num_classes)

    for box, label_id, name in zip(boxes, labels or [], label_names):
        x1, y1, x2, y2 = box
        color = colors[int(label_id) % len(colors)]

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=line_width,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        if show_labels and name:
            ax.text(
                x1,
                y1 - 4,
                name,
                color="white",
                fontsize=font_size,
                fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.85),
                clip_on=True,
            )

    image_path = target.get("image_path", "")
    if image_path:
        ax.set_title(str(image_path).split("\\")[-1].split("/")[-1], fontsize=10)

    return ax
