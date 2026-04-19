"""감시·통로 시나리오용 YOLO(COCO) 클래스 프리셋 및 크롭 추출."""
from __future__ import annotations

import numpy as np
from PIL import Image

# COCO (Ultralytics YOLOv8): 사람·차량·휴대물 — 공장·캠퍼스 안전 스토리에 맞춘 부분집합
FACTORY_SURVEILLANCE_CLASS_IDS: tuple[int, ...] = (
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
    24,  # backpack
    26,  # handbag
    28,  # suitcase
)

FACTORY_SURVEILLANCE_LABELS: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}


def load_yolo(model_path: str = "yolov8n.pt"):
    from ultralytics import YOLO as _YOLO

    return _YOLO(model_path)


def run_yolo_crops(
    image: Image.Image,
    *,
    model,
    max_crops: int = 3,
    conf: float = 0.25,
    class_ids: tuple[int, ...] = FACTORY_SURVEILLANCE_CLASS_IDS,
) -> tuple[list[Image.Image], str, int, set[int]]:
    """탐지 박스 상위 max_crops 개를 크롭. (크롭, 요약문, 전체 탐지 수, 클래스 집합)."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    results = model.predict(
        source=np.array(rgb),
        classes=list(class_ids),
        conf=conf,
        verbose=False,
    )[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return [], "no_detections", 0, set()

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    all_cls = {int(x) for x in cls.tolist()}
    n_det = len(boxes)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort(-areas)

    crops: list[Image.Image] = []
    desc_parts: list[str] = []
    for idx in order[:max_crops]:
        x1, y1, x2, y2 = map(int, xyxy[idx])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(rgb.crop((x1, y1, x2, y2)))
        c = int(cls[idx])
        name = FACTORY_SURVEILLANCE_LABELS.get(c, f"class_{c}")
        desc_parts.append(f"{name}:{conf_arr[idx]:.2f}")

    summary = ", ".join(desc_parts) if desc_parts else "no_detections"
    return crops, summary, n_det, all_cls


def yolo_heuristic_risk(
    *,
    num_detections: int,
    class_ids_detected: set[int],
) -> str:
    """1단계 대체용: 탐지 수·유형만으로 low/med/high."""
    if num_detections == 0:
        return "low"
    vehicles = {1, 2, 3, 5, 7}
    if class_ids_detected & vehicles:
        return "med"
    if num_detections >= 3:
        return "med"
    if num_detections >= 1:
        return "low"
    return "low"
