"""감시·통로 시나리오용 YOLO(COCO) 클래스 프리셋 및 크롭 추출."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

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


def yolo_imgsz_from_image_size(width_height: tuple[int, int]) -> tuple[int, int]:
    """외부 스크립트와 동일: ``(W,H)`` → Ultralytics ``imgsz`` 용 ``(H,W)``."""
    w, h = width_height
    return h, w


def run_yolo_crops(
    image: Image.Image,
    *,
    model,
    predict_source: Path | str | None = None,
    yolo_device: str = "cpu",
    max_crops: int = 0,
    conf: float = 0.25,
    class_ids: tuple[int, ...] | None = None,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    vlm_budget: bool = True,
    context_max_side: int | None = None,
    context_resize_scale: float | None = None,
    max_bbox_area_numerator: int = 1,
    max_bbox_area_denominator: int = 4,
) -> tuple[
    list[Image.Image],
    list[tuple[int, int, int, int]],
    str,
    int,
    set[int],
]:
    """
    탐지 박스를 면적 큰 순으로 보며, **크기 필터**를 통과하는 것만 최대 max_crops 개까지 크롭.

    반환:
      - crops: 잘라낸 PIL(원본 좌표계와 동일 해상도)
      - bboxes: 각 crop에 대응하는 원본 이미지 기준 (x1,y1,x2,y2)
      - summary: 로그/JSON용(필터를 통과해 실제로 보내는 박스만 열거; VLM 프롬프트에는 쓰지 않음)
      - n_det: YOLO raw 탐지 개수(필터 전)
      - all_cls: 탐지에 등장한 클래스 id 집합(필터 전)

    min_crop_short_side: 0이면 끔. 0보다 크면 min(너비,높이) < 이면 제외.
    min_crop_area: 0이면 끔. 0보다 크면 (너비*높이) < 이면 제외.
    vlm_budget: True면 탐지 박스에 대해 과대/중복/포함/저해상 투영 중복 제거 및
        총 픽셀 예산(전망+박스 면적 합 ≤ 원본 픽셀) 적용 후 ``max_crops`` 로 자름.
    max_crops: 0이면 예산·필터 후 남은 박스를 **모두** 크롭(외부 스크립트와 동일).
    class_ids: ``None``이면 COCO 전 클래스(``classes=`` 미지정). 지정 시 해당 id만.
    context_max_side: ``vlm_budget`` 이고 ``context_resize_scale`` 이 없을 때 전망 픽셀
        계산용(미주입 시 960).
    context_resize_scale: 설정 시 전망을 가로·세로 이 비율로 본다는 가정으로 예산·투영 중복.
    max_bbox_area_numerator / max_bbox_area_denominator: 원본 대비 최대 박스 면적 상한
        (기본: 면적 < 원본×numerator/denominator 만 유지 → 1/4).
    predict_source: 디스크 경로가 있으면 ``model.predict(str(path))`` 로 Ultralytics 가
        이미지를 읽게 함(원본 스크립트와 동일). ``None`` 이면 PIL 을 임시 JPEG 로 저장 후 추론.
    yolo_device: 기본 ``\"cpu\"`` — llama-server(Qwen) GPU 와 VRAM 을 나누지 않음.
    """
    from qwen_vlm.vision.yolo_vlm_budget import apply_yolo_vlm_budget_filters

    rgb = image.convert("RGB")
    w, h = rgb.size
    imgsz = yolo_imgsz_from_image_size((w, h))

    tmp_path: str | None = None
    try:
        if predict_source is not None:
            src = str(Path(predict_source).expanduser().resolve())
        else:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            rgb.save(tmp_path, format="JPEG", quality=95)
            src = tmp_path
        pred_kw: dict[str, object] = {
            "source": src,
            "conf": conf,
            "verbose": False,
            "device": yolo_device,
            "imgsz": imgsz,
        }
        if class_ids is not None:
            pred_kw["classes"] = list(class_ids)
        results = model.predict(**pred_kw)[0]
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], "no_detections", 0, set()

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    all_cls = {int(x) for x in cls.tolist()}
    n_det = len(boxes)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort(-areas)

    candidates: list[tuple[tuple[int, int, int, int], int, float]] = []
    for idx in order:
        x1, y1, x2, y2 = map(int, xyxy[idx])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if min_crop_short_side > 0 and min(bw, bh) < min_crop_short_side:
            continue
        if min_crop_area > 0 and area < min_crop_area:
            continue
        box = (x1, y1, x2, y2)
        candidates.append((box, int(cls[idx]), float(conf_arr[idx])))

    bboxes_work = [c[0] for c in candidates]
    if vlm_budget and bboxes_work:
        cms = None if context_resize_scale is not None else (
            context_max_side if context_max_side is not None else 960
        )
        bboxes_work = apply_yolo_vlm_budget_filters(
            bboxes_work,
            image_size=(w, h),
            context_max_side=cms,
            context_resize_scale=context_resize_scale,
            max_area_numerator=max_bbox_area_numerator,
            max_area_denominator=max_bbox_area_denominator,
        )

    meta_by_box: dict[tuple[int, int, int, int], tuple[int, float]] = {}
    for box, c_i, cf in candidates:
        if box not in meta_by_box:
            meta_by_box[box] = (c_i, cf)

    crops: list[Image.Image] = []
    bboxes: list[tuple[int, int, int, int]] = []
    desc_parts: list[str] = []
    for box in bboxes_work:
        if max_crops > 0 and len(crops) >= max_crops:
            break
        if box not in meta_by_box:
            continue
        x1, y1, x2, y2 = box
        crops.append(rgb.crop((x1, y1, x2, y2)))
        bboxes.append(box)
        c, cf = meta_by_box[box]
        name = FACTORY_SURVEILLANCE_LABELS.get(c, f"class_{c}")
        desc_parts.append(f"{name}:{cf:.2f}")

    if not crops and n_det == 0:
        summary = "no_detections"
    elif not crops and n_det > 0:
        summary = "all_crops_filtered_by_size"
    else:
        summary = ", ".join(desc_parts) if desc_parts else "no_detections"
    return crops, bboxes, summary, n_det, all_cls


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
