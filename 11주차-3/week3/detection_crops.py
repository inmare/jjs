"""YOLO / Torchvision 공통: raw bbox → crop (VLM budget 동일)."""
from __future__ import annotations

from typing import Any

from PIL import Image

from qwen_vlm.vision.yolo import FACTORY_SURVEILLANCE_LABELS, run_yolo_crops
from qwen_vlm.vision.yolo_vlm_budget import apply_yolo_vlm_budget_filters, bbox_pixel_area

from week3.config import (
    CROP_AREA_DENOMINATOR,
    CROP_AREA_NUMERATOR,
    YOLO_CONTEXT_SCALE,
)
from week3.detectors.base import DetectorSpec
from week3.detectors.torchvision_det import detect_torchvision_raw


def _finalize_raw_dets(
    rgb_work: Image.Image,
    raw_dets: list[tuple[tuple[int, int, int, int], int, float]],
    n_det: int,
    all_cls: set[int],
    *,
    vlm_budget: bool = True,
    max_crops: int = 0,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    original_full_image_size: tuple[int, int] | None = None,
    vlm_overview_max_side_for_budget: int = 0,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]], str, int, set[int]]:
    w, h = rgb_work.size

    filtered: list[tuple[tuple[int, int, int, int], int, float]] = []
    for box, c_i, cf in raw_dets:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if min_crop_short_side > 0 and min(bw, bh) < min_crop_short_side:
            continue
        if min_crop_area > 0 and area < min_crop_area:
            continue
        filtered.append((box, c_i, cf))

    seen: set[tuple[int, int, int, int]] = set()
    dedup: list[tuple[tuple[int, int, int, int], int, float]] = []
    for box, c_i, cf in filtered:
        if box in seen:
            continue
        seen.add(box)
        dedup.append((box, c_i, cf))
    dedup.sort(key=lambda t: bbox_pixel_area(t[0]), reverse=True)

    bboxes_work = [c[0] for c in dedup]
    if vlm_budget and bboxes_work:
        bboxes_work = apply_yolo_vlm_budget_filters(
            bboxes_work,
            image_size=(w, h),
            context_max_side=None,
            context_resize_scale=YOLO_CONTEXT_SCALE,
            max_area_numerator=CROP_AREA_NUMERATOR,
            max_area_denominator=CROP_AREA_DENOMINATOR,
            original_image_size=original_full_image_size,
            vlm_overview_max_side=vlm_overview_max_side_for_budget,
            crop_max_side_for_budget=0,
        )

    meta_by_box: dict[tuple[int, int, int, int], tuple[int, float]] = {}
    for box, c_i, cf in dedup:
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
        crops.append(rgb_work.crop((x1, y1, x2, y2)).copy())
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


def run_detector_crops(
    ctx: Image.Image,
    *,
    spec: DetectorSpec,
    backend: Any,
    device: str,
    original_full_image_size: tuple[int, int],
    vlm_overview_max_side_for_budget: int,
    max_crops: int = 0,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]], str, int, set[int]]:
    if spec.backend == "yolo":
        return run_yolo_crops(
            ctx,
            model=backend,
            yolo_device=device,
            vlm_budget=True,
            context_resize_scale=YOLO_CONTEXT_SCALE,
            max_bbox_area_numerator=CROP_AREA_NUMERATOR,
            max_bbox_area_denominator=CROP_AREA_DENOMINATOR,
            original_full_image_size=original_full_image_size,
            vlm_overview_max_side_for_budget=vlm_overview_max_side_for_budget,
            yolo_imgsz_max_long=spec.imgsz_max_long,
            max_crops=max_crops,
        )
    raw_dets, n_det, all_cls = detect_torchvision_raw(
        ctx,
        backend,
        imgsz_max_long=spec.imgsz_max_long,
    )
    return _finalize_raw_dets(
        ctx.convert("RGB").copy(),
        raw_dets,
        n_det,
        all_cls,
        max_crops=max_crops,
        original_full_image_size=original_full_image_size,
        vlm_overview_max_side_for_budget=vlm_overview_max_side_for_budget,
    )
