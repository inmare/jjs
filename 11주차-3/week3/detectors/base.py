"""탐지기 스펙 파싱·로드."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from week3.config import (
    CROP_AREA_NUMERATOR,
    DEFAULT_TORCHVISION_MODELS,
    DEFAULT_YOLO_IMGSZ_MAX_LONG,
    DEFAULT_YOLO_MODELS,
    THUMB_MAX_SIDE,
    method_name_for_spec,
)


@dataclass(frozen=True)
class DetectorSpec:
    backend: str  # yolo | torchvision
    model_id: str
    imgsz_max_long: int = 1536

    def label(self) -> str:
        isz = "native" if self.imgsz_max_long <= 0 else str(self.imgsz_max_long)
        if self.backend == "yolo":
            stem = self.model_id.replace(".pt", "").replace(".", "_")
            return f"YOLO_{stem}_ISZ{isz}"
        return f"TV_{self.model_id}_ISZ{isz}"

    def method_name(self) -> str:
        return method_name_for_spec(self)


# YOLO 1종 × imgsz 2 + Torchvision 3종 × imgsz 1 (기본)
def _default_specs() -> tuple[DetectorSpec, ...]:
    specs: list[DetectorSpec] = []
    yolo_model = DEFAULT_YOLO_MODELS[0]
    for isz in DEFAULT_YOLO_IMGSZ_MAX_LONG:
        specs.append(DetectorSpec("yolo", yolo_model, isz))
    tv_isz = DEFAULT_YOLO_IMGSZ_MAX_LONG[0] if DEFAULT_YOLO_IMGSZ_MAX_LONG else 1536
    for tv in DEFAULT_TORCHVISION_MODELS:
        specs.append(DetectorSpec("torchvision", tv, tv_isz))
    return tuple(specs)


DEFAULT_DETECTOR_SPECS: tuple[DetectorSpec, ...] = _default_specs()


def parse_detector_spec(text: str) -> DetectorSpec:
    """
    ``backend:model_id:imgsz`` 또는 ``backend:model_id`` (imgsz=1536).

    예: ``yolo:yolov8n.pt:640``, ``torchvision:ssd300_vgg16:1536``
    """
    parts = [p.strip() for p in text.split(":") if p.strip()]
    if len(parts) < 2:
        raise ValueError(
            f"detector spec 형식: backend:model_id[:imgsz], got {text!r}"
        )
    backend = parts[0].lower()
    if backend not in ("yolo", "torchvision", "tv"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "tv":
        backend = "torchvision"
    model_id = parts[1]
    imgsz = int(parts[2]) if len(parts) > 2 else 1536
    return DetectorSpec(backend, model_id, imgsz)


def specs_from_strings(items: list[str]) -> list[DetectorSpec]:
    return [parse_detector_spec(x) for x in items]


def expand_legacy_yolo_args(
    yolo_models: list[str],
    imgsz_list: list[int],
) -> list[DetectorSpec]:
    specs: list[DetectorSpec] = []
    for m in yolo_models:
        for isz in imgsz_list:
            specs.append(DetectorSpec("yolo", m, isz))
    return specs


def load_detector_backend(
    spec: DetectorSpec,
    device: str,
    cache: dict[str, Any],
) -> Any:
    key = f"{spec.backend}:{spec.model_id}"
    if key in cache:
        return cache[key]
    if spec.backend == "yolo":
        from qwen_vlm.vision.yolo import load_yolo

        from week3.config import resolve_yolo_weights_path

        path = resolve_yolo_weights_path(spec.model_id)
        print(f"Loading YOLO: {path}", flush=True)
        cache[key] = load_yolo(path)
    else:
        from week3.detectors.torchvision_det import load_torchvision_detector

        print(f"Loading Torchvision: {spec.model_id} on {device}", flush=True)
        cache[key] = load_torchvision_detector(spec.model_id, device)
    return cache[key]
