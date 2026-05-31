"""Torchvision COCO detection → bbox (ctx 좌표)."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from qwen_vlm.utils.image_resize import resize_max_side

# torchvision model_id → (builder, weights enum)
_TV_REGISTRY: dict[str, tuple[str, str]] = {
    "fasterrcnn_mobilenet_v3_large_fpn": (
        "fasterrcnn_mobilenet_v3_large_fpn",
        "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    ),
    "retinanet_resnet50_fpn": (
        "retinanet_resnet50_fpn",
        "RetinaNet_ResNet50_FPN_Weights",
    ),
    "ssd300_vgg16": (
        "ssd300_vgg16",
        "SSD300_VGG16_Weights",
    ),
}


def load_torchvision_detector(model_id: str, device: str) -> dict[str, Any]:
    if model_id not in _TV_REGISTRY:
        known = ", ".join(sorted(_TV_REGISTRY))
        raise ValueError(f"Unknown torchvision model {model_id!r}. Known: {known}")

    import torchvision
    from torchvision.models import detection as det

    builder_name, weights_enum_name = _TV_REGISTRY[model_id]
    builder = getattr(det, builder_name)
    weights_cls = getattr(det, weights_enum_name)
    weights = weights_cls.DEFAULT
    model = builder(weights=weights)
    model.eval()
    dev = torch.device(device if device else "cpu")
    model.to(dev)
    preprocess = weights.transforms()
    return {
        "model": model,
        "preprocess": preprocess,
        "device": dev,
        "model_id": model_id,
    }


def detect_torchvision_raw(
    image: Image.Image,
    handle: dict[str, Any],
    *,
    score_threshold: float = 0.25,
    imgsz_max_long: int = 0,
) -> tuple[list[tuple[tuple[int, int, int, int], int, float]], int, set[int]]:
    """
    ctx 좌표계 박스, (box, label, score), n_det(raw count before score), all labels.
    """
    rgb = image.convert("RGB")
    ctx_w, ctx_h = rgb.size
    det_img = resize_max_side(rgb.copy(), imgsz_max_long) if imgsz_max_long > 0 else rgb
    det_w, det_h = det_img.size
    sx = ctx_w / det_w
    sy = ctx_h / det_h

    preprocess = handle["preprocess"]
    model = handle["model"]
    dev = handle["device"]

    tensor = preprocess(det_img)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"unexpected preprocess output: {type(tensor)}")
    tensor = tensor.to(dev)

    with torch.inference_mode():
        out = model([tensor])[0]

    boxes = out["boxes"].detach().cpu()
    scores = out["scores"].detach().cpu()
    labels = out["labels"].detach().cpu()
    n_raw = len(boxes)
    all_cls: set[int] = set()
    raw_dets: list[tuple[tuple[int, int, int, int], int, float]] = []

    for i in range(n_raw):
        sc = float(scores[i])
        if sc < score_threshold:
            continue
        lab = int(labels[i])
        all_cls.add(lab)
        x1, y1, x2, y2 = boxes[i].tolist()
        x1 = int(max(0, min(ctx_w, round(x1 * sx))))
        y1 = int(max(0, min(ctx_h, round(y1 * sy))))
        x2 = int(max(0, min(ctx_w, round(x2 * sx))))
        y2 = int(max(0, min(ctx_h, round(y2 * sy))))
        if x2 <= x1 or y2 <= y1:
            continue
        raw_dets.append(((x1, y1, x2, y2), lab, sc))

    return raw_dets, n_raw, all_cls
