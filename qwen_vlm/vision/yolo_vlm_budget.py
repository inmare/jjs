"""
YOLO 박스를 VLM에 넣기 전에 필터·총 픽셀 예산으로 줄여 비전 토큰을 억제.

외부 실험 스크립트와 동일한 의도:
과대 박스 제거, 중복·포함 관계 제거, 저해상 전망 좌표계에서의 중복 제거,
(전망 픽셀 + 원본 좌표 박스 면적 합)이 원본 이미지 픽셀 수를 넘지 않도록
면적이 작은 박스부터 제거.
"""
from __future__ import annotations


def _clamp_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    left = max(0, min(w - 1, int(round(x1))))
    top = max(0, min(h - 1, int(round(y1))))
    right = max(left + 1, min(w, int(round(x2))))
    bottom = max(top + 1, min(h, int(round(y2))))
    return left, top, right, bottom


def bbox_pixel_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def context_pixel_size(orig_w: int, orig_h: int, context_max_side: int) -> tuple[int, int]:
    """``resize_max_side`` 와 동일한 결과 크기 (긴 변 상한)."""
    m = max(orig_w, orig_h)
    if m <= context_max_side:
        return orig_w, orig_h
    scale = context_max_side / m
    return max(1, int(orig_w * scale)), max(1, int(orig_h * scale))


def context_view_pixel_size(
    orig_w: int,
    orig_h: int,
    *,
    context_max_side: int | None = None,
    context_resize_scale: float | None = None,
) -> tuple[int, int]:
    """
    저해상 **전망** 한 장의 (w,h). ``context_resize_scale`` 이 있으면 스크립트처럼
    가로·세로를 각각 해당 비율로 반올림; 없으면 ``context_max_side``(긴 변 캡).
    """
    if context_resize_scale is not None:
        if context_resize_scale <= 0:
            raise ValueError("context_resize_scale must be positive")
        return (
            max(1, int(round(orig_w * context_resize_scale))),
            max(1, int(round(orig_h * context_resize_scale))),
        )
    cms = context_max_side if context_max_side is not None else 960
    return context_pixel_size(orig_w, orig_h, cms)


def filter_large_objects(
    bboxes: list[tuple[int, int, int, int]],
    *,
    original_pixel_count: int,
    max_area_numerator: int = 1,
    max_area_denominator: int = 4,
) -> list[tuple[int, int, int, int]]:
    """박스 면적이 원본의 num/den 보다 크면 제외 (기본: 전체의 1/4 초과 제외)."""
    return [
        b
        for b in bboxes
        if bbox_pixel_area(b) * max_area_denominator < original_pixel_count * max_area_numerator
    ]


def filter_duplicate_bboxes(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for b in bboxes:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def filter_contained_objects(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    drop: set[int] = set()
    for i, inner in enumerate(bboxes):
        for j, outer in enumerate(bboxes):
            if i == j:
                continue
            if _bbox_contains(outer, inner):
                drop.add(i)
                break
    return [b for k, b in enumerate(bboxes) if k not in drop]


def filter_duplicate_projected_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    *,
    image_size: tuple[int, int],
    context_max_side: int | None = None,
    context_resize_scale: float | None = None,
) -> list[tuple[int, int, int, int]]:
    ow, oh = image_size
    rw, rh = context_view_pixel_size(
        ow, oh,
        context_max_side=context_max_side,
        context_resize_scale=context_resize_scale,
    )
    x_scale = rw / ow
    y_scale = rh / oh
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for b in bboxes:
        x1, y1, x2, y2 = b
        proj = _clamp_bbox(
            x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale, rw, rh
        )
        if proj in seen:
            continue
        seen.add(proj)
        out.append(b)
    return out


def enforce_vlm_pixel_budget(
    bboxes: list[tuple[int, int, int, int]],
    *,
    image_size: tuple[int, int],
    context_max_side: int | None = None,
    context_resize_scale: float | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    (전망 w×h) + Σ(원본 좌표 박스 면적) ≤ 원본 w×h 가 되도록,
    면적이 작은 박스부터 제거.
    """
    ow, oh = image_size
    original_pixel_count = ow * oh
    rw, rh = context_view_pixel_size(
        ow, oh,
        context_max_side=context_max_side,
        context_resize_scale=context_resize_scale,
    )
    total = rw * rh + sum(bbox_pixel_area(b) for b in bboxes)
    if total <= original_pixel_count:
        return bboxes

    remove: set[int] = set()
    order = sorted(
        range(len(bboxes)),
        key=lambda i: (bbox_pixel_area(bboxes[i]), i),
    )
    for i in order:
        if total <= original_pixel_count:
            break
        remove.add(i)
        total -= bbox_pixel_area(bboxes[i])
    return [b for j, b in enumerate(bboxes) if j not in remove]


def apply_yolo_vlm_budget_filters(
    bboxes: list[tuple[int, int, int, int]],
    *,
    image_size: tuple[int, int],
    context_max_side: int | None = None,
    context_resize_scale: float | None = None,
    max_area_numerator: int = 1,
    max_area_denominator: int = 4,
) -> list[tuple[int, int, int, int]]:
    """박스 리스트에 대해 필터 체인을 한 번에 적용."""
    if context_resize_scale is None and context_max_side is None:
        context_max_side = 960
    ow, oh = image_size
    original_pixel_count = ow * oh
    filtered = filter_large_objects(
        bboxes,
        original_pixel_count=original_pixel_count,
        max_area_numerator=max_area_numerator,
        max_area_denominator=max_area_denominator,
    )
    filtered = filter_duplicate_bboxes(filtered)
    filtered = filter_contained_objects(filtered)
    filtered = filter_duplicate_projected_bboxes(
        filtered,
        image_size=image_size,
        context_max_side=context_max_side,
        context_resize_scale=context_resize_scale,
    )
    filtered = enforce_vlm_pixel_budget(
        filtered,
        image_size=image_size,
        context_max_side=context_max_side,
        context_resize_scale=context_resize_scale,
    )
    return filtered
