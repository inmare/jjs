"""PIL 리사이즈 유틸 (실험 파이프라인·HR-Bench 공통)."""
from __future__ import annotations

from PIL import Image


def resize_max_side_dims(w: int, h: int, max_side: int) -> tuple[int, int]:
    """긴 변 캡만 적용한 (w,h). ``resize_max_side`` 와 같은 비율."""
    m = max(w, h)
    if m <= max_side:
        return w, h
    scale = max_side / m
    return max(1, int(w * scale)), max(1, int(h * scale))


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """긴 변이 `max_side` 를 넘지 않도록 비율 유지 축소."""
    w, h = img.size
    nw, nh = resize_max_side_dims(w, h, max_side)
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def resize_uniform_scale(img: Image.Image, scale: float) -> Image.Image:
    """가로·세로를 동일 비율로 스케일 (스크립트 ``RESIZE_SCALE`` 과 동일). ``scale`` ≤ 0 이면 ValueError."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    w, h = img.size
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if nw == w and nh == h:
        return img
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def resize_max_long(img: Image.Image, max_long: int) -> Image.Image:
    """긴 변(가로·세로 중 큰 쪽)이 `max_long` 을 넘지 않도록 축소. 0 이하면 원본."""
    if max_long <= 0:
        return img
    w, h = img.size
    long_ = max(w, h)
    if long_ <= max_long:
        return img
    s = max_long / float(long_)
    return img.resize(
        (max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS
    )
