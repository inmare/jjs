"""
연속 프레임 **효율 처리**용: 저해상도 휘도(그레이) 비교 + YOLO 탐지 **서명** 일치.

numpy 없이 **Pillow** 만 사용( ``main.py`` 만 쓰는 환경에서도 import 가능 ).

**한계:** YOLO 서명은 ``n_det`` + **클래스 id 집합**만 쓰므로, 같은 수·같은 종이라도
박스가 크게 옮겨가면 "동일" 으로 볼 수 있다. MSE(픽셀) 와 **조합( ``mse_then_yolo`` )** 권장.
"""
from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageChops

DEFAULT_THUMB_SIZE = (160, 90)


def luma_thumb(
    img: Image.Image, size: tuple[int, int] = DEFAULT_THUMB_SIZE
) -> Image.Image:
    """``RGB`` → 리사이즈 → ``L`` (간단 Y 구성 사용)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(size, Image.Resampling.LANCZOS).convert("L")


def mean_abs_diff_luma(
    a: Image.Image, b: Image.Image, size: tuple[int, int] = DEFAULT_THUMB_SIZE
) -> float:
    """
    저해상도 **휘도** 썸에서 `ImageChops.difference` 의 **평균 픽셀값**에 해당하는 지표
    (0에 가까울수록 유사, 보통 0~255; 임계는 ``bench_with_frame_gating`` 의
    ``mse_threshold``).
    """
    ta = luma_thumb(a, size)
    tb = luma_thumb(b, size)
    w, h = ta.size
    diff = ImageChops.difference(ta, tb)
    hst = diff.histogram()
    return float(sum(i * c for i, c in enumerate(hst)) / (w * h))


def yolo_fingerprint(n_det: int, class_ids: Sequence[int] | None) -> str:
    s = "" if not class_ids else ",".join(str(c) for c in sorted(int(x) for x in class_ids))
    return f"{n_det}|{s}"


def _centers_normalized_xyxy(
    bboxes: Sequence[tuple[int, int, int, int]], w: int, h: int
) -> list[tuple[float, float]]:
    """각 박스의 중심을 (0~1)로 정규화한 뒤 (y,x) 순으로 정렬해 **대응** 순서를 맞춘다."""
    if w <= 0 or h <= 0:
        return []
    out: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in bboxes:
        cx = 0.5 * (x1 + x2) / float(w)
        cy = 0.5 * (y1 + y2) / float(h)
        out.append((cx, cy))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def is_crop_layout_stable(
    bboxes_a: Sequence[tuple[int, int, int, int]],
    bboxes_b: Sequence[tuple[int, int, int, int]],
    wa: int,
    ha: int,
    wb: int,
    hb: int,
    *,
    max_norm_center_shift: float,
) -> bool:
    """
    **True** = 크롭 개수·정규화 박스 중심이 이전과 비슷해 **Qwen VLM 경로를 생략**해도 된다고 볼 수 있음.

    - 개수가 다르면 False(재검사).
    - 개수 0=0이면 True(빈 장면 지속).
    - 각 박스는 (y,x)로 정렬한 뒤 동일 인덱스끼리 L2(정규화 좌표)로 비교.
    - `max_norm_center_shift`: 한 박스당 허용 최대 이동(0~1 스케일, 대각선 아님). 예: 0.02.
    """
    if len(bboxes_a) != len(bboxes_b):
        return False
    ca = _centers_normalized_xyxy(bboxes_a, wa, ha)
    cb = _centers_normalized_xyxy(bboxes_b, wb, hb)
    if len(ca) != len(cb):
        return False
    for (xa, ya), (xb, yb) in zip(ca, cb):
        d = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
        if d > max_norm_center_shift:
            return False
    return True


def crop_gate_fingerprint(
    w: int,
    h: int,
    bboxes: Sequence[tuple[int, int, int, int]],
) -> str:
    """로그/JSON: 크롭 개수 + (cx/W,cy/H) 정규화 중심을 정렬한 문자열."""
    if w <= 0 or h <= 0:
        return f"{len(bboxes)}|bad_wh"
    parts: list[str] = []
    for x1, y1, x2, y2 in bboxes:
        cx = 0.5 * (x1 + x2) / float(w)
        cy = 0.5 * (y1 + y2) / float(h)
        parts.append(f"{cx:.3f},{cy:.3f}")
    parts.sort()
    return f"{len(bboxes)}|" + ",".join(parts)
