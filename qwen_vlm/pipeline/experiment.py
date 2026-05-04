"""
VLM 실험용 파이프라인 (Qwen + 선택적 Smol + Ultralytics YOLO).

- ``bench`` / ``bench_with_frame_gating`` : 프레임마다 Qwen-only 와 (저해상도 전망+YOLO 크롭) VLM.
  게이트: MSE(휘도) / **클래스 서명** / **크롭 위치 안정**(`--use-crop-layout-gate`) 중 선택.
- ``parallel-yolo-smol`` : YOLO∥Smol 후 Qwen(다중 이미지+좌표 텍스트).
- ``two-stage`` : Smol(또는 YOLO JSON 폴백) → Qwen.

지원하는 실행 모드(``uv run python experiment_pipeline.py <mode>`` 또는 ``uv run python -m qwen_vlm.pipeline.experiment <mode>``) 요약:

- **bench**: 프레임마다 (1) Qwen-only, (2) YOLO 크롭 후 YOLO+Qwen 을 *순차* 실행해
  `qwen_only` / `yolo_qwen` 블록을 JSON 으로 남긴다. 베이스라인 비교용.
- **sequence-compare**: 베이스라인(매 프레임 YOLO+크롭 Qwen만)과
  :func:`bench_with_frame_gating` 결과를 한 JSON/HTML에 묶는다.
- **two-stage**: 1단계로 SmolVLM(또는 실패 시 YOLO JSON 폴백) → 2단계 Qwen.
  ``--skip-qwen-if-low`` 시 1단계 ``risk=low`` 이면 Qwen 호출을 생략한다.
- **parallel-yolo-smol**: 1단계에서 **YOLO** 와 **SmolVLM(HTTP API)** 를
  :class:`concurrent.futures.ThreadPoolExecutor` 로 **같은 프레임에 대해 병렬** 실행한 뒤,
  **저해상도 전망 1장 + (원본 좌표 설명) + YOLO 크롭(``crop_max_side``)** 과 Smol 텍스트로
  Qwen을 1회 호출한다(클래스명·점수는 VLM 프롬프트에 넣지 않음).

**주의:** ``parallel-yolo-smol`` 은 Qwen·Smol **두 llama-server** + Ultralytics YOLO 가
  동시에 GPU/CPU 를 쓰므로 VRAM 이 빠듯한 환경(예: 8GB)에서는 OOM 이 날 수 있다.

``qwen_vlm.pipeline.week`` (루트 ``run_week_experiments.py``) 는 8GB 대비로 단계마다 서버를 켰다 끄는 편이며,
3단계에서만 Qwen+Smol 을 **동시에** 띄운 뒤 위 병렬 파이프를 돌린다.

**Qwen llama-server:** ``--base-url`` 이 루프백이고 짧은 시간 안에 ``/models`` 에 응답이 없으면
(기본) ``vendor`` 의 llama-server+GGUF 를 자동 기동하고, 실행이 끝나면 종료합니다. 끄려면 ``--no-spawn-llama`` .
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from openai import OpenAI
from PIL import Image

from qwen_vlm.main import (
    ROOT,
    chat_vlm_multi,
    host_port_from_openai_base_url,
    pil_to_data_url,
    start_llama_server,
    wait_for_server,
)
from qwen_vlm.gating.frame_gating import (
    crop_gate_fingerprint,
    is_crop_layout_stable,
    luma_thumb,
    mean_abs_diff_luma,
    yolo_fingerprint,
)
from qwen_vlm.reporting.experiment_metrics import (
    resource_snapshot,
    usage_to_dict,
    write_sequence_compare_html,
    write_single_image_experiment_html_from_payload,
)
from qwen_vlm.utils.image_resize import resize_max_side, resize_uniform_scale
from qwen_vlm.utils.openai_compat import normalize_openai_base_url
from qwen_vlm.utils.stdio_utf8 import configure_stdio_utf8
from qwen_vlm.vision.tokens import (
    VENDOR_QWEN3VL_4B_MAX_PIXELS,
    VENDOR_QWEN3VL_4B_MIN_PIXELS,
    smart_resize,
    vision_grid_tokens,
)
from qwen_vlm.vision.yolo import (
    load_yolo,
    run_yolo_crops,
    yolo_heuristic_risk,
)


def prep_context_image(
    img: Image.Image,
    *,
    context_max_side: int,
    context_resize_scale: float | None,
) -> Image.Image:
    """저해상 전망 1장. ``context_resize_scale`` 이 있으면 스크립트式 비율 축소, 없으면 긴 변 캡."""
    if context_resize_scale is not None:
        return resize_uniform_scale(img, context_resize_scale)
    return resize_max_side(img, context_max_side)


def prep_yolo_crop_for_vlm(crop: Image.Image, crop_max_side: int) -> Image.Image:
    """다음 노드(URL 인코드 등)까지 독립 픽셀 버퍼을 보장하려 크롭은 항상 ``.copy()`` 후 리사이즈."""
    pc = crop.copy()
    if crop_max_side <= 0:
        return pc
    return resize_max_side(pc, crop_max_side)

DEFAULT_PROMPT_KO = (
    "이 감시 화면에서 사람·차량·이상 징후가 있는지 짧게 요약해줘. 한국어로."
)
STAGE1_PROMPT_EN = (
    'Reply with ONLY valid JSON on a single line, no markdown fences. Fields: '
    '{"objects":["..."],"risk":"low|med|high","note":"..."} '
    'Use at most 4 short object CATEGORY hints (avoid dumping OCR/sign text). '
    '"note" MUST be your own concise English clause (≤20 words)—do not output the words '
    '"one short English sentence" literally.'
)


def approx_vision_tokens_pil(img: Image.Image) -> int:
    w, h = img.size
    rh, rw = smart_resize(
        h,
        w,
        factor=32,
        min_pixels=VENDOR_QWEN3VL_4B_MIN_PIXELS,
        max_pixels=VENDOR_QWEN3VL_4B_MAX_PIXELS,
    )
    return vision_grid_tokens(rh, rw, 16, 2)


def vlm_yolo_crops_coordinate_preamble_ko(
    orig_w: int,
    orig_h: int,
    *,
    context_max_side: int | None = None,
    context_resize_scale: float | None = None,
    crop_max_side: int,
    bboxes_xyxy_orig: list[tuple[int, int, int, int]],
) -> str:
    """Qwen YOLO+다중이미지 경로: 클래스명 없이 원본 크기·좌표·이미지 슬롯만 안내."""
    if context_resize_scale is not None:
        ctx_line = (
            f"이미지 #0: 전체 맥락(원본 대비 가로·세로 각 {context_resize_scale:g}배로 리사이즈한 저해상 전망)."
        )
    else:
        cms = context_max_side if context_max_side is not None else 960
        ctx_line = f"이미지 #0: 전체 맥락(긴 변 최대 {cms}px로 축소)."
    lines = [
        f"원본(픽셀) 크기: {orig_w}×{orig_h}.",
        ctx_line,
    ]
    crop_hint = (
        "VLM 입력은 원본에서 잘라낸 해상도 그대로."
        if crop_max_side <= 0
        else f"VLM 입력은 긴 변 최대 {crop_max_side}px로 맞춤."
    )
    if bboxes_xyxy_orig:
        lines.append("아래 #1~ 는 원본에서 잘라낸 영역(좌표만 기술, 클래스/점수는 사용하지 않음):")
        for i, (x1, y1, x2, y2) in enumerate(bboxes_xyxy_orig, start=1):
            lines.append(
                f"  · 이미지 #{i}: [x1,y1,x2,y2]=[{x1},{y1},{x2},{y2}] (원본 픽셀). {crop_hint}"
            )
    else:
        lines.append("추가 잘라내기 이미지 없음(탐지 없음 또는 크기 필터로 제외).")
    return "\n".join(lines)


_IMAGE_SUFFIX = frozenset({".jpg", ".jpeg", ".png"})


def list_frames(frames_dir: Path, max_frames: int) -> list[Path]:
    """
    연속 프레임 후보를 ``frames_dir`` 에서 고른다.

    1. 루트에 ``frame_*.jpg`` / ``*.jpg`` 등이 있으면(데모처럼 한 폴더에 평탄하게) 그 순서.
    2. 그렇지 않고 **바로 아래 하위 폴더**에만 이미지가 있으면(ShanghaiTech
       ``testing/frames/<시퀀스_id>/*.jpg``) **이름순 첫 시퀀스 폴더만** 열어
       그 안 이미지를 깊이 우선 정렬로 잘라 ``max_frames`` 장.
    3. 여전히 없으면 ``frames_dir`` 전체에서 이미지를 재귀 수집·정렬해 앞에서부터.
    """
    root = frames_dir.resolve()
    if not root.is_dir():
        return []

    def _is_image(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in _IMAGE_SUFFIX

    flat: list[Path] = []
    for pat in (
        "frame_*.jpg",
        "frame_*.jpeg",
        "frame_*.png",
        "*.jpg",
        "*.jpeg",
        "*.png",
    ):
        flat.extend(root.glob(pat))
    flat = sorted(set(flat))
    prefer = [p for p in flat if p.name.lower().startswith("frame_")]
    if prefer:
        return prefer[:max_frames]
    if flat:
        return flat[:max_frames]

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for sd in subdirs:
        nested = sorted(p for p in sd.rglob("*") if _is_image(p))
        if nested:
            return nested[:max_frames]

    all_deep = sorted(p for p in root.rglob("*") if _is_image(p))
    return all_deep[:max_frames]


def _project_path(p: Path) -> Path:
    """저장소 루트(``ROOT``) 기준으로 상대 경로를 해석한다. 절대 경로는 그대로 resolve."""
    e = p.expanduser()
    if e.is_absolute():
        return e.resolve()
    return (ROOT / e).resolve()


def _posix_relative_to_repo(p: Path) -> str:
    """표시·JSON 용: ``ROOT`` 기준 상대 posix (저장소 밖이면 절대 posix)."""
    r = _project_path(p)
    try:
        return r.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return r.as_posix().replace("\\", "/")


def resolve_input_frames(
    frames_dir: Path,
    max_frames: int,
    single_image: Path | None,
) -> list[Path]:
    """
    벤치·주간 실험용 입력 경로 리스트.

    ``single_image``가 주어지면 해당 파일 1장만 (HR-Bench/고해상도 단일 장면); 아니면
    ``list_frames(frames_dir, max_frames)`` (연속 프레임 — 평탄 폴더 또는
    ShanghaiTech 식 ``frames/<시퀀스>/*.jpg`` 중첩 구조).
    """
    if single_image is not None:
        p = _project_path(single_image)
        if not p.is_file():
            raise FileNotFoundError(f"이미지 파일 없음: {p}")
        return [p]
    return list_frames(_project_path(frames_dir), max_frames)


def _gated_ko_mse_skip_all(
    *,
    mse_val: float,
    mse_threshold: float,
    ref_frame: str,
    approx_qwen_only: int,
    approx_yolo_path: int,
) -> tuple[str, str, str]:
    """
    (qwen_only.reply, yolo_qwen.reply, frame_gating.summary_ko)
    *흑백 썸네일* 평균 차이: 작을수록 이전 프레임과 닮음(대략 0~255 스케일).
    """
    q = (
        "[VLM 생략 — Qwen-only 경로]\n"
        f"· 이전 화면과 너무 비슷하다고 판단해, 이번에 큰 AI(Qwen)에 요청을 보내지 않았습니다.\n"
        f"· 측정값 흑백·유사도(작은 썸네일 기준): {mse_val:.3f} — 기준 {mse_threshold} 미만이면"
        f" '거의 같다'고 봅니다(숫자가 작을수록 닮음).\n"
        f"· 이번에 서버에 사용된 VLM 토큰(입력): 0 (호출 없음)\n"
        f"· 참고(만약 1장만 보냈다면의 비전 토큰 추정): 약 {approx_qwen_only} (규칙 기반 추정)\n"
        f"· 직전 풀 추론이 있었던 프레임: {ref_frame}"
    )
    y = (
        "[VLM+YOLO 경로 — 둘 다 생략]\n"
        f"· 흑백·유사도 {mse_val:.3f} < {mse_threshold} 이므로, 물체 찾기(YOLO)와 "
        f"크롭 포함 VLM을 이번 프레임에서는 돌리지 않았습니다.\n"
        f"· 이번에 사용된 VLM 토큰(입력): 0 (호출 없음)\n"
        f"· 참고: 이전 풀 추론이 썼을 법한 크롭 포함 비전 토큰(추정): 약 {approx_yolo_path}\n"
        f"· 직전 풀 추론이 있었던 프레임: {ref_frame}"
    )
    sk = f"흑백 유사 {mse_val:.2f} < 기준 {mse_threshold} → YOLO·VLM 전부 생략"
    return (q, y, sk)


def _gated_ko_vlm_only_skip(
    *,
    mse_val: float | None,
    mse_threshold: float,
    ref_frame: str,
    fp_sig: str,
    t_yolo: float,
    n_det: int,
    yolo_summary: str,
    approx_qwen_only: int,
    approx_yolo_path: int,
) -> tuple[str, str, str]:
    mse_line = (
        f"· 참고(흑백·유사도): {mse_val:.3f} (기준 {mse_threshold}는 '전부 생략'에만 쓰임)\n"
        if mse_val is not None
        else ""
    )
    q = (
        "[VLM 생략 — Qwen-only / 첫·두 번 모두]\n"
        f"· 물체 찾기(YOLO)는 이 프레임에서 실행했고, 찾은 개수 {n_det} — 요약: {yolo_summary}\n"
        f"· 탐지 서명(개수|클래스ID들): {fp_sig} — 이 값이 직전 풀 추론 프레임({ref_frame})과 같다고 보아, "
        f"1번·2번 VLM(Qwen) 호출을 건너뛰었습니다.\n"
        f"· YOLO만 걸린 시간(대략): {t_yolo:.2f}초\n" + mse_line + "\n"
        f"· 이번 VLM 토큰(입력): 0 (VLM 미호출)\n"
        f"· 이번 그림(1장) 비전 토큰 추정: 약 {approx_qwen_only} / 크롭까지면 약 {approx_yolo_path}\n"
        f"· 직전 풀 추론이 있었던 프레임: {ref_frame}"
    )
    y = (
        "[크롭 포함 VLM 생략]\n"
        f"· YOLO는 수행({t_yolo:.2f}초, 탐지 {n_det}개, {yolo_summary}).\n"
        f"· 탐지 서명이 직전 풀 프레임({ref_frame})과 같아, VLM(설명)만 보내지 않음.\n"
        f"· 이 VLM 토큰(입력): 0\n"
        f"· 비전 토큰 추정(보냈다면): 약 {approx_yolo_path}\n"
        f"· 직전 풀 추론 프레임: {ref_frame}"
    )
    sk = f"탐지 서명 동일({fp_sig}) — VLM만 생략, YOLO {t_yolo:.2f}s"
    return (q, y, sk)


def _gated_ko_crop_count_vlm_skip(
    *,
    n_crops: int,
    n_det: int,
    gate_min: int,
    gate_max: int,
    reason: str,
    t_yolo: float,
    yolo_summary: str,
    approx_qwen_only: int,
    approx_yolo_path: int,
) -> tuple[str, str, str]:
    """크롭 개수(전송 후보)가 범위 밖일 때 Qwen-only·YOLO+Qwen VLM 둘 다 생략 안내."""
    if reason == "below_min":
        band = f"전송 크롭 {n_crops}개 < 하한 {gate_min}개"
    elif reason == "above_max":
        band = f"전송 크롭 {n_crops}개 > 상한 {gate_max}개"
    else:
        band = f"전송 크롭 {n_crops}개 (사유={reason})"
    q = (
        "[VLM 생략 — 크롭 개수 게이트]\n"
        f"· YOLO는 수행했습니다({t_yolo:.2f}초, 탐지 {n_det}개).\n"
        f"· {band} 이라 이번 프레임에서는 Qwen-only·크롭 포함 Qwen 호출을 하지 않았습니다.\n"
        f"· 요약: {yolo_summary}\n"
        f"· 비전 토큰 추정(보냈다면): 전망만 약 {approx_qwen_only} / 크롭 포함 약 {approx_yolo_path}"
    )
    y = (
        "[YOLO+Qwen VLM 생략 — 크롭 개수 게이트]\n"
        f"· {band}\n"
        f"· YOLO {t_yolo:.2f}s, 탐지 {n_det}개, {yolo_summary}\n"
        f"· 비전 토큰(가정): 약 {approx_yolo_path}"
    )
    sk = f"크롭 개수 게이트({band}) — YOLO만 {t_yolo:.2f}s"
    return (q, y, sk)


def _gated_qwen_only_skipped_dict(
    *, reply: str, skip_reason: str, ref_frame: str
) -> dict[str, object]:
    return {
        "vlm_skipped": True,
        "skip_reason": skip_reason,
        "reused_from_full_frame": ref_frame,
        "seconds": 0.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "usage": None,
        "approx_image_tokens": None,
        "reply": reply,
        "reply_preview": reply[:500],
        "resources_after": None,
    }


def _gated_yolo_qwen_mse_all_skipped_dict(
    *, reply: str, skip_reason: str, ref_frame: str
) -> dict[str, object]:
    return {
        "vlm_skipped": True,
        "yolo_skipped": True,
        "skip_reason": skip_reason,
        "reused_from_full_frame": ref_frame,
        "yolo_seconds": 0.0,
        "vlm_seconds": 0.0,
        "total_seconds": 0.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "usage": None,
        "approx_image_tokens": None,
        "reply": reply,
        "reply_preview": reply[:500],
        "resources_after": None,
    }


def _gated_yolo_qwen_vlm_only_skipped_dict(
    *,
    reply: str,
    skip_reason: str,
    ref_frame: str,
    t_yolo: float,
) -> dict[str, object]:
    return {
        "vlm_skipped": True,
        "yolo_skipped": False,
        "skip_reason": skip_reason,
        "reused_from_full_frame": ref_frame,
        "yolo_seconds": round(t_yolo, 3),
        "vlm_seconds": 0.0,
        "total_seconds": round(t_yolo, 3),
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "usage": None,
        "approx_image_tokens": None,
        "reply": reply,
        "reply_preview": reply[:500],
        "resources_after": None,
    }


def _merge_parallel_stage1(*, smol_ok: bool, smol_text: str) -> str:
    """경량 VLM 1단계 텍스트만. YOLO 좌표는 ``vlm_yolo_crops_coordinate_preamble_ko`` 에만 넣는다."""
    if smol_ok:
        return f"[SmolVLM 병렬 응답] {smol_text}"
    return f"[SmolVLM 병렬 응답] (오류) {smol_text}"


def run_yolo_smol_parallel_qwen(
    *,
    client_qwen: OpenAI,
    model_qwen: str,
    client_small: OpenAI,
    model_small: str,
    frames: list[Path],
    user_prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
    stage1_prompt: str,
    skip_qwen_if_low: bool,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    yolo_vlm_budget: bool = True,
    context_resize_scale: float | None = None,
    yolo_class_ids: tuple[int, ...] | None = None,
    max_bbox_area_numerator: int = 1,
    max_bbox_area_denominator: int = 4,
    yolo_device: str = "cpu",
) -> list[dict]:
    """
    YOLO(Ultralytics)와 SmolVLM(llama-server)을 **같은 프레임**에 대해
    스레드 풀(최대 2 워커)로 **동시에** 돌린 뒤, **저해상도 전망 + 잘라내기(리사이즈) + Smol 텍스트**로
    Qwen 1회 호출.

    - YOLO 클래스·점수는 VLM 프롬프트에 넣지 않고, **원본 좌표**와 **크롭 이미지**만 사용.
    - 이미지 ``#0`` = ``context_max_side`` 전망, ``#1~`` = ``crop_max_side`` 로 맞춘 크롭.
    - Smol 쪽 서버 URL은 ``client_small`` 이 가리킨다(별도 ``llama-server``).

    **타이밍:** ``wall_parallel_seconds`` 는 두 future 가 모두 완료될 때까지의 **벽시계**이며
    이론상 ``max(t_yolo, t_smol)`` 에 수렴한다.
    """
    yolo = load_yolo(yolo_model_path)
    out: list[dict] = []
    n_frames = len(frames)

    for i, fp in enumerate(frames):
        print(
            f"[yolo+smol||+qwen] 프레임 {i + 1}/{n_frames}: {fp.name}",
            flush=True,
        )
        img = Image.open(fp).convert("RGB")
        orig_wh = img.size
        low = prep_context_image(
            img,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
        )
        low_url = pil_to_data_url(low)
        t_par0 = time.perf_counter()

        def _run_yolo() -> tuple[object, ...]:
            t0 = time.perf_counter()
            cms = None if context_resize_scale is not None else context_max_side
            crops, bboxes, summary, n_det, all_cls = run_yolo_crops(
                img,
                model=yolo,
                predict_source=fp,
                yolo_device=yolo_device,
                max_crops=max_crops,
                class_ids=yolo_class_ids,
                min_crop_short_side=min_crop_short_side,
                min_crop_area=min_crop_area,
                vlm_budget=yolo_vlm_budget,
                context_max_side=cms,
                context_resize_scale=context_resize_scale,
                max_bbox_area_numerator=max_bbox_area_numerator,
                max_bbox_area_denominator=max_bbox_area_denominator,
            )
            dt = time.perf_counter() - t0
            return crops, bboxes, summary, n_det, all_cls, dt

        def _run_smol() -> tuple[str, str, object | None, float]:
            t0 = time.perf_counter()
            try:
                txt, usage_s = chat_vlm_multi(
                    client=client_small,
                    model=model_small,
                    prompt=stage1_prompt,
                    image_data_urls=[low_url],
                    max_tokens=256,
                    content_image_first=True,
                )
                return "ok", txt, usage_s, time.perf_counter() - t0
            except Exception as e:
                return "err", repr(e), None, time.perf_counter() - t0

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_y = pool.submit(_run_yolo)
            f_s = pool.submit(_run_smol)
            y_pack = f_y.result()
            s_status, s_body, s_usage, t_smol = f_s.result()

        crops, bboxes, summary, n_det, all_cls, t_yolo = y_pack
        wall_parallel = time.perf_counter() - t_par0

        smol_ok = s_status == "ok"
        if smol_ok and s_usage is not None:
            stage1_usage_d = usage_to_dict(s_usage)
        else:
            stage1_usage_d = None

        stage1_merged = _merge_parallel_stage1(smol_ok=smol_ok, smol_text=s_body)
        preamble = vlm_yolo_crops_coordinate_preamble_ko(
            orig_w=orig_wh[0],
            orig_h=orig_wh[1],
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
            crop_max_side=crop_max_side,
            bboxes_xyxy_orig=bboxes,
        )
        if smol_ok:
            risk = parse_risk_from_stage1(s_body)
        else:
            risk = yolo_heuristic_risk(
                num_detections=n_det, class_ids_detected=all_cls
            )
        res_after_parallel = resource_snapshot(
            label="yolo_smol_parallel_after",
        )

        qwen_image_urls: list[str] = [low_url]
        crop_dims_q: list[list[int]] = []
        for c in crops:
            rc = prep_yolo_crop_for_vlm(c, crop_max_side)
            crop_dims_q.append([rc.width, rc.height])
            qwen_image_urls.append(pil_to_data_url(rc))

        row: dict = {
            "frame": str(fp.name),
            "yolo": {
                "n_detections": n_det,
                "summary": summary,
                "n_crops": len(crops),
                "crop_bboxes_xyxy_orig": [
                    [b[0], b[1], b[2], b[3]] for b in bboxes
                ],
            },
            "yolo_smol_parallel": {
                "yolo_seconds": round(t_yolo, 3),
                "smol_seconds": round(t_smol, 3),
                "wall_parallel_seconds": round(wall_parallel, 3),
            },
            "smol_text": (s_body if smol_ok else "")[:12000],
            "stage1_merged": stage1_merged[:12000],
            "smol_error": (None if smol_ok else s_body),
            "parsed_risk": risk,
            "stage1_usage": stage1_usage_d,
            "stage1_resources_after": res_after_parallel,
            "input_image": {
                "source_frame": fp.name,
                "path": str(fp.resolve()),
                "original_width": orig_wh[0],
                "original_height": orig_wh[1],
                "width": low.size[0],
                "height": low.size[1],
                "max_side_cap": context_max_side,
                "context_resize_scale": context_resize_scale,
                "crop_max_side_cap": crop_max_side,
                "qwen_n_images": len(qwen_image_urls),
                "qwen_crop_dimensions_wh": crop_dims_q,
            },
        }
        if stage1_usage_d is not None:
            row["stage1_prompt_tokens"] = stage1_usage_d.get("prompt_tokens")
        row["qwen_called"] = True

        if skip_qwen_if_low and risk == "low":
            row["qwen_called"] = False
            row["qwen_reply"] = "(skipped: parallel stage1 risk=low)"
            row["qwen_usage"] = None
            row["qwen_seconds"] = None
            row["qwen_resources_after"] = None
            row["qwen_prompt_tokens"] = None
            row["total_seconds"] = round(wall_parallel, 3)
            out.append(row)
            continue

        q_prompt = (
            f"{preamble}\n\n"
            "다음은 **잘라내기 좌표(위)** 와 **경량 VLM(Smol) 응답** 입니다(탐지 클래스·점수는 사용하지 않음). "
            "이미지 #0=전망(저해상도), #1~=좌표에 따른 잘라내기(리사이즈). "
            "텍스트와 픽셀이 맞지 않으면 **이미지**를 기준으로 판단하세요.\n\n"
            f"{stage1_merged}\n\n"
            f"{user_prompt}"
        )
        t_q0 = time.perf_counter()
        reply, usage = chat_vlm_multi(
            client=client_qwen,
            model=model_qwen,
            prompt=q_prompt,
            image_data_urls=qwen_image_urls,
            max_tokens=max_tokens,
        )
        qu_d = usage_to_dict(usage)
        row["qwen_seconds"] = round(time.perf_counter() - t_q0, 3)
        row["qwen_prompt_tokens"] = getattr(usage, "prompt_tokens", None) if usage else None
        row["qwen_completion_tokens"] = (qu_d or {}).get("completion_tokens")
        row["qwen_total_tokens"] = (qu_d or {}).get("total_tokens")
        row["qwen_usage"] = qu_d
        row["qwen_reply"] = reply[:12000]
        row["qwen_resources_after"] = resource_snapshot(
            label="yolo_smol_parallel_qwen_after",
        )
        row["total_seconds"] = round(
            wall_parallel + row["qwen_seconds"],
            3,
        )
        out.append(row)

    return out


def parse_risk_from_stage1(text: str) -> str:
    text = text.strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            r = str(d.get("risk", "med")).lower()
            if r in ("low", "med", "high"):
                return r
        elif isinstance(d, str):
            r = d.strip().lower()
            if r in ("low", "med", "high"):
                return r
    except json.JSONDecodeError:
        pass
    m = re.search(r'"risk"\s*:\s*"([^"]+)"', text, re.I)
    if m:
        r = m.group(1).lower()
        if r in ("low", "med", "high"):
            return r
    return "med"


def bench(
    *,
    client: OpenAI,
    model: str,
    frames: list[Path],
    prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    yolo_vlm_budget: bool = True,
    context_resize_scale: float | None = None,
    yolo_class_ids: tuple[int, ...] | None = None,
    max_bbox_area_numerator: int = 1,
    max_bbox_area_denominator: int = 4,
    yolo_device: str = "cpu",
) -> list[dict]:
    """
    동일 프레임에 대해 **순서대로** (1) Qwen-only, (2) YOLO 수행 후 YOLO+Qwen.

    - Qwen-only: 컨텍스트 한 변 ``context_max_side`` 로 맞춘 1장만 전송.
    - YOLO+Qwen: 동일 **저해상도 전망 1장** + 크롭(최대 ``max_crops``, ``min_crop_*`` 통과분만)을
      ``crop_max_side`` 로 리사이즈해 **여러 장**을 한 요청에 보낸다. VLM 프롬프트는 **클래스명 없이**
      원본 크기·픽셀 좌표만 설명( ``vlm_yolo_crops_coordinate_preamble_ko`` ).

    벤치 JSON 은 ``experiment_metrics`` HTML 이 ``qwen_only`` / ``yolo_qwen`` 를 읽는다.
    """
    yolo = load_yolo(yolo_model_path)
    rows: list[dict] = []
    n_frames = len(frames)

    for i, fp in enumerate(frames):
        print(f"[bench] 프레임 {i + 1}/{n_frames}: {fp.name}", flush=True)
        img = Image.open(fp).convert("RGB")
        orig_w, orig_h = img.size
        ctx = prep_context_image(
            img,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
        )

        row: dict = {"frame": str(fp.name), "approx_vision_tokens": {}}

        t0 = time.perf_counter()
        url = pil_to_data_url(ctx)
        tok_ctx = approx_vision_tokens_pil(ctx)
        row["approx_vision_tokens"]["qwen_only"] = tok_ctx

        text_q, usage_q = chat_vlm_multi(
            client=client,
            model=model,
            prompt=prompt,
            image_data_urls=[url],
            max_tokens=max_tokens,
        )
        t_q = time.perf_counter() - t0
        pt_q = getattr(usage_q, "prompt_tokens", None) if usage_q else None

        t1 = time.perf_counter()
        cms = None if context_resize_scale is not None else context_max_side
        crops, bboxes, summary, n_det, all_cls = run_yolo_crops(
            img,
            model=yolo,
            predict_source=fp,
            yolo_device=yolo_device,
            max_crops=max_crops,
            class_ids=yolo_class_ids,
            min_crop_short_side=min_crop_short_side,
            min_crop_area=min_crop_area,
            vlm_budget=yolo_vlm_budget,
            context_max_side=cms,
            context_resize_scale=context_resize_scale,
            max_bbox_area_numerator=max_bbox_area_numerator,
            max_bbox_area_denominator=max_bbox_area_denominator,
        )
        t_yolo = time.perf_counter() - t1

        urls_y = [url]
        tok_sum = tok_ctx
        crop_dims: list[list[int]] = []
        for c in crops:
            rc = prep_yolo_crop_for_vlm(c, crop_max_side)
            crop_dims.append([rc.width, rc.height])
            urls_y.append(pil_to_data_url(rc))
            tok_sum += approx_vision_tokens_pil(rc)
        row["approx_vision_tokens"]["yolo_qwen_images"] = tok_sum
        row["yolo"] = {
            "n_detections": n_det,
            "summary": summary,
            "n_crops_sent": len(crops),
        }
        vlm_preamble = vlm_yolo_crops_coordinate_preamble_ko(
            orig_w=orig_w,
            orig_h=orig_h,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
            crop_max_side=crop_max_side,
            bboxes_xyxy_orig=bboxes,
        )
        row["images"] = {
            "source_path": str(fp.resolve()),
            "qwen_context": {
                "width": ctx.width,
                "height": ctx.height,
                "max_side_cap": context_max_side,
                "context_resize_scale": context_resize_scale,
                "original_width": orig_w,
                "original_height": orig_h,
            },
            "yolo_qwen": {
                "context_width": ctx.width,
                "context_height": ctx.height,
                "crop_max_side_cap": crop_max_side,
                "n_images_sent": len(urls_y),
                "crop_dimensions_wh": crop_dims,
                "crop_bboxes_xyxy_orig": [
                    [b[0], b[1], b[2], b[3]] for b in bboxes
                ],
            },
        }

        y_prompt = f"{vlm_preamble}\n\n{prompt}"
        t2 = time.perf_counter()
        text_y, usage_y = chat_vlm_multi(
            client=client,
            model=model,
            prompt=y_prompt,
            image_data_urls=urls_y,
            max_tokens=max_tokens,
        )
        t_y = time.perf_counter() - t2
        pt_y = getattr(usage_y, "prompt_tokens", None) if usage_y else None
        uq_d = usage_to_dict(usage_q)
        uy_d = usage_to_dict(usage_y)

        row["qwen_only"] = {
            "seconds": round(t_q, 3),
            "prompt_tokens": pt_q,
            "completion_tokens": (uq_d or {}).get("completion_tokens"),
            "total_tokens": (uq_d or {}).get("total_tokens"),
            "usage": uq_d,
            "approx_image_tokens": tok_ctx,
            "reply": text_q[:12000],
            "reply_preview": text_q[:500],
            "resources_after": resource_snapshot(label="bench_qwen_only_after"),
        }
        row["yolo_qwen"] = {
            "yolo_seconds": round(t_yolo, 3),
            "vlm_seconds": round(t_y, 3),
            "total_seconds": round(t_yolo + t_y, 3),
            "prompt_tokens": pt_y,
            "completion_tokens": (uy_d or {}).get("completion_tokens"),
            "total_tokens": (uy_d or {}).get("total_tokens"),
            "usage": uy_d,
            "approx_image_tokens": tok_sum,
            "reply": text_y[:12000],
            "reply_preview": text_y[:500],
            "resources_after": resource_snapshot(label="bench_yolo_qwen_after"),
        }
        rows.append(row)

    return rows


def bench_yolo_qwen_only(
    *,
    client: OpenAI,
    model: str,
    frames: list[Path],
    prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    yolo_vlm_budget: bool = True,
    context_resize_scale: float | None = None,
    yolo_class_ids: tuple[int, ...] | None = None,
    max_bbox_area_numerator: int = 1,
    max_bbox_area_denominator: int = 4,
    yolo_device: str = "cpu",
) -> list[dict]:
    """
    **베이스라인:** 프레임마다 YOLO 후 **크롭 포함 Qwen만** 호출한다.

    Qwen-only(전망 1장 단독)는 실행하지 않으며 ``qwen_only`` 필드는
    ``skip_reason=baseline_yolo_qwen_only`` 인 생략 메시지로 채운다.
    """
    yolo = load_yolo(yolo_model_path)
    rows: list[dict] = []
    n_frames = len(frames)

    for i, fp in enumerate(frames):
        print(f"[bench_yolo_qwen_only] 프레임 {i + 1}/{n_frames}: {fp.name}", flush=True)
        img = Image.open(fp).convert("RGB")
        orig_w, orig_h = img.size
        ctx = prep_context_image(
            img,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
        )
        row: dict = {"frame": str(fp.name), "approx_vision_tokens": {}}
        tok_ctx = approx_vision_tokens_pil(ctx)

        t1 = time.perf_counter()
        cms = None if context_resize_scale is not None else context_max_side
        crops, bboxes, summary, n_det, all_cls = run_yolo_crops(
            img,
            model=yolo,
            predict_source=fp,
            yolo_device=yolo_device,
            max_crops=max_crops,
            class_ids=yolo_class_ids,
            min_crop_short_side=min_crop_short_side,
            min_crop_area=min_crop_area,
            vlm_budget=yolo_vlm_budget,
            context_max_side=cms,
            context_resize_scale=context_resize_scale,
            max_bbox_area_numerator=max_bbox_area_numerator,
            max_bbox_area_denominator=max_bbox_area_denominator,
        )
        t_yolo = time.perf_counter() - t1

        url = pil_to_data_url(ctx)
        urls_y = [url]
        tok_sum = tok_ctx
        crop_dims: list[list[int]] = []
        for c in crops:
            rc = prep_yolo_crop_for_vlm(c, crop_max_side)
            crop_dims.append([rc.width, rc.height])
            urls_y.append(pil_to_data_url(rc))
            tok_sum += approx_vision_tokens_pil(rc)
        row["approx_vision_tokens"]["qwen_only"] = 0
        row["approx_vision_tokens"]["yolo_qwen_images"] = tok_sum
        row["yolo"] = {
            "n_detections": n_det,
            "summary": summary,
            "n_crops_sent": len(crops),
        }
        vlm_preamble = vlm_yolo_crops_coordinate_preamble_ko(
            orig_w=orig_w,
            orig_h=orig_h,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
            crop_max_side=crop_max_side,
            bboxes_xyxy_orig=bboxes,
        )
        row["images"] = {
            "source_path": str(fp.resolve()),
            "qwen_context": {
                "width": ctx.width,
                "height": ctx.height,
                "max_side_cap": context_max_side,
                "context_resize_scale": context_resize_scale,
                "original_width": orig_w,
                "original_height": orig_h,
            },
            "yolo_qwen": {
                "context_width": ctx.width,
                "context_height": ctx.height,
                "crop_max_side_cap": crop_max_side,
                "n_images_sent": len(urls_y),
                "crop_dimensions_wh": crop_dims,
                "crop_bboxes_xyxy_orig": [
                    [b[0], b[1], b[2], b[3]] for b in bboxes
                ],
            },
        }
        row["benchmark_branch"] = "yolo_qwen_every_frame"
        row["frame_gating"] = None
        row["qwen_only"] = _gated_qwen_only_skipped_dict(
            reply=(
                "[베이스라인] 매 프레임 **YOLO + 크롭 포함 Qwen** 만 실행합니다. "
                "Qwen-only(전망 1장 단독)는 비교를 위해 생략했습니다."
            ),
            skip_reason="baseline_yolo_qwen_only",
            ref_frame="",
        )
        y_prompt = f"{vlm_preamble}\n\n{prompt}"
        t2 = time.perf_counter()
        text_y, usage_y = chat_vlm_multi(
            client=client,
            model=model,
            prompt=y_prompt,
            image_data_urls=urls_y,
            max_tokens=max_tokens,
        )
        t_y = time.perf_counter() - t2
        pt_y = getattr(usage_y, "prompt_tokens", None) if usage_y else None
        uy_d = usage_to_dict(usage_y)
        row["yolo_qwen"] = {
            "yolo_seconds": round(t_yolo, 3),
            "vlm_seconds": round(t_y, 3),
            "total_seconds": round(t_yolo + t_y, 3),
            "prompt_tokens": pt_y,
            "completion_tokens": (uy_d or {}).get("completion_tokens"),
            "total_tokens": (uy_d or {}).get("total_tokens"),
            "usage": uy_d,
            "approx_image_tokens": tok_sum,
            "reply": text_y[:12000],
            "reply_preview": text_y[:500],
            "resources_after": resource_snapshot(label="bench_yolo_qwen_only_after"),
        }
        rows.append(row)

    return rows


def _yolo_qwen_vlm_called(row: dict) -> bool:
    """``yolo_qwen`` 경로에서 실제 multimodal Qwen 호출이 있었는지."""
    yq = row.get("yolo_qwen") or {}
    if yq.get("vlm_skipped") or yq.get("yolo_skipped"):
        return False
    return True


def compare_sequence_yolo_qwen_baseline_vs_gated(
    *,
    client: OpenAI,
    model: str,
    frames: list[Path],
    prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
    frame_gate: str,
    mse_threshold: float,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    use_crop_layout_gate: bool = False,
    crop_gate_max_shift: float = 0.02,
    gate_min_crops_for_vlm: int = 0,
    gate_max_crops_for_vlm: int = 0,
    yolo_vlm_budget: bool = True,
    context_resize_scale: float | None = None,
    yolo_class_ids: tuple[int, ...] | None = None,
    max_bbox_area_numerator: int = 1,
    max_bbox_area_denominator: int = 4,
    yolo_device: str = "cpu",
) -> dict:
    """
    베이스라인(매 프레임 YOLO+Qwen)과 게이트 파이프라인 결과를 한 묶음 dict 로 반환.
    """
    fg = (frame_gate or "off").strip().lower()
    if fg in ("", "off", "none"):
        fg = "yolo"
    t0 = time.perf_counter()
    baseline_rows = bench_yolo_qwen_only(
        client=client,
        model=model,
        frames=frames,
        prompt=prompt,
        max_tokens=max_tokens,
        context_max_side=context_max_side,
        crop_max_side=crop_max_side,
        yolo_model_path=yolo_model_path,
        max_crops=max_crops,
        min_crop_short_side=min_crop_short_side,
        min_crop_area=min_crop_area,
        yolo_vlm_budget=yolo_vlm_budget,
        context_resize_scale=context_resize_scale,
        yolo_class_ids=yolo_class_ids,
        max_bbox_area_numerator=max_bbox_area_numerator,
        max_bbox_area_denominator=max_bbox_area_denominator,
        yolo_device=yolo_device,
    )
    t1 = time.perf_counter()
    gated_rows = bench_with_frame_gating(
        client=client,
        model=model,
        frames=frames,
        prompt=prompt,
        max_tokens=max_tokens,
        context_max_side=context_max_side,
        crop_max_side=crop_max_side,
        yolo_model_path=yolo_model_path,
        max_crops=max_crops,
        frame_gate=fg,
        mse_threshold=mse_threshold,
        min_crop_short_side=min_crop_short_side,
        min_crop_area=min_crop_area,
        use_crop_layout_gate=use_crop_layout_gate,
        crop_gate_max_shift=crop_gate_max_shift,
        gate_min_crops_for_vlm=gate_min_crops_for_vlm,
        gate_max_crops_for_vlm=gate_max_crops_for_vlm,
        yolo_vlm_budget=yolo_vlm_budget,
        context_resize_scale=context_resize_scale,
        yolo_class_ids=yolo_class_ids,
        max_bbox_area_numerator=max_bbox_area_numerator,
        max_bbox_area_denominator=max_bbox_area_denominator,
        yolo_device=yolo_device,
    )
    t2 = time.perf_counter()
    n = len(frames)
    b_vlm = sum(1 for r in baseline_rows if _yolo_qwen_vlm_called(r))
    g_vlm = sum(1 for r in gated_rows if _yolo_qwen_vlm_called(r))
    g_q_only = sum(
        1
        for r in gated_rows
        if not (r.get("qwen_only") or {}).get("vlm_skipped", False)
    )

    def _yq_pt_ct_tt(r: dict) -> tuple[int | None, int | None, int | None]:
        yq = r.get("yolo_qwen") or {}
        return (
            yq.get("prompt_tokens"),
            yq.get("completion_tokens"),
            yq.get("total_tokens"),
        )

    def _qo_pt_ct_tt(r: dict) -> tuple[int | None, int | None, int | None]:
        qo = r.get("qwen_only") or {}
        return (
            qo.get("prompt_tokens"),
            qo.get("completion_tokens"),
            qo.get("total_tokens"),
        )

    b_pt = b_ct = b_tt = 0
    for r in baseline_rows:
        if not _yolo_qwen_vlm_called(r):
            continue
        pt, ct, tt = _yq_pt_ct_tt(r)
        if pt is not None:
            b_pt += int(pt)
        if ct is not None:
            b_ct += int(ct)
        if tt is not None:
            b_tt += int(tt)
    b_y_sec = sum(
        float((r.get("yolo_qwen") or {}).get("total_seconds") or 0)
        for r in baseline_rows
    )

    gq_pt = gq_ct = gq_tt = 0
    gq_sec = 0.0
    for r in gated_rows:
        qo = r.get("qwen_only") or {}
        if qo.get("vlm_skipped"):
            continue
        gq_sec += float(qo.get("seconds") or 0)
        pt, ct, tt = _qo_pt_ct_tt(r)
        if pt is not None:
            gq_pt += int(pt)
        if ct is not None:
            gq_ct += int(ct)
        if tt is not None:
            gq_tt += int(tt)

    gy_pt = gy_ct = gy_tt = 0
    gy_sec = 0.0
    for r in gated_rows:
        yq = r.get("yolo_qwen") or {}
        gy_sec += float(yq.get("total_seconds") or 0)
        if not _yolo_qwen_vlm_called(r):
            continue
        pt, ct, tt = _yq_pt_ct_tt(r)
        if pt is not None:
            gy_pt += int(pt)
        if ct is not None:
            gy_ct += int(ct)
        if tt is not None:
            gy_tt += int(tt)

    return {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "frame_gate": fg,
            "seconds_baseline": round(t1 - t0, 3),
            "seconds_gated": round(t2 - t1, 3),
            "seconds_total": round(t2 - t0, 3),
        },
        "summary": {
            "frames": n,
            "baseline_yolo_qwen_vlm_calls": b_vlm,
            "gated_yolo_qwen_vlm_calls": g_vlm,
            "gated_qwen_only_vlm_calls": g_q_only,
            "saved_yolo_qwen_vlm_calls_vs_baseline": b_vlm - g_vlm,
            "baseline_yolo_qwen_wall_seconds_sum": round(b_y_sec, 3),
            "gated_qwen_only_wall_seconds_sum": round(gq_sec, 3),
            "gated_yolo_qwen_wall_seconds_sum": round(gy_sec, 3),
            "baseline_yolo_qwen_prompt_tokens_sum": b_pt,
            "baseline_yolo_qwen_completion_tokens_sum": b_ct,
            "baseline_yolo_qwen_total_tokens_sum": b_tt,
            "gated_qwen_only_prompt_tokens_sum": gq_pt,
            "gated_qwen_only_completion_tokens_sum": gq_ct,
            "gated_qwen_only_total_tokens_sum": gq_tt,
            "gated_yolo_qwen_prompt_tokens_sum": gy_pt,
            "gated_yolo_qwen_completion_tokens_sum": gy_ct,
            "gated_yolo_qwen_total_tokens_sum": gy_tt,
        },
        "baseline": baseline_rows,
        "gated": gated_rows,
    }


def bench_with_frame_gating(
    *,
    client: OpenAI,
    model: str,
    frames: list[Path],
    prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
    frame_gate: str,
    mse_threshold: float,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    use_crop_layout_gate: bool = False,
    crop_gate_max_shift: float = 0.02,
    gate_min_crops_for_vlm: int = 0,
    gate_max_crops_for_vlm: int = 0,
    yolo_vlm_budget: bool = True,
    context_resize_scale: float | None = None,
    yolo_class_ids: tuple[int, ...] | None = None,
    max_bbox_area_numerator: int = 1,
    max_bbox_area_denominator: int = 4,
    yolo_device: str = "cpu",
) -> list[dict]:
    """
    :func:`bench` 와 동일한 스키마이며, 연속 프레임에서 중복을 줄이기 위한 **게이트**를 쓴다.

    **모드 요약 (``frame_gate``)**

    - **off** — 게이트 없음. 프레임마다 :func:`bench` 풀 추론.
    - **mse** — **휘도(MSE)만** 사용. 이전 썸과 ``mse_threshold`` 보다 **유사**하면
      YOLO·VLM **둘 다** 생략(``skip_all_mse``). MSE로 “다르다”고 보면 **매번 풀 추론**이며,
      *YOLO 탐지 서명으로 VLM만 생략하는 경로는 없다* (``yolo``/``mse_then_yolo`` 전용).
    - **yolo** — **MSE는 전혀 쓰지 않는다** (휘도 비교·``mse_threshold`` 무관). 매 프레임 YOLO는
      수행. 이전 프레임과 “장면이 같다”고 볼 때 Qwen(VLM)만 생략:
      ``use_crop_layout_gate=False`` → ``n_det`` + 탐지 **클래스 id 집합** 문자열(``yolo_fingerprint``)이
      같으면 스킵; ``True`` → **크롭 개수+박스 중심**이 ``crop_gate_max_shift`` 안에서 안정이면
      스킵(클래스는 비교에 안 씀).
    - **mse_then_yolo** — ① MSE로 전부 생략 가능, ② 아니면 YOLO 후 서명/크롭게이트로 VLM만 생략.

    ``gate_min_crops_for_vlm`` / ``gate_max_crops_for_vlm`` 가 0보다 크면, 필터 후 **전송 크롭 개수**가
    범위를 벗어날 때 Qwen-only·YOLO+Qwen VLM을 둘 다 호출하지 않는다(YOLO는 수행).

    **첫 프레임**은 항상 풀 파이프라인.
    """
    mode = (frame_gate or "off").strip().lower()
    if mode in ("", "off", "none"):
        return bench(
            client=client,
            model=model,
            frames=frames,
            prompt=prompt,
            max_tokens=max_tokens,
            context_max_side=context_max_side,
            crop_max_side=crop_max_side,
            yolo_model_path=yolo_model_path,
            max_crops=max_crops,
            min_crop_short_side=min_crop_short_side,
            min_crop_area=min_crop_area,
            yolo_vlm_budget=yolo_vlm_budget,
            context_resize_scale=context_resize_scale,
            yolo_class_ids=yolo_class_ids,
            max_bbox_area_numerator=max_bbox_area_numerator,
            max_bbox_area_denominator=max_bbox_area_denominator,
            yolo_device=yolo_device,
        )
    if mode not in ("mse", "yolo", "mse_then_yolo"):
        raise ValueError(
            f"frame_gate 는 off|mse|yolo|mse_then_yolo 인데: {frame_gate!r}"
        )

    yolo = load_yolo(yolo_model_path)
    rows: list[dict] = []
    n_frames = len(frames)
    prev_thumb = None
    prev_yolo_fp: str | None = None
    prev_bboxes: list[tuple[int, int, int, int]] | None = None
    prev_wh: tuple[int, int] | None = None
    last_full_row: dict | None = None

    for i, fp in enumerate(frames):
        t_frame0 = time.perf_counter()
        print(
            f"[bench+gate={mode}] 프레임 {i + 1}/{n_frames}: {fp.name}",
            flush=True,
        )
        img = Image.open(fp).convert("RGB")
        orig_w, orig_h = img.size
        ctx = prep_context_image(
            img,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
        )
        cur_thumb = luma_thumb(ctx)
        timings_ms: dict[str, float] = {
            "read_resize_ms": round(
                (time.perf_counter() - t_frame0) * 1000, 3
            ),
        }
        mse_val: float | None = None
        t_mse0 = time.perf_counter()
        if i > 0 and prev_thumb is not None and mode in ("mse", "mse_then_yolo"):
            mse_val = mean_abs_diff_luma(prev_thumb, cur_thumb)
        timings_ms["mse_ms"] = round((time.perf_counter() - t_mse0) * 1000, 3)

        skip_all_mse = bool(
            i > 0
            and mse_val is not None
            and mse_val < mse_threshold
            and mode in ("mse", "mse_then_yolo")
        )
        if skip_all_mse and last_full_row is not None:
            ref_name = str(last_full_row.get("frame", ""))
            av_prev = last_full_row.get("approx_vision_tokens", {}) or {}
            approx_y = int(
                av_prev.get("yolo_qwen_images")
                or av_prev.get("qwen_only")
                or 0
            )
            approx_q = int(approx_vision_tokens_pil(ctx))
            q_msg, y_msg, sko = _gated_ko_mse_skip_all(
                mse_val=float(mse_val),
                mse_threshold=mse_threshold,
                ref_frame=ref_name,
                approx_qwen_only=approx_q,
                approx_yolo_path=approx_y,
            )
            t_end = time.perf_counter()
            timings_ms["qwen_only_ms"] = 0.0
            timings_ms["yolo_qwen_ms"] = 0.0
            timings_ms["total_ms"] = round((t_end - t_frame0) * 1000, 3)
            y_img: dict = {
                "source_path": str(fp.resolve()),
                "qwen_context": {
                    "width": ctx.width,
                    "height": ctx.height,
                    "max_side_cap": context_max_side,
                },
            }
            if "yolo_qwen" in (last_full_row.get("images") or {}):
                y_img["yolo_qwen"] = last_full_row["images"]["yolo_qwen"]
            rows.append(
                {
                    "frame": str(fp.name),
                    "approx_vision_tokens": {
                        "qwen_only": approx_q,
                        "yolo_qwen_images": approx_y,
                    },
                    "yolo": last_full_row.get("yolo", {}),
                    "frame_gating": {
                        "mode": mode,
                        "path": "skip_all_mse",
                        "mse": mse_val,
                        "yolo_fingerprint": None,
                        "summary_ko": sko,
                        "reference_full_inference_frame": ref_name,
                        "api_vlm_input_tokens": 0,
                    },
                    "timings_ms": timings_ms,
                    "qwen_only": _gated_qwen_only_skipped_dict(
                        reply=q_msg,
                        skip_reason="skip_all_mse",
                        ref_frame=ref_name,
                    ),
                    "yolo_qwen": _gated_yolo_qwen_mse_all_skipped_dict(
                        reply=y_msg,
                        skip_reason="skip_all_mse",
                        ref_frame=ref_name,
                    ),
                    "images": y_img,
                }
            )
            prev_thumb = cur_thumb
            continue

        t_y0 = time.perf_counter()
        cms = None if context_resize_scale is not None else context_max_side
        crops, bboxes, summary, n_det, all_cls = run_yolo_crops(
            img,
            model=yolo,
            predict_source=fp,
            yolo_device=yolo_device,
            max_crops=max_crops,
            class_ids=yolo_class_ids,
            min_crop_short_side=min_crop_short_side,
            min_crop_area=min_crop_area,
            vlm_budget=yolo_vlm_budget,
            context_max_side=cms,
            context_resize_scale=context_resize_scale,
            max_bbox_area_numerator=max_bbox_area_numerator,
            max_bbox_area_denominator=max_bbox_area_denominator,
        )
        t_yolo = time.perf_counter() - t_y0
        fp_sig = yolo_fingerprint(n_det, all_cls)
        crop_fp = crop_gate_fingerprint(orig_w, orig_h, bboxes)
        timings_ms["yolo_ms"] = round(t_yolo * 1000, 3)
        n_crops_sent = len(crops)
        cc_reason = ""
        if gate_min_crops_for_vlm > 0 and n_crops_sent < gate_min_crops_for_vlm:
            cc_reason = "below_min"
        elif gate_max_crops_for_vlm > 0 and n_crops_sent > gate_max_crops_for_vlm:
            cc_reason = "above_max"
        if cc_reason:
            tok_ctx_cc = approx_vision_tokens_pil(ctx)
            tok_sum_cc = tok_ctx_cc
            crop_dims_cc: list[list[int]] = []
            for c in crops:
                rc = prep_yolo_crop_for_vlm(c, crop_max_side)
                crop_dims_cc.append([rc.width, rc.height])
                tok_sum_cc += approx_vision_tokens_pil(rc)
            q_msg_cc, y_msg_cc, sko_cc = _gated_ko_crop_count_vlm_skip(
                n_crops=n_crops_sent,
                n_det=n_det,
                gate_min=gate_min_crops_for_vlm,
                gate_max=gate_max_crops_for_vlm,
                reason=cc_reason,
                t_yolo=t_yolo,
                yolo_summary=summary,
                approx_qwen_only=int(tok_ctx_cc),
                approx_yolo_path=int(tok_sum_cc),
            )
            t_end_cc = time.perf_counter()
            timings_ms["qwen_only_ms"] = 0.0
            timings_ms["yolo_qwen_ms"] = 0.0
            timings_ms["total_ms"] = round((t_end_cc - t_frame0) * 1000, 3)
            n_urls_cc = 1 + len(crops)
            rows.append(
                {
                    "frame": str(fp.name),
                    "approx_vision_tokens": {
                        "qwen_only": int(tok_ctx_cc),
                        "yolo_qwen_images": int(tok_sum_cc),
                    },
                    "yolo": {
                        "n_detections": n_det,
                        "summary": summary,
                        "n_crops_sent": n_crops_sent,
                    },
                    "images": {
                        "source_path": str(fp.resolve()),
                        "qwen_context": {
                            "width": ctx.width,
                            "height": ctx.height,
                            "max_side_cap": context_max_side,
                            "original_width": orig_w,
                            "original_height": orig_h,
                        },
                        "yolo_qwen": {
                            "context_width": ctx.width,
                            "context_height": ctx.height,
                            "crop_max_side_cap": crop_max_side,
                            "n_images_sent": n_urls_cc,
                            "crop_dimensions_wh": crop_dims_cc,
                            "crop_bboxes_xyxy_orig": [
                                [b[0], b[1], b[2], b[3]] for b in bboxes
                            ],
                        },
                    },
                    "frame_gating": {
                        "mode": mode,
                        "path": "skip_vlm_crop_count_gate",
                        "crop_count_gate": cc_reason,
                        "gate_min_crops_for_vlm": gate_min_crops_for_vlm,
                        "gate_max_crops_for_vlm": gate_max_crops_for_vlm,
                        "mse": mse_val,
                        "yolo_fingerprint": fp_sig,
                        "crop_gate_fingerprint": crop_fp,
                        "use_crop_layout_gate": use_crop_layout_gate,
                        "crop_gate_max_shift": crop_gate_max_shift,
                        "summary_ko": sko_cc,
                        "reference_full_inference_frame": "",
                        "api_vlm_input_tokens": 0,
                    },
                    "timings_ms": timings_ms,
                    "qwen_only": _gated_qwen_only_skipped_dict(
                        reply=q_msg_cc,
                        skip_reason="skip_vlm_crop_count_gate",
                        ref_frame="",
                    ),
                    "yolo_qwen": _gated_yolo_qwen_vlm_only_skipped_dict(
                        reply=y_msg_cc,
                        skip_reason="skip_vlm_crop_count_gate",
                        ref_frame="",
                        t_yolo=t_yolo,
                    ),
                }
            )
            prev_yolo_fp = fp_sig
            prev_bboxes = list(bboxes)
            prev_wh = (orig_w, orig_h)
            prev_thumb = cur_thumb
            continue
        if use_crop_layout_gate:
            layout_ok = bool(
                prev_bboxes is not None
                and prev_wh is not None
                and last_full_row is not None
                and is_crop_layout_stable(
                    prev_bboxes,
                    bboxes,
                    prev_wh[0],
                    prev_wh[1],
                    orig_w,
                    orig_h,
                    max_norm_center_shift=crop_gate_max_shift,
                )
            )
            skip_vlm = bool(
                i > 0
                and mode in ("yolo", "mse_then_yolo")
                and layout_ok
            )
        else:
            skip_vlm = bool(
                i > 0
                and mode in ("yolo", "mse_then_yolo")
                and prev_yolo_fp is not None
                and fp_sig == prev_yolo_fp
                and last_full_row is not None
            )
        if skip_vlm:
            tok_ctx = approx_vision_tokens_pil(ctx)
            crop_dims: list[list[int]] = []
            tok_sum = tok_ctx
            for c in crops:
                rc = prep_yolo_crop_for_vlm(c, crop_max_side)
                crop_dims.append([rc.width, rc.height])
                tok_sum += approx_vision_tokens_pil(rc)
            ref_name = str(last_full_row.get("frame", ""))
            mse_v = mse_val if mse_val is not None else None
            q_msg, y_msg, sko = _gated_ko_vlm_only_skip(
                mse_val=mse_v,
                mse_threshold=mse_threshold,
                ref_frame=ref_name,
                fp_sig=fp_sig,
                t_yolo=t_yolo,
                n_det=n_det,
                yolo_summary=summary,
                approx_qwen_only=int(tok_ctx),
                approx_yolo_path=int(tok_sum),
            )
            t_end = time.perf_counter()
            timings_ms["qwen_only_ms"] = 0.0
            timings_ms["yolo_qwen_ms"] = 0.0
            timings_ms["total_ms"] = round((t_end - t_frame0) * 1000, 3)
            n_urls = 1 + len(crops)
            skip_path = (
                "skip_vlm_crop_layout"
                if use_crop_layout_gate
                else "skip_vlm_yolo_fp"
            )
            rows.append(
                {
                    "frame": str(fp.name),
                    "approx_vision_tokens": {
                        "qwen_only": int(tok_ctx),
                        "yolo_qwen_images": int(tok_sum),
                    },
                    "yolo": {
                        "n_detections": n_det,
                        "summary": summary,
                        "n_crops_sent": len(crops),
                    },
                    "images": {
                        "source_path": str(fp.resolve()),
                        "qwen_context": {
                            "width": ctx.width,
                            "height": ctx.height,
                            "max_side_cap": context_max_side,
                            "original_width": orig_w,
                            "original_height": orig_h,
                        },
                        "yolo_qwen": {
                            "context_width": ctx.width,
                            "context_height": ctx.height,
                            "crop_max_side_cap": crop_max_side,
                            "n_images_sent": n_urls,
                            "crop_dimensions_wh": crop_dims,
                            "crop_bboxes_xyxy_orig": [
                                [b[0], b[1], b[2], b[3]] for b in bboxes
                            ],
                        },
                    },
                    "frame_gating": {
                        "mode": mode,
                        "path": skip_path,
                        "mse": mse_val,
                        "yolo_fingerprint": fp_sig,
                        "crop_gate_fingerprint": crop_fp,
                        "use_crop_layout_gate": use_crop_layout_gate,
                        "crop_gate_max_shift": crop_gate_max_shift,
                        "summary_ko": sko,
                        "reference_full_inference_frame": ref_name,
                        "api_vlm_input_tokens": 0,
                    },
                    "timings_ms": timings_ms,
                    "qwen_only": _gated_qwen_only_skipped_dict(
                        reply=q_msg,
                        skip_reason=skip_path,
                        ref_frame=ref_name,
                    ),
                    "yolo_qwen": _gated_yolo_qwen_vlm_only_skipped_dict(
                        reply=y_msg,
                        skip_reason=skip_path,
                        ref_frame=ref_name,
                        t_yolo=t_yolo,
                    ),
                }
            )
            prev_yolo_fp = fp_sig
            prev_bboxes = list(bboxes)
            prev_wh = (orig_w, orig_h)
            prev_thumb = cur_thumb
            continue

        t0q = time.perf_counter()
        url = pil_to_data_url(ctx)
        tok_ctx = approx_vision_tokens_pil(ctx)
        text_q, usage_q = chat_vlm_multi(
            client=client,
            model=model,
            prompt=prompt,
            image_data_urls=[url],
            max_tokens=max_tokens,
        )
        t_q = time.perf_counter() - t0q
        pt_q = getattr(usage_q, "prompt_tokens", None) if usage_q else None
        uq_d = usage_to_dict(usage_q)
        t2q = time.perf_counter()
        urls_y = [url]
        tok_sum = tok_ctx
        crop_dims = []
        for c in crops:
            rc = prep_yolo_crop_for_vlm(c, crop_max_side)
            crop_dims.append([rc.width, rc.height])
            urls_y.append(pil_to_data_url(rc))
            tok_sum += approx_vision_tokens_pil(rc)
        vlm_preamble = vlm_yolo_crops_coordinate_preamble_ko(
            orig_w=orig_w,
            orig_h=orig_h,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
            crop_max_side=crop_max_side,
            bboxes_xyxy_orig=bboxes,
        )
        y_prompt = f"{vlm_preamble}\n\n{prompt}"
        text_y, usage_y = chat_vlm_multi(
            client=client,
            model=model,
            prompt=y_prompt,
            image_data_urls=urls_y,
            max_tokens=max_tokens,
        )
        t_y = time.perf_counter() - t2q
        pt_y = getattr(usage_y, "prompt_tokens", None) if usage_y else None
        uy_d = usage_to_dict(usage_y)
        timings_ms["qwen_only_ms"] = round(t_q * 1000, 3)
        timings_ms["yolo_qwen_ms"] = round(t_y * 1000, 3)
        timings_ms["total_ms"] = round((time.perf_counter() - t_frame0) * 1000, 3)
        full_row: dict = {
            "frame": str(fp.name),
            "approx_vision_tokens": {
                "qwen_only": tok_ctx,
                "yolo_qwen_images": tok_sum,
            },
            "yolo": {
                "n_detections": n_det,
                "summary": summary,
                "n_crops_sent": len(crops),
            },
            "images": {
                "source_path": str(fp.resolve()),
                "qwen_context": {
                    "width": ctx.width,
                    "height": ctx.height,
                    "max_side_cap": context_max_side,
                    "context_resize_scale": context_resize_scale,
                    "original_width": orig_w,
                    "original_height": orig_h,
                },
                "yolo_qwen": {
                    "context_width": ctx.width,
                    "context_height": ctx.height,
                    "crop_max_side_cap": crop_max_side,
                    "n_images_sent": len(urls_y),
                    "crop_dimensions_wh": crop_dims,
                    "crop_bboxes_xyxy_orig": [
                        [b[0], b[1], b[2], b[3]] for b in bboxes
                    ],
                },
            },
            "frame_gating": {
                "mode": mode,
                "path": "full",
                "mse": mse_val,
                "yolo_fingerprint": fp_sig,
                "crop_gate_fingerprint": crop_fp,
                "use_crop_layout_gate": use_crop_layout_gate,
                "crop_gate_max_shift": crop_gate_max_shift,
            },
            "timings_ms": timings_ms,
            "qwen_only": {
                "seconds": round(t_q, 3),
                "prompt_tokens": pt_q,
                "completion_tokens": (uq_d or {}).get("completion_tokens"),
                "total_tokens": (uq_d or {}).get("total_tokens"),
                "usage": uq_d,
                "approx_image_tokens": tok_ctx,
                "reply": text_q[:12000],
                "reply_preview": text_q[:500],
                "resources_after": resource_snapshot(
                    label="bench_qwen_only_after"
                ),
            },
            "yolo_qwen": {
                "yolo_seconds": round(t_yolo, 3),
                "vlm_seconds": round(t_y, 3),
                "total_seconds": round(t_yolo + t_y, 3),
                "prompt_tokens": pt_y,
                "completion_tokens": (uy_d or {}).get("completion_tokens"),
                "total_tokens": (uy_d or {}).get("total_tokens"),
                "usage": uy_d,
                "approx_image_tokens": tok_sum,
                "reply": text_y[:12000],
                "reply_preview": text_y[:500],
                "resources_after": resource_snapshot(
                    label="bench_yolo_qwen_after"
                ),
            },
        }
        last_full_row = copy.deepcopy(full_row)
        rows.append(full_row)
        prev_yolo_fp = fp_sig
        prev_bboxes = list(bboxes)
        prev_wh = (orig_w, orig_h)
        prev_thumb = cur_thumb

    return rows


def run_two_stage(
    *,
    client_qwen: OpenAI,
    model_qwen: str,
    client_small: OpenAI | None,
    model_small: str | None,
    frames: list[Path],
    user_prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
    stage1_prompt: str,
    skip_qwen_if_low: bool,
    precached_stage1: dict[str, dict] | None = None,
    min_crop_short_side: int = 0,
    min_crop_area: int = 0,
    yolo_vlm_budget: bool = True,
    context_resize_scale: float | None = None,
    yolo_class_ids: tuple[int, ...] | None = None,
    max_bbox_area_numerator: int = 1,
    max_bbox_area_denominator: int = 4,
    yolo_device: str = "cpu",
) -> list[dict]:
    """
    **Smol(또는 YOLO 폴백) → Qwen** 2단계. YOLO+Smol 병렬이 아님(대안은 ``run_yolo_smol_parallel_qwen``).

    - ``precached_stage1``: `run_week_experiments` 가 Phase2 에서 Smol 응답을 미리
      뽑아 두면(파일명→dict), 3단계 Qwen 기동 직전에 Smol 을 **다시 띄울 필요 없이**
      1단계 텍스트만 주입한다(VRAM/시간 절약).
    - Smol API 성공 시 ``stage1_source=small_vlm``; 그렇지 않으면 YOLO 휴리스틱 JSON.
    - ``skip_qwen_if_low``: 1단계 JSON 의 ``risk`` 가 ``low`` 이면 Qwen 비호출(엣지에서만
      쓰기 좋은 정책; 실서비스는 도메인에 맞게 조정).
    """
    yolo = load_yolo(yolo_model_path)
    out: list[dict] = []
    n_frames = len(frames)

    for i, fp in enumerate(frames):
        print(f"[two-stage] 프레임 {i + 1}/{n_frames}: {fp.name}", flush=True)
        img = Image.open(fp).convert("RGB")
        low = prep_context_image(
            img,
            context_max_side=context_max_side,
            context_resize_scale=context_resize_scale,
        )
        low_url = pil_to_data_url(low)

        t_s0 = time.perf_counter()
        stage1_source = "no_small_vlm"
        stage1_text = ""
        prior_err: str | None = None

        stage1_usage_d: dict | None = None
        stage1_res_after: dict | None = None
        stage1_img_meta: dict | None = None

        if precached_stage1 is not None and fp.name in precached_stage1:
            c = precached_stage1[fp.name]
            stage1_text = str(c["text"])
            stage1_source = "small_vlm"
            stage1_usage_d = c.get("usage")
            stage1_res_after = c.get("resources_after")
            stage1_img_meta = c.get("image_sent")
        elif client_small is not None and model_small:
            try:
                stage1_text, usage_s = chat_vlm_multi(
                    client=client_small,
                    model=model_small,
                    prompt=stage1_prompt,
                    image_data_urls=[low_url],
                    max_tokens=256,
                    content_image_first=True,
                )
                stage1_source = "small_vlm"
                stage1_usage_d = usage_to_dict(usage_s)
                stage1_res_after = resource_snapshot(label="two_stage_smol_after")
                stage1_img_meta = {
                    "source_frame": fp.name,
                    "path": str(fp.resolve()),
                    "width": low.size[0],
                    "height": low.size[1],
                    "context_max_side": context_max_side,
                    "context_resize_scale": context_resize_scale,
                }
            except Exception as e:
                prior_err = str(e)
                stage1_source = "small_vlm_error"

        if stage1_source != "small_vlm":
            cms = None if context_resize_scale is not None else context_max_side
            _, _, summary, n_det, all_cls = run_yolo_crops(
                img,
                model=yolo,
                predict_source=fp,
                yolo_device=yolo_device,
                max_crops=max_crops,
                class_ids=yolo_class_ids,
                min_crop_short_side=min_crop_short_side,
                min_crop_area=min_crop_area,
                vlm_budget=yolo_vlm_budget,
                context_max_side=cms,
                context_resize_scale=context_resize_scale,
                max_bbox_area_numerator=max_bbox_area_numerator,
                max_bbox_area_denominator=max_bbox_area_denominator,
            )
            risk = yolo_heuristic_risk(
                num_detections=n_det, class_ids_detected=all_cls
            )
            payload: dict = {
                "source": "yolo_fallback",
                "summary": summary,
                "risk": risk,
                "n_detections": n_det,
            }
            if prior_err is not None:
                payload["small_vlm_error"] = prior_err
            stage1_text = json.dumps(payload, ensure_ascii=False)
            stage1_source = "yolo_fallback"

        if precached_stage1 is not None and fp.name in precached_stage1:
            t_stage1 = float(precached_stage1[fp.name].get("seconds", 0.0))
        else:
            t_stage1 = time.perf_counter() - t_s0
        risk = parse_risk_from_stage1(stage1_text)

        row: dict = {
            "frame": str(fp.name),
            "stage1_source": stage1_source,
            "stage1_seconds": round(t_stage1, 3),
            "stage1_text": stage1_text[:12000],
            "parsed_risk": risk,
            "qwen_called": True,
            "input_image": {
                "source_frame": fp.name,
                "path": str(fp.resolve()),
                "width": low.size[0],
                "height": low.size[1],
                "max_side_cap": context_max_side,
                "context_resize_scale": context_resize_scale,
            },
        }
        if stage1_usage_d is not None:
            row["stage1_usage"] = stage1_usage_d
            row["stage1_prompt_tokens"] = stage1_usage_d.get("prompt_tokens")
        if stage1_res_after is not None:
            row["stage1_resources_after"] = stage1_res_after
        if stage1_img_meta is not None:
            row["stage1_image_sent"] = stage1_img_meta

        if skip_qwen_if_low and risk == "low":
            row["qwen_called"] = False
            row["qwen_reply"] = "(skipped: stage1 risk=low)"
            row["total_seconds"] = round(t_stage1, 3)
            out.append(row)
            continue

        q_prompt = (
            f"다음은 엣지(경량) 분석 결과입니다. 참고만 하고, 이미지를 직접 보고 판단해줘.\n"
            f"{stage1_text}\n\n"
            f"{user_prompt}"
        )
        t_q0 = time.perf_counter()
        reply, usage = chat_vlm_multi(
            client=client_qwen,
            model=model_qwen,
            prompt=q_prompt,
            image_data_urls=[low_url],
            max_tokens=max_tokens,
        )
        row["qwen_seconds"] = round(time.perf_counter() - t_q0, 3)
        qu_d = usage_to_dict(usage)
        row["qwen_prompt_tokens"] = getattr(usage, "prompt_tokens", None) if usage else None
        row["qwen_completion_tokens"] = (qu_d or {}).get("completion_tokens")
        row["qwen_total_tokens"] = (qu_d or {}).get("total_tokens")
        row["qwen_usage"] = qu_d
        row["qwen_reply"] = reply[:12000]
        row["qwen_resources_after"] = resource_snapshot(label="two_stage_qwen_after")
        row["total_seconds"] = round(t_stage1 + row["qwen_seconds"], 3)
        out.append(row)

    return out


def _is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower().strip("[]")
    return h in ("127.0.0.1", "localhost", "::1")


def _vlm_server_reachable(base: str, *, timeout_s: float = 1.5) -> bool:
    url = f"{base.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def _maybe_spawn_llama_for_experiment(
    args: argparse.Namespace,
    base_normalized: str,
) -> tuple[subprocess.Popen | None, object | None, str]:
    """
    ``--base-url`` (정규화됨) 이 루프백에서 살아 있지 않으면 llama-server 를 띄운다.

    Returns:
        (프로세스, 로그 핸들 또는 None, OpenAI base URL)
    """
    if args.no_spawn_llama:
        return None, None, base_normalized
    try:
        host, port = host_port_from_openai_base_url(base_normalized)
    except ValueError:
        return None, None, base_normalized
    if not _is_loopback_host(host):
        return None, None, base_normalized
    if _vlm_server_reachable(base_normalized):
        return None, None, base_normalized
    print(
        f"[experiment] Qwen VLM ({base_normalized}) 에 연결되지 않아 "
        "llama-server 를 자동 기동합니다…",
        flush=True,
    )
    proc, api_base, log_f = start_llama_server(
        llama_server=_project_path(Path(args.llama_server)),
        gguf=_project_path(Path(args.llama_gguf)),
        mmproj=_project_path(Path(args.llama_mmproj)),
        host=host,
        port=port,
        ngl=args.llama_ngl,
        flash_attn=args.llama_flash_attn,
        model=args.model,
        ctx_size=args.llama_ctx_size,
        log_file=_project_path(Path(args.llama_log_file)),
    )
    wait_for_server(api_base, args.server_timeout, child=proc)
    return proc, log_f, normalize_openai_base_url(api_base)


def _stop_spawned_llama(proc: subprocess.Popen | None, log_f: object | None) -> None:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_f is not None:
        try:
            log_f.close()
        except OSError:
            pass


def main() -> None:
    configure_stdio_utf8()
    p = argparse.ArgumentParser(
        description="VLM 파이프라인 실험: bench, sequence-compare, two-stage, parallel-yolo-smol",
    )
    p.add_argument(
        "mode",
        choices=("bench", "sequence-compare", "two-stage", "parallel-yolo-smol"),
        help=(
            "bench: Qwen-only+YOLO+Q 순차; sequence-compare: YOLO+Qwen 매프레임 vs 게이트; "
            "two-stage: Smol→Qwen; parallel-yolo-smol: (YOLO∥Smol)+Q, --small-vlm-url 필수"
        ),
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=Path(
            "data/datasets/shanghaitech/shanghaitech/testing/frames"
        ),
    )
    p.add_argument(
        "--single-image",
        type=Path,
        default=None,
        help=(
            "이미지 파일 1장만 사용(고해상도 단일 입력). 지정 시 디렉터리에서 "
            "프레임을 나열하지 않고 이 경로만 씀. 연속 프레임 모드와 동일한 bench·게이트 파이프라인."
        ),
    )
    p.add_argument("--max-frames", type=int, default=6)
    p.add_argument("--context-max-side", type=int, default=960)
    p.add_argument(
        "--context-resize-scale",
        type=float,
        default=0.5,
        help=(
            "저해상 전망: 가로·세로 각 이 비율로 리사이즈(YOLO 픽셀 예산·외부 스크립트와 동일). "
            "--context-by-long-edge 와 함께 쓰지 말 것."
        ),
    )
    p.add_argument(
        "--context-by-long-edge",
        action="store_true",
        help="전망을 비율 축소 대신 --context-max-side(긴 변 캡)만 사용",
    )
    p.add_argument(
        "--crop-max-side",
        type=int,
        default=0,
        help="크롭 VLM 입력 긴 변 상한. 0=원본 크롭 그대로(스크립트 기본)",
    )
    p.add_argument(
        "--min-crop-short-side",
        type=int,
        default=0,
        metavar="PX",
        help=(
            "YOLO가 잘라 줄 탐지 상자(원본 해상도)에서 가로·세로 중 '짧은 쪽' 길이(픽셀)가 "
            "이 값**미만**이면 너무 작은 박스로 보고 VLM에 보내지 않음. 0=이 규칙 끔 (CLI 플래그 이름 그대로 유지)"
        ),
    )
    p.add_argument(
        "--min-crop-area",
        type=int,
        default=0,
        metavar="PX2",
        help=(
            "탐지 상자 면적(원본에서 가로×세로, 픽셀²)이 이 값**미만**이면 VLM에 보내지 않음. "
            "0=이 규칙 끔. 작은 노이즈·먼지 박스 줄이는 용도"
        ),
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLAMA_OPENAI_BASE", "http://127.0.0.1:8765/v1"),
        help="Qwen llama-server OpenAI 베이스 (/v1 포함)",
    )
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-local"))
    p.add_argument("--model", default="qwen3-vl-4b-q8")
    p.add_argument("--prompt", default=DEFAULT_PROMPT_KO)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--yolo-model",
        default="yolo26n.pt",
        help="Ultralytics 가중치 (기본: 외부 프로파일 스크립트와 동일 계열)",
    )
    p.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="YOLO→VLM 크롭 상한. 0=예산·필터 후 전부(스크립트와 동일)",
    )
    p.add_argument(
        "--yolo-surveillance-classes-only",
        action="store_true",
        help="YOLO predict 에 COCO 부분 클래스만(공장 감시 프리셋). 기본은 전 클래스",
    )
    p.add_argument(
        "--yolo-max-bbox-area-num",
        type=int,
        default=1,
        metavar="N",
        help="박스 면적 상한 분자(원본×N/M 미만만 유지)",
    )
    p.add_argument(
        "--yolo-max-bbox-area-den",
        type=int,
        default=4,
        metavar="M",
        help="박스 면적 상한 분모(기본 1/4)",
    )
    p.add_argument(
        "--yolo-device",
        default="cpu",
        help='Ultralytics predict device (기본 cpu, llama-server GPU 와 분리). 예: cpu, cuda:0',
    )
    p.add_argument(
        "--disable-yolo-vlm-budget",
        action="store_true",
        help=(
            "YOLO 박스를 VLM에 넣기 전 픽셀 예산·중복·포함 필터를 끔 "
            "(기본: 켜짐 — 전망 픽셀+박스 면적 합이 원본 이하가 되도록 크롭 축소)"
        ),
    )
    p.add_argument(
        "--small-vlm-url",
        default=os.environ.get("SMALL_VLM_OPENAI_BASE", ""),
        help="경량 VLM llama-server 베이스 (예: http://127.0.0.1:8766/v1). 비우면 YOLO 대체만",
    )
    p.add_argument(
        "--small-vlm-model",
        default=os.environ.get("SMALL_VLM_MODEL", "small-vlm"),
        help="경량 서버의 -a 모델 별칭",
    )
    p.add_argument("--stage1-prompt", default=STAGE1_PROMPT_EN)
    p.add_argument(
        "--skip-qwen-if-low",
        action="store_true",
        help="two-stage: risk=low 이면 Qwen 호출 생략",
    )
    p.add_argument(
        "--frame-gate",
        default=os.environ.get("BENCH_FRAME_GATE", "off"),
        help=(
            "bench 프레임 게이트: off | mse | yolo | mse_then_yolo. "
            "mse* 는 휘도 썸(작을수록 유사)→mse·mse_then_yolo에서 "
            "임계 미만이면 YOLO·VLM 전부 생략. "
            "yolo 는 MSE 없이 YOLO만(매 프레임 YOLO) 후 "
            "탐지 서명(또는 --use-crop-layout-gate)으로 VLM만 생략. "
            "mse_then_yolo 는 둘 다."
        ),
    )
    p.add_argument(
        "--mse-threshold",
        type=float,
        default=2.5,
        help=(
            "휘도 썸 평균 절대차(0~255). "
            "frame_gate 가 mse 또는 mse_then_yolo 일 때만 사용: "
            "이 값**보다 작으면** (더 비슷하면) YOLO+큐웬 전부 스킵. "
            "frame_gate=yolo 일 때는 무시됨."
        ),
    )
    p.add_argument(
        "--use-crop-layout-gate",
        action="store_true",
        help=(
            "벤치+frame-gate yolo|mse_then_yolo: 탐지 '클래스 목록' 대신, "
            "필터 통과한 **크롭 개수**와 **박스 중심 위치**(화면마다 0~1로 맞춤)가 "
            "이전 프레임과 비슷할 때만 VLM 생략"
        ),
    )
    p.add_argument(
        "--crop-gate-max-shift",
        type=float,
        default=0.02,
        metavar="RATIO",
        help=(
            "위 --use-crop-layout-gate 켰을 때만 사용. 이전 프레임과 **박스 중심**이 "
            "화면 너비·높이에 대해 최대 **몇 %%만큼** 움직여도 '같은 자리'로 볼지(0.02≈2%%). "
            "값이 **작을수록** 조금만 움직여도 '다른 장면'이 되어 VLM을 **더 자주** 호출"
        ),
    )
    p.add_argument(
        "--gate-min-crops-for-vlm",
        type=int,
        default=0,
        metavar="N",
        help=(
            "frame-gate 가 mse|yolo|mse_then_yolo 일 때: 필터 후 VLM에 보낼 크롭 개수가 "
            "N 미만이면 Qwen-only·YOLO+Qwen VLM 모두 생략. 0=끔"
        ),
    )
    p.add_argument(
        "--gate-max-crops-for-vlm",
        type=int,
        default=0,
        metavar="N",
        help=(
            "위와 같으나 크롭 개수가 N 초과일 때 생략. 0=끔"
        ),
    )
    p.add_argument("--json-out", type=Path, default=None, help="결과 JSON 저장 경로")
    p.add_argument(
        "--compare-html-out",
        type=Path,
        default=None,
        help="sequence-compare: 요약 HTML 경로(미지정 시 --json-out 과 같은 stem 의 .html)",
    )
    p.add_argument(
        "--no-spawn-llama",
        action="store_true",
        help="루프백 Qwen --base-url 이 꺼져 있어도 llama-server 를 자동 기동하지 않음",
    )
    p.add_argument(
        "--llama-server",
        default="vendor/llama-cpp-win-cuda/llama-server.exe",
        help="자동 기동 시 llama-server 실행 파일(저장소 루트 기준 상대 가능)",
    )
    p.add_argument(
        "--llama-gguf",
        default="vendor/qwen3-vl-4b-q8-gguf/Qwen3VL-4B-Instruct-Q8_0.gguf",
    )
    p.add_argument(
        "--llama-mmproj",
        default="vendor/qwen3-vl-4b-q8-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
    )
    p.add_argument("--llama-ngl", type=int, default=99)
    p.add_argument(
        "--llama-flash-attn",
        choices=("on", "off", "auto"),
        default="on",
    )
    p.add_argument("--llama-ctx-size", type=int, default=8192)
    p.add_argument(
        "--server-timeout",
        type=float,
        default=180.0,
        help="자동 기동 후 llama-server 준비(예: /models) 최대 대기(초)",
    )
    p.add_argument(
        "--llama-log-file",
        type=Path,
        default=Path("vendor/llama-server-experiment.log"),
        help="자동 기동 시 llama-server stderr 로그 파일",
    )
    args = p.parse_args()
    yolo_vlm_budget = not args.disable_yolo_vlm_budget
    ctx_scale: float | None = None if args.context_by_long_edge else args.context_resize_scale
    yolo_cls = None
    if args.yolo_surveillance_classes_only:
        from qwen_vlm.vision.yolo import FACTORY_SURVEILLANCE_CLASS_IDS

        yolo_cls = FACTORY_SURVEILLANCE_CLASS_IDS

    spawn_llama_proc: subprocess.Popen | None = None
    spawn_llama_log = None
    try:
        base = normalize_openai_base_url(args.base_url)
        spawn_llama_proc, spawn_llama_log, base = _maybe_spawn_llama_for_experiment(
            args, base
        )
        client_qwen = OpenAI(base_url=base, api_key=args.api_key)

        try:
            frames = resolve_input_frames(
                args.frames_dir, args.max_frames, args.single_image
            )
        except FileNotFoundError as e:
            print(f"{e}", file=sys.stderr)
            sys.exit(1)
        if not frames:
            print(f"프레임 없음: {args.frames_dir}", file=sys.stderr)
            sys.exit(1)

        if args.mode == "sequence-compare":
            fg = (args.frame_gate or "off").strip().lower()
            if fg in ("", "off", "none"):
                fg = "yolo"
                print(
                    "[sequence-compare] frame_gate 가 off 이므로 게이트 분기는 yolo 로 둡니다.",
                    flush=True,
                )
            out = compare_sequence_yolo_qwen_baseline_vs_gated(
                client=client_qwen,
                model=args.model,
                frames=frames,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                context_max_side=args.context_max_side,
                crop_max_side=args.crop_max_side,
                yolo_model_path=args.yolo_model,
                max_crops=args.max_crops,
                frame_gate=fg,
                mse_threshold=args.mse_threshold,
                min_crop_short_side=args.min_crop_short_side,
                min_crop_area=args.min_crop_area,
                use_crop_layout_gate=args.use_crop_layout_gate,
                crop_gate_max_shift=args.crop_gate_max_shift,
                gate_min_crops_for_vlm=args.gate_min_crops_for_vlm,
                gate_max_crops_for_vlm=args.gate_max_crops_for_vlm,
                yolo_vlm_budget=yolo_vlm_budget,
                context_resize_scale=ctx_scale,
                yolo_class_ids=yolo_cls,
                max_bbox_area_numerator=args.yolo_max_bbox_area_num,
                max_bbox_area_denominator=args.yolo_max_bbox_area_den,
                yolo_device=args.yolo_device,
            )
            out.setdefault("meta", {})["frames_dir"] = _posix_relative_to_repo(
                args.frames_dir
            )
            print(json.dumps(out, ensure_ascii=False, indent=2))
            if args.json_out:
                args.json_out.write_text(
                    json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                html_p = args.compare_html_out
                if html_p is None:
                    html_p = args.json_out.with_suffix(".html")
                try:
                    write_sequence_compare_html(out, html_p)
                    print(f"[sequence-compare] HTML: {html_p}", flush=True)
                except OSError as e:
                    print(f"[sequence-compare] HTML 생략: {e}", flush=True)
            return
    
        if args.mode == "bench":
            if (args.frame_gate or "off").strip().lower() not in ("", "off", "none"):
                rows = bench_with_frame_gating(
                    client=client_qwen,
                    model=args.model,
                    frames=frames,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    context_max_side=args.context_max_side,
                    crop_max_side=args.crop_max_side,
                    yolo_model_path=args.yolo_model,
                    max_crops=args.max_crops,
                    frame_gate=args.frame_gate,
                    mse_threshold=args.mse_threshold,
                    min_crop_short_side=args.min_crop_short_side,
                    min_crop_area=args.min_crop_area,
                    use_crop_layout_gate=args.use_crop_layout_gate,
                    crop_gate_max_shift=args.crop_gate_max_shift,
                    gate_min_crops_for_vlm=args.gate_min_crops_for_vlm,
                    gate_max_crops_for_vlm=args.gate_max_crops_for_vlm,
                    yolo_vlm_budget=yolo_vlm_budget,
                    context_resize_scale=ctx_scale,
                    yolo_class_ids=yolo_cls,
                    max_bbox_area_numerator=args.yolo_max_bbox_area_num,
                    max_bbox_area_denominator=args.yolo_max_bbox_area_den,
                    yolo_device=args.yolo_device,
                )
            else:
                rows = bench(
                    client=client_qwen,
                    model=args.model,
                    frames=frames,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    context_max_side=args.context_max_side,
                    crop_max_side=args.crop_max_side,
                    yolo_model_path=args.yolo_model,
                    max_crops=args.max_crops,
                    min_crop_short_side=args.min_crop_short_side,
                    min_crop_area=args.min_crop_area,
                    yolo_vlm_budget=yolo_vlm_budget,
                    context_resize_scale=ctx_scale,
                    yolo_class_ids=yolo_cls,
                    max_bbox_area_numerator=args.yolo_max_bbox_area_num,
                    max_bbox_area_denominator=args.yolo_max_bbox_area_den,
                    yolo_device=args.yolo_device,
                )
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            if args.json_out:
                args.json_out.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if args.json_out and args.single_image and rows:
                onepage = args.json_out.with_name(args.json_out.stem + "_singleview.html")
                pl = {
                    "input_mode": "single_image",
                    "single_image": str(args.single_image.resolve()),
                    "user_prompt": args.prompt,
                    "generated_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime()
                    ),
                    "bench": rows,
                    "yolo_smol_parallel_qwen": [],
                    "two_stage_skip_low": [],
                    "frame_count": len(frames),
                }
                try:
                    write_single_image_experiment_html_from_payload(pl, onepage)
                    print(f"[bench] singleview: {onepage}", flush=True)
                except (OSError, ValueError) as e:
                    print(f"[bench] singleview 생략: {e}", flush=True)
            return
    
        if args.mode == "parallel-yolo-smol":
            small_url = (args.small_vlm_url or "").strip()
            if not small_url:
                print(
                    "parallel-yolo-smol 는 --small-vlm-url(경량 VLM llama-server) 이 필요합니다.",
                    file=sys.stderr,
                )
                sys.exit(2)
            b = normalize_openai_base_url(small_url)
            client_small = OpenAI(base_url=b, api_key=args.api_key)
            rows = run_yolo_smol_parallel_qwen(
                client_qwen=client_qwen,
                model_qwen=args.model,
                client_small=client_small,
                model_small=args.small_vlm_model,
                frames=frames,
                user_prompt=args.prompt,
                max_tokens=args.max_tokens,
                context_max_side=args.context_max_side,
                crop_max_side=args.crop_max_side,
                yolo_model_path=args.yolo_model,
                max_crops=args.max_crops,
                stage1_prompt=args.stage1_prompt,
                skip_qwen_if_low=args.skip_qwen_if_low,
                min_crop_short_side=args.min_crop_short_side,
                min_crop_area=args.min_crop_area,
                yolo_vlm_budget=yolo_vlm_budget,
                context_resize_scale=ctx_scale,
                yolo_class_ids=yolo_cls,
                max_bbox_area_numerator=args.yolo_max_bbox_area_num,
                max_bbox_area_denominator=args.yolo_max_bbox_area_den,
                yolo_device=args.yolo_device,
            )
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            if args.json_out:
                args.json_out.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            return
    
        small_url = (args.small_vlm_url or "").strip()
        client_small: OpenAI | None = None
        model_small: str | None = None
        if small_url:
            b = normalize_openai_base_url(small_url)
            client_small = OpenAI(base_url=b, api_key=args.api_key)
            model_small = args.small_vlm_model
    
        rows = run_two_stage(
            client_qwen=client_qwen,
            model_qwen=args.model,
            client_small=client_small,
            model_small=model_small,
            frames=frames,
            user_prompt=args.prompt,
            max_tokens=args.max_tokens,
            context_max_side=args.context_max_side,
            crop_max_side=args.crop_max_side,
            yolo_model_path=args.yolo_model,
            max_crops=args.max_crops,
            stage1_prompt=args.stage1_prompt,
            skip_qwen_if_low=args.skip_qwen_if_low,
            min_crop_short_side=args.min_crop_short_side,
            min_crop_area=args.min_crop_area,
            yolo_vlm_budget=yolo_vlm_budget,
            context_resize_scale=ctx_scale,
            yolo_class_ids=yolo_cls,
            max_bbox_area_numerator=args.yolo_max_bbox_area_num,
            max_bbox_area_denominator=args.yolo_max_bbox_area_den,
            yolo_device=args.yolo_device,
        )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        if args.json_out:
            args.json_out.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    
    
    finally:
        _stop_spawned_llama(spawn_llama_proc, spawn_llama_log)

if __name__ == "__main__":
    main()
