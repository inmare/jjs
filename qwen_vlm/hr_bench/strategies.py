"""HR-Bench용 4가지 입력 전략 (PIL·MCQ 기준). experiment_pipeline 유틸 재사용."""
from __future__ import annotations

import base64
import concurrent.futures
import copy
import io
import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI
from PIL import Image

from qwen_vlm.main import chat_vlm_multi, pil_to_data_url
from qwen_vlm.hr_bench.prompt_build import (
    HR_BENCH_SMOL_STAGE1_PROMPT_EN,
    compose_hr_bench_mc_with_supplementary_en,
    hr_bench_yolo_context_sentence_en,
    summarize_smol_for_hr_bench_mc_en,
)
from qwen_vlm.pipeline.experiment import (
    parse_risk_from_stage1,
    prep_context_image,
    prep_yolo_crop_for_vlm,
    yolo_heuristic_risk,
)
from qwen_vlm.reporting.experiment_metrics import resource_snapshot, usage_to_dict
from qwen_vlm.utils.image_resize import resize_max_long, resize_max_side, resize_uniform_scale
from qwen_vlm.hr_bench.single_llama import SmolSingleServerHooks
from qwen_vlm.vision.yolo import FACTORY_SURVEILLANCE_CLASS_IDS, load_yolo, run_yolo_crops

LogFn = Callable[[str], None]
_SMOL_STRATEGY_NAMES = frozenset({"yolo_smol_parallel", "yolo_smol_sequential"})


def noop_log(_: str) -> None:
    pass


def _jpeg_data_url_thumb(img: Image.Image, *, max_side: int = 900, quality: int = 82) -> str:
    rgb = img.convert("RGB").copy()
    thumb = resize_max_side(rgb, max_side) if max_side > 0 else rgb
    if thumb is rgb:
        thumb = thumb.copy()
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _yolo_budget_kwargs(full_img: Image.Image, cfg: HRBenchStrategyConfig) -> dict[str, Any]:
    """원본 크기 기준 면적·실제 VLM 픽셀 예산 필터(YOLO 캔버스 좌표)."""
    return {
        "original_full_image_size": (full_img.width, full_img.height),
        "vlm_overview_max_side_for_budget": cfg.yolo_overview_max_side,
        "crop_max_side_for_budget": cfg.crop_max_side,
    }


STRATEGIES_ALL = (
    "qwen_only",
    "yolo_lowres_crops",
    "yolo_smol_parallel",
    "yolo_smol_sequential",
)


@dataclass
class HRBenchStrategyConfig:
    # None 이면 전망은 context_max_side(긴 변 캡)만 사용.
    context_max_side: int = 960
    context_resize_scale: float | None = 0.5
    crop_max_side: int = 0
    max_crops: int = 0
    min_crop_short_side: int = 0
    min_crop_area: int = 0
    yolo_vlm_budget: bool = True
    yolo_model_path: str = "yolo26n.pt"
    yolo_surveillance_classes_only: bool = False
    yolo_max_bbox_area_numerator: int = 1
    yolo_max_bbox_area_denominator: int = 4
    yolo_device: str = "cpu"  # ultralytics predict device; cpu 권장(llama GPU 분리)
    # YOLO 전략 전용: VLM에 보낼 전망의 긴 변 최대 픽셀.
    # 0이면 ctx(context_resize_scale 적용본)를 그대로 사용.
    # > 0이면 원본에서 긴 변을 이 크기 이하로 축소한 전망(Smol 및 Qwen #0 동일 소스).
    # YOLO 탐지 및 크롭은 ctx 기준으로 유지.
    yolo_overview_max_side: int = 960
    # True면 qwen_only 전략에서 원본 이미지를 그대로 VLM에 전달(리사이즈 없음).
    # 실험 대조군 역할. qwen_image_max_long_side > 0 이면 그 값으로 캡.
    qwen_only_use_original: bool = True
    qwen_image_max_long_side: int = 0
    max_tokens: int = 8
    stage1_prompt: str = HR_BENCH_SMOL_STAGE1_PROMPT_EN
    smol_max_tokens: int = 256


def _yolo_run_common_kwargs(cfg: HRBenchStrategyConfig) -> dict[str, Any]:
    cms = None if cfg.context_resize_scale is not None else cfg.context_max_side
    return {
        "class_ids": FACTORY_SURVEILLANCE_CLASS_IDS
        if cfg.yolo_surveillance_classes_only
        else None,
        "context_max_side": cms,
        "context_resize_scale": cfg.context_resize_scale,
        "max_bbox_area_numerator": cfg.yolo_max_bbox_area_numerator,
        "max_bbox_area_denominator": cfg.yolo_max_bbox_area_denominator,
        "yolo_device": cfg.yolo_device,
    }


def _yolo_make_overview(
    img: Image.Image,
    ctx: Image.Image,
    yolo_overview_max_side: int,
) -> Image.Image:
    """YOLO 전략용 VLM 전망 이미지 반환.

    yolo_overview_max_side > 0이면 원본에서 긴 변을 이 픽셀 이하로 축소한 전망을 반환(Smol 및 Qwen 슬롯 #0 동일 소스).
    0이면 ctx(context_resize_scale 적용본)를 그대로 반환.
    """
    if yolo_overview_max_side > 0:
        return resize_max_side(img, yolo_overview_max_side)
    return ctx


def _scale_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[tuple[int, int, int, int]]:
    """bboxes를 src 좌표계에서 dst 좌표계로 선형 변환."""
    if src_w == dst_w and src_h == dst_h:
        return bboxes
    sx, sy = dst_w / src_w, dst_h / src_h
    return [
        (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
        for x1, y1, x2, y2 in bboxes
    ]


def _prep_qwen_only_image(img: Image.Image, cfg: HRBenchStrategyConfig) -> Image.Image:
    if cfg.qwen_image_max_long_side > 0:
        return resize_max_long(img, cfg.qwen_image_max_long_side)
    if cfg.qwen_only_use_original:
        return img
    if cfg.context_resize_scale is not None:
        return resize_uniform_scale(img, cfg.context_resize_scale)
    return resize_max_side(img, cfg.context_max_side)


def run_sample_qwen_only(
    *,
    img: Image.Image,
    mcq_prompt: str,
    client: OpenAI,
    model: str,
    cfg: HRBenchStrategyConfig,
    log: LogFn = noop_log,
) -> dict[str, Any]:
    vis = _prep_qwen_only_image(img, cfg)
    url = pil_to_data_url(vis)
    log(
        f"  [qwen_only] 이미지 {vis.size[0]}×{vis.size[1]} → Qwen 요청…",
    )
    t0 = time.perf_counter()
    raw, usage = chat_vlm_multi(
        client=client,
        model=model,
        prompt=mcq_prompt,
        image_data_urls=[url],
        max_tokens=cfg.max_tokens,
    )
    elapsed = time.perf_counter() - t0
    u = usage_to_dict(usage)
    res = resource_snapshot(label="hr_qwen_only_after")
    pt = getattr(usage, "prompt_tokens", None) if usage else None
    log(f"  [qwen_only] 완료 {elapsed:.2f}s prompt_tokens={pt}")
    dbg = {
        "thumbnail_original_jpeg_url": _jpeg_data_url_thumb(img),
        "qwen_sent_thumbnail_url": _jpeg_data_url_thumb(vis),
        "qwen_prompt": mcq_prompt,
        "qwen_sent_image_sizes": [list(vis.size)],
    }
    return {
        "raw_reply": raw,
        "seconds": round(elapsed, 3),
        "prompt_tokens": pt,
        "usage": u,
        "resources_after": res,
        "stage1_prompt_tokens": None,
        "wall_parallel_seconds": None,
        "yolo_seconds": None,
        "smol_seconds": None,
        "hr_debug": dbg,
    }


def run_sample_yolo_lowres_crops(
    *,
    img: Image.Image,
    mcq_prompt: str,
    client: OpenAI,
    model: str,
    cfg: HRBenchStrategyConfig,
    yolo: object,
    log: LogFn = noop_log,
) -> dict[str, Any]:
    ctx = prep_context_image(
        img,
        context_max_side=cfg.context_max_side,
        context_resize_scale=cfg.context_resize_scale,
    )
    ctx_w, ctx_h = ctx.size
    overview_img = _yolo_make_overview(img, ctx, cfg.yolo_overview_max_side)
    ov_w, ov_h = overview_img.size
    low_url = pil_to_data_url(overview_img)
    log(f"  [yolo+crops] 전망 {ov_w}×{ov_h} · YOLO(ctx {ctx_w}×{ctx_h} 기준)…")
    t_y0 = time.perf_counter()
    crops, bboxes, summary, n_det, all_cls = run_yolo_crops(
        ctx,
        model=yolo,
        max_crops=cfg.max_crops,
        min_crop_short_side=cfg.min_crop_short_side,
        min_crop_area=cfg.min_crop_area,
        vlm_budget=cfg.yolo_vlm_budget,
        **_yolo_run_common_kwargs(cfg),
        **_yolo_budget_kwargs(img, cfg),
    )
    t_yolo = time.perf_counter() - t_y0
    log(
        f"  [yolo+crops] 탐지 {n_det} · 크롭 전송 {len(crops)} "
        f"(vlm_budget={cfg.yolo_vlm_budget}) · YOLO {t_yolo:.2f}s",
    )
    urls_y = [low_url]
    for c in crops:
        rc = prep_yolo_crop_for_vlm(c, cfg.crop_max_side)
        urls_y.append(pil_to_data_url(rc))
    bboxes_ov = _scale_bboxes(bboxes, ctx_w, ctx_h, ov_w, ov_h)
    yolo_sentence_en = hr_bench_yolo_context_sentence_en(
        overview_w=ov_w,
        overview_h=ov_h,
        canvas_w=ctx_w,
        canvas_h=ctx_h,
        n_raw_detections=n_det,
        n_crops_sent=len(crops),
        bboxes_ov=bboxes_ov,
        crop_max_side=cfg.crop_max_side,
        context_resize_scale=cfg.context_resize_scale,
        context_max_side=cfg.context_max_side if cfg.context_resize_scale is None else None,
    )
    y_prompt = compose_hr_bench_mc_with_supplementary_en(
        mcq_prompt,
        yolo_context_sentence_en=yolo_sentence_en,
        compact_vlm_sentence_en=None,
    )
    log(f"  [yolo+crops] Qwen 다중이미지({len(urls_y)}장)…")
    t_q0 = time.perf_counter()
    raw, usage = chat_vlm_multi(
        client=client,
        model=model,
        prompt=y_prompt,
        image_data_urls=urls_y,
        max_tokens=cfg.max_tokens,
    )
    t_q = time.perf_counter() - t_q0
    u = usage_to_dict(usage)
    res = resource_snapshot(label="hr_yolo_crops_qwen_after")
    pt = getattr(usage, "prompt_tokens", None) if usage else None
    total = t_yolo + t_q
    log(f"  [yolo+crops] Qwen {t_q:.2f}s · 합계 {total:.2f}s prompt_tokens={pt}")
    crop_thumbs = [
        _jpeg_data_url_thumb(prep_yolo_crop_for_vlm(c, cfg.crop_max_side))
        for c in crops
    ]
    dbg = {
        "thumbnail_original_jpeg_url": _jpeg_data_url_thumb(img),
        "yolo_canvas_size_px": {"w": ctx_w, "h": ctx_h},
        "overview_thumbnail_url": _jpeg_data_url_thumb(overview_img),
        "overview_sent_size": [ov_w, ov_h],
        "crop_thumbnails_jpeg_urls": crop_thumbs,
        "yolo_bbox_xyxy_on_canvas": [list(b) for b in bboxes],
        "yolo_bbox_xyxy_overview": [list(b) for b in bboxes_ov],
        "yolo_summary": summary,
        "yolo_context_sentence_en": yolo_sentence_en,
        "qwen_prompt": y_prompt,
    }
    return {
        "raw_reply": raw,
        "seconds": round(total, 3),
        "prompt_tokens": pt,
        "usage": u,
        "resources_after": res,
        "yolo_seconds": round(t_yolo, 3),
        "vlm_seconds": round(t_q, 3),
        "n_detections": n_det,
        "n_crops_sent": len(crops),
        "stage1_prompt_tokens": None,
        "wall_parallel_seconds": None,
        "smol_seconds": None,
        "hr_debug": dbg,
    }


def run_sample_yolo_smol_parallel(
    *,
    img: Image.Image,
    mcq_prompt: str,
    client_qwen: OpenAI,
    model_qwen: str,
    client_small: OpenAI,
    model_small: str,
    cfg: HRBenchStrategyConfig,
    yolo: object,
    log: LogFn = noop_log,
    after_smol_start_qwen: Callable[[], OpenAI] | None = None,
    single_gpu_serial_yolo_smol: bool = False,
) -> dict[str, Any]:
    low = prep_context_image(
        img,
        context_max_side=cfg.context_max_side,
        context_resize_scale=cfg.context_resize_scale,
    )
    low_w, low_h = low.size
    overview_img = _yolo_make_overview(img, low, cfg.yolo_overview_max_side)
    low_wh = overview_img.size
    low_url = pil_to_data_url(overview_img)
    log(f"  [yolo∥smol] 전망 {low_wh[0]}×{low_wh[1]} · YOLO(ctx {low_w}×{low_h} 기준)…")
    t_par0 = time.perf_counter()

    def _run_yolo() -> tuple[Any, ...]:
        t0 = time.perf_counter()
        crops, bboxes, summary, n_det, all_cls = run_yolo_crops(
            low,
            model=yolo,
            max_crops=cfg.max_crops,
            min_crop_short_side=cfg.min_crop_short_side,
            min_crop_area=cfg.min_crop_area,
            vlm_budget=cfg.yolo_vlm_budget,
            **_yolo_run_common_kwargs(cfg),
            **_yolo_budget_kwargs(img, cfg),
        )
        dt = time.perf_counter() - t0
        return crops, bboxes, summary, n_det, all_cls, dt

    def _run_smol() -> tuple[str, str, object | None, float]:
        t0 = time.perf_counter()
        try:
            txt, usage_s = chat_vlm_multi(
                client=client_small,
                model=model_small,
                prompt=cfg.stage1_prompt,
                image_data_urls=[low_url],
                max_tokens=cfg.smol_max_tokens,
                content_image_first=True,
            )
            return "ok", txt, usage_s, time.perf_counter() - t0
        except Exception as e:
            return "err", repr(e), None, time.perf_counter() - t0

    if single_gpu_serial_yolo_smol:
        log("  [yolo∥smol→단일GPU] YOLO 후 Smol 순차 (Wall 병렬 없음)")
        y_pack = _run_yolo()
        s_status, s_body, s_usage, t_smol = _run_smol()
    else:
        log("  [yolo∥smol] 병렬 시작…")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_y = pool.submit(_run_yolo)
            f_s = pool.submit(_run_smol)
            y_pack = f_y.result()
            s_status, s_body, s_usage, t_smol = f_s.result()

    crops, bboxes, summary, n_det, all_cls, t_yolo = y_pack
    wall_parallel = time.perf_counter() - t_par0
    smol_ok = s_status == "ok"
    stage1_usage_d = usage_to_dict(s_usage) if smol_ok and s_usage else None
    st1_pt = (stage1_usage_d or {}).get("prompt_tokens") if stage1_usage_d else None
    log(
        f"  [yolo∥smol] 벽시계 {wall_parallel:.2f}s (YOLO {t_yolo:.2f}s · Smol {t_smol:.2f}s) "
        f"탐지 {n_det} 크롭 {len(crops)} smol_ok={smol_ok} smol_pt={st1_pt}",
    )

    bboxes_ov = _scale_bboxes(bboxes, low_w, low_h, low_wh[0], low_wh[1])
    yolo_sentence_en = hr_bench_yolo_context_sentence_en(
        overview_w=low_wh[0],
        overview_h=low_wh[1],
        canvas_w=low_w,
        canvas_h=low_h,
        n_raw_detections=n_det,
        n_crops_sent=len(crops),
        bboxes_ov=bboxes_ov,
        crop_max_side=cfg.crop_max_side,
        context_resize_scale=cfg.context_resize_scale,
        context_max_side=cfg.context_max_side if cfg.context_resize_scale is None else None,
    )
    smol_summary_en = summarize_smol_for_hr_bench_mc_en(s_body, ok=smol_ok)
    q_prompt = compose_hr_bench_mc_with_supplementary_en(
        mcq_prompt,
        yolo_context_sentence_en=yolo_sentence_en,
        compact_vlm_sentence_en=smol_summary_en,
    )
    risk = (
        parse_risk_from_stage1(s_body)
        if smol_ok
        else yolo_heuristic_risk(num_detections=n_det, class_ids_detected=all_cls)
    )
    res_after_parallel = resource_snapshot(label="hr_parallel_after_stage1")

    qwen_image_urls: list[str] = [low_url]
    for c in crops:
        rc = prep_yolo_crop_for_vlm(c, cfg.crop_max_side)
        qwen_image_urls.append(pil_to_data_url(rc))

    log(f"  [yolo∥smol] Qwen {len(qwen_image_urls)}장…")
    if after_smol_start_qwen is not None:
        client_qwen = after_smol_start_qwen()
    t_q0 = time.perf_counter()
    raw, usage = chat_vlm_multi(
        client=client_qwen,
        model=model_qwen,
        prompt=q_prompt,
        image_data_urls=qwen_image_urls,
        max_tokens=cfg.max_tokens,
    )
    t_q = time.perf_counter() - t_q0
    u = usage_to_dict(usage)
    res_q = resource_snapshot(label="hr_parallel_qwen_after")
    pt = getattr(usage, "prompt_tokens", None) if usage else None
    total = wall_parallel + t_q
    crop_thumbs = [
        _jpeg_data_url_thumb(prep_yolo_crop_for_vlm(c, cfg.crop_max_side))
        for c in crops
    ]
    dbg = {
        "thumbnail_original_jpeg_url": _jpeg_data_url_thumb(img),
        "yolo_canvas_size_px": {"w": low_w, "h": low_h},
        "overview_thumbnail_url": _jpeg_data_url_thumb(overview_img),
        "overview_sent_size": list(low_wh),
        "crop_thumbnails_jpeg_urls": crop_thumbs,
        "yolo_bbox_xyxy_on_canvas": [list(b) for b in bboxes],
        "yolo_bbox_xyxy_overview": [list(b) for b in bboxes_ov],
        "smol_prompt": cfg.stage1_prompt,
        "smol_reply_text": s_body or "",
        "smol_reply_preview": (
            (s_body or "")[:4000] + ("…" if smol_ok and len(s_body or "") > 4000 else "")
        ),
        "smol_ok": smol_ok,
        "yolo_context_sentence_en": yolo_sentence_en,
        "compact_vision_sentence_en": smol_summary_en,
        "qwen_prompt": q_prompt,
        "parsed_risk": risk,
    }

    log(f"  [yolo∥smol] Qwen {t_q:.2f}s · 총 {total:.2f}s prompt_tokens={pt}")
    return {
        "raw_reply": raw,
        "seconds": round(total, 3),
        "prompt_tokens": pt,
        "usage": u,
        "resources_after": res_q,
        "stage1_resources_after": res_after_parallel,
        "stage1_prompt_tokens": st1_pt,
        "stage1_usage": stage1_usage_d,
        "wall_parallel_seconds": round(wall_parallel, 3),
        "yolo_seconds": round(t_yolo, 3),
        "smol_seconds": round(t_smol, 3),
        "n_detections": n_det,
        "n_crops_sent": len(crops),
        "parsed_risk": risk,
        "smol_ok": smol_ok,
        "hr_debug": dbg,
    }


def run_sample_yolo_smol_sequential(
    *,
    img: Image.Image,
    mcq_prompt: str,
    client_qwen: OpenAI,
    model_qwen: str,
    client_small: OpenAI,
    model_small: str,
    cfg: HRBenchStrategyConfig,
    yolo: object,
    log: LogFn = noop_log,
    after_smol_start_qwen: Callable[[], OpenAI] | None = None,
) -> dict[str, Any]:
    low = prep_context_image(
        img,
        context_max_side=cfg.context_max_side,
        context_resize_scale=cfg.context_resize_scale,
    )
    low_w, low_h = low.size
    overview_img = _yolo_make_overview(img, low, cfg.yolo_overview_max_side)
    low_wh = overview_img.size
    low_url = pil_to_data_url(overview_img)

    log(f"  [yolo→smol→qwen] 전망 {low_wh[0]}×{low_wh[1]} · YOLO(ctx {low_w}×{low_h} 기준)…")
    t_y0 = time.perf_counter()
    crops, bboxes, summary, n_det, all_cls = run_yolo_crops(
        low,
        model=yolo,
        max_crops=cfg.max_crops,
        min_crop_short_side=cfg.min_crop_short_side,
        min_crop_area=cfg.min_crop_area,
        vlm_budget=cfg.yolo_vlm_budget,
        **_yolo_run_common_kwargs(cfg),
        **_yolo_budget_kwargs(img, cfg),
    )
    t_yolo = time.perf_counter() - t_y0
    log(
        f"  [yolo→smol→qwen] YOLO {t_yolo:.2f}s 탐지 {n_det} 크롭 {len(crops)} "
        f"(vlm_budget={cfg.yolo_vlm_budget})"
    )

    log("  [yolo→smol→qwen] Smol…")
    t_s0 = time.perf_counter()
    smol_ok = False
    s_body = ""
    s_usage = None
    try:
        s_body, s_usage = chat_vlm_multi(
            client=client_small,
            model=model_small,
            prompt=cfg.stage1_prompt,
            image_data_urls=[low_url],
            max_tokens=cfg.smol_max_tokens,
            content_image_first=True,
        )
        smol_ok = True
    except Exception as e:
        s_body = repr(e)
    t_smol = time.perf_counter() - t_s0
    stage1_usage_d = usage_to_dict(s_usage) if smol_ok and s_usage else None
    st1_pt = (stage1_usage_d or {}).get("prompt_tokens") if stage1_usage_d else None
    log(
        f"  [yolo→smol→qwen] Smol {t_smol:.2f}s ok={smol_ok} prompt_tokens={st1_pt}",
    )

    bboxes_ov = _scale_bboxes(bboxes, low_w, low_h, low_wh[0], low_wh[1])
    yolo_sentence_en = hr_bench_yolo_context_sentence_en(
        overview_w=low_wh[0],
        overview_h=low_wh[1],
        canvas_w=low_w,
        canvas_h=low_h,
        n_raw_detections=n_det,
        n_crops_sent=len(crops),
        bboxes_ov=bboxes_ov,
        crop_max_side=cfg.crop_max_side,
        context_resize_scale=cfg.context_resize_scale,
        context_max_side=cfg.context_max_side if cfg.context_resize_scale is None else None,
    )
    smol_summary_en = summarize_smol_for_hr_bench_mc_en(s_body, ok=smol_ok)
    qwen_image_urls: list[str] = [low_url]
    for c in crops:
        rc = prep_yolo_crop_for_vlm(c, cfg.crop_max_side)
        qwen_image_urls.append(pil_to_data_url(rc))

    q_prompt = compose_hr_bench_mc_with_supplementary_en(
        mcq_prompt,
        yolo_context_sentence_en=yolo_sentence_en,
        compact_vlm_sentence_en=smol_summary_en,
    )
    log(f"  [yolo→smol→qwen] Qwen {len(qwen_image_urls)}장…")
    if after_smol_start_qwen is not None:
        client_qwen = after_smol_start_qwen()
    t_q0 = time.perf_counter()
    raw, usage = chat_vlm_multi(
        client=client_qwen,
        model=model_qwen,
        prompt=q_prompt,
        image_data_urls=qwen_image_urls,
        max_tokens=cfg.max_tokens,
    )
    t_q = time.perf_counter() - t_q0
    u = usage_to_dict(usage)
    res_q = resource_snapshot(label="hr_sequential_qwen_after")
    pt = getattr(usage, "prompt_tokens", None) if usage else None
    total = t_yolo + t_smol + t_q
    log(f"  [yolo→smol→qwen] Qwen {t_q:.2f}s · 총 {total:.2f}s prompt_tokens={pt}")
    crop_thumbs = [
        _jpeg_data_url_thumb(prep_yolo_crop_for_vlm(c, cfg.crop_max_side))
        for c in crops
    ]
    dbg = {
        "thumbnail_original_jpeg_url": _jpeg_data_url_thumb(img),
        "yolo_canvas_size_px": {"w": low_w, "h": low_h},
        "overview_thumbnail_url": _jpeg_data_url_thumb(overview_img),
        "overview_sent_size": list(low_wh),
        "crop_thumbnails_jpeg_urls": crop_thumbs,
        "yolo_bbox_xyxy_on_canvas": [list(b) for b in bboxes],
        "yolo_bbox_xyxy_overview": [list(b) for b in bboxes_ov],
        "smol_prompt": cfg.stage1_prompt,
        "smol_reply_text": s_body or "",
        "smol_reply_preview": (
            (s_body or "")[:4000] + ("…" if smol_ok and len(s_body or "") > 4000 else "")
        ),
        "smol_ok": smol_ok,
        "yolo_context_sentence_en": yolo_sentence_en,
        "compact_vision_sentence_en": smol_summary_en,
        "qwen_prompt": q_prompt,
    }
    return {
        "raw_reply": raw,
        "seconds": round(total, 3),
        "prompt_tokens": pt,
        "usage": u,
        "resources_after": res_q,
        "stage1_prompt_tokens": st1_pt,
        "stage1_usage": stage1_usage_d,
        "wall_parallel_seconds": None,
        "yolo_seconds": round(t_yolo, 3),
        "smol_seconds": round(t_smol, 3),
        "n_detections": n_det,
        "n_crops_sent": len(crops),
        "smol_ok": smol_ok,
        "hr_debug": dbg,
    }


def run_strategy_on_indices(
    *,
    strategy: str,
    ds: Any,
    indices: list[int],
    client_qwen: OpenAI,
    model_qwen: str,
    client_small: OpenAI | None,
    model_small: str | None,
    cfg: HRBenchStrategyConfig,
    log: LogFn = noop_log,
    smol_single_server_hooks: SmolSingleServerHooks | None = None,
) -> list[dict[str, Any]]:
    from qwen_vlm.hr_bench.io import (
        format_mcq_prompt,
        norm_gt,
        parse_pred,
        snapshot_hr_dataset_row,
        to_pil,
    )

    if strategy not in STRATEGIES_ALL:
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy in _SMOL_STRATEGY_NAMES:
        if client_small is None or not model_small:
            raise ValueError(f"{strategy} 는 Smol client·model 이 필요합니다")

    yolo = None
    if strategy != "qwen_only":
        log(f"[{strategy}] YOLO 모델 로드: {cfg.yolo_model_path}")
        yolo = load_yolo(cfg.yolo_model_path)

    rows: list[dict[str, Any]] = []
    n_run = len(indices)
    for pos, i in enumerate(indices):
        snapshot = snapshot_hr_dataset_row(ds, i)
        gt = norm_gt(snapshot.get("answer"))
        idx = int(snapshot.get("index", i))
        mcq = format_mcq_prompt(snapshot)
        done_before_pct = 100.0 * pos / n_run if n_run else 0.0
        log(
            f"[{strategy}] ({pos + 1}/{n_run}, 이전까지 {done_before_pct:.0f}% 완료) "
            f"ds행={i} · HR idx={idx} · 풀이 중…"
        )
        pil = snapshot.get("image")
        if pil is None:
            log("  건너뜀: image 컬럼 없음")
            continue
        if not isinstance(pil, Image.Image):
            try:
                pil = to_pil(pil)
            except Exception as e:
                log(f"  건너뜀: 이미지 로드 실패 {e}")
                continue

        try:
            if smol_single_server_hooks is not None and strategy in _SMOL_STRATEGY_NAMES:
                smol_single_server_hooks.before_smol_sample()
            if strategy == "qwen_only":
                meta = run_sample_qwen_only(
                    img=pil,
                    mcq_prompt=mcq,
                    client=client_qwen,
                    model=model_qwen,
                    cfg=cfg,
                    log=log,
                )
            elif strategy == "yolo_lowres_crops":
                meta = run_sample_yolo_lowres_crops(
                    img=pil,
                    mcq_prompt=mcq,
                    client=client_qwen,
                    model=model_qwen,
                    cfg=cfg,
                    yolo=yolo,
                    log=log,
                )
            elif strategy == "yolo_smol_parallel":
                cb = (
                    smol_single_server_hooks.after_smol_get_qwen_client
                    if smol_single_server_hooks
                    else None
                )
                meta = run_sample_yolo_smol_parallel(
                    img=pil,
                    mcq_prompt=mcq,
                    client_qwen=client_qwen,
                    model_qwen=model_qwen,
                    client_small=client_small,  # type: ignore[arg-type]
                    model_small=model_small or "",
                    cfg=cfg,
                    yolo=yolo,
                    log=log,
                    after_smol_start_qwen=cb,
                    single_gpu_serial_yolo_smol=smol_single_server_hooks is not None,
                )
            else:
                cb = (
                    smol_single_server_hooks.after_smol_get_qwen_client
                    if smol_single_server_hooks
                    else None
                )
                meta = run_sample_yolo_smol_sequential(
                    img=pil,
                    mcq_prompt=mcq,
                    client_qwen=client_qwen,
                    model_qwen=model_qwen,
                    client_small=client_small,  # type: ignore[arg-type]
                    model_small=model_small or "",
                    cfg=cfg,
                    yolo=yolo,
                    log=log,
                    after_smol_start_qwen=cb,
                )
        except Exception as e:
            fail_pct = 100.0 * (pos + 1) / n_run if n_run else 100.0
            log(
                f"[{strategy}]   → 샘플 {pos + 1}/{n_run} 오류 ({fail_pct:.1f}%) · "
                f"ds행={i} idx={idx}: {e}"
            )
            continue
        finally:
            if smol_single_server_hooks is not None and strategy in _SMOL_STRATEGY_NAMES:
                smol_single_server_hooks.after_qwen_sample()

        raw = meta.get("raw_reply") or ""
        pred = parse_pred(str(raw))
        ok = bool(gt and pred and gt == pred)
        done_pct = 100.0 * (pos + 1) / n_run if n_run else 100.0
        mark = "정답" if ok else "오답"
        log(
            f"[{strategy}]   → 샘플 {pos + 1}/{n_run} 완료 ({done_pct:.1f}%) · "
            f"ds행={i} idx={idx} · {mark} (gt={gt!s} pred={pred!s})"
        )
        meta_out = dict(meta)
        dbg = meta_out.pop("hr_debug", None)
        meta_out["hr_debug"] = copy.deepcopy(dbg) if isinstance(dbg, dict) else dbg

        rows.append(
            {
                "dataset_row_index": i,
                "index": idx,
                "ground_truth_letter": gt,
                "pred_letter": pred,
                "correct": ok,
                "raw_reply": str(raw)[:2000],
                "strategy": strategy,
                **meta_out,
            }
        )
    return rows
