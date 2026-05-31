import argparse
import json
import gc
import math
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

QWEN_SMOL_DIR = Path(__file__).resolve().parent
_REPO = QWEN_SMOL_DIR.parent
_W3 = _REPO / "11주차-3"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))
if str(QWEN_SMOL_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_SMOL_DIR))

RUN_DIR_PREFIX = "qwen_smol"
RUN_STAMP_FMT = "%Y%m%d_%H%M%S"
METHODS = ["Smol_YOLO_Qwen_4B", "Single_VLM_Qwen_4B", "Thumb_Smol_YOLO_Qwen_4B"]


def format_run_stamp(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime(RUN_STAMP_FMT)


def make_run_dir(base: Path | None = None, stamp: str | None = None) -> Path:
    root = base if base is not None else QWEN_SMOL_DIR
    run_dir = root / f"{RUN_DIR_PREFIX}_{stamp or format_run_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_run_dir(output: str = "") -> Path:
    """결과 폴더. 비우면 ``qwen_smol/qwen_smol_YYYYMMDD_HHMMSS/`` 생성."""
    if not output.strip():
        return make_run_dir()
    p = Path(output).expanduser()
    if p.suffix.lower() == ".csv":
        p = p.parent
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()

try:
    import pyarrow  # noqa: F401 — HR-Bench parquet (datasets) + Windows import order
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pyarrow 가 필요합니다 (HR-Bench parquet 로드). "
        "uv 사용 시: uv sync --group dev && uv run python qwen_smol/benchmark_runner.py ..."
    ) from exc
import faulthandler
faulthandler.enable()

# Windows DLL conflict workaround: import transformers before datasets
print("Loading transformers...", flush=True)
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration

print("Loading torch...", flush=True)
import torch
print("Loading pandas...", flush=True)
import pandas as pd
print("Loading PIL...", flush=True)
from PIL import Image

print("Loading local modules...", flush=True)
from qwen_vlm.utils.image_resize import resize_max_side, resize_uniform_scale
from qwen_vlm.vision.yolo import load_yolo, run_yolo_crops

from week3.config import DEFAULT_BENCHMARKS  # noqa: E402
from week3.datasets import (  # noqa: E402
    compute_correctness,
    extract_gt_answer,
    extract_image,
    extract_question,
    is_multiple_choice,
    load_benchmark_dataset,
)

# Qwen 입력 정책 (11주차-3 CROP8 · THUMB512)
# - YOLO crop: ctx(0.5×)에서 잘라낸 뒤 CROP8 필터만 적용, 추가 리사이즈 없음 (week3 와 동일)
# - crop > BBOX_PROMPT_MAX_CROPS 이면 bbox 텍스트 생략 + naive grid 로 1장 병합 (rectpack 대신)
# - Thumb_Smol 썸네일: 긴 변 512px 고정
# - Single_VLM: 원본 그대로 (Qwen processor 가 smart_resize)
THUMB_MAX_SIDE = 512
YOLO_CONTEXT_SCALE = 0.5
# 0 = CROP8·VLM 예산 필터 후 남는 crop 전부 (week3 DEFAULT_MAX_CROPS_VLM 과 동일)
DEFAULT_YOLO_MAX_CROPS = 0
# bbox 좌표 프롬프트: crop 이 이 개수를 **초과**하면 생략하고 grid merge 사용
BBOX_PROMPT_MAX_CROPS = 4
MERGED_BACKGROUND = (255, 255, 255)
CROP_AREA_NUMERATOR = 8
CROP_AREA_DENOMINATOR = 100
YOLO_DEVICE = "cpu"  # Qwen GPU VRAM 과 분리
DEFAULT_MAX_NEW_TOKENS = 64


def parse_str_list(s: str) -> list[str]:
    return [p.strip() for p in s.replace(",", " ").split() if p.strip()]


def sample_range(n_total: int, start: int, max_samples: int) -> tuple[int, int]:
    end = n_total if max_samples <= 0 else min(n_total, start + max_samples)
    return start, end


def result_file_stem(benchmark: str, method: str) -> str:
    return f"{benchmark}_{method}"


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def scale_bboxes_ctx_to_original(
    bboxes_ctx: list[tuple[int, int, int, int]],
    ctx_size: tuple[int, int],
    orig_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """YOLO ctx 좌표 → 원본 픽셀 좌표."""
    cw, ch = ctx_size
    ow, oh = orig_size
    if cw <= 0 or ch <= 0:
        return []
    sx = ow / cw
    sy = oh / ch
    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in bboxes_ctx:
        out.append(
            (
                int(round(x1 * sx)),
                int(round(y1 * sy)),
                int(round(x2 * sx)),
                int(round(y2 * sy)),
            )
        )
    return out


def build_yolo_crop_bbox_preamble_en(
    *,
    orig_w: int,
    orig_h: int,
    bboxes_orig: list[tuple[int, int, int, int]],
    first_crop_image_index: int,
    has_thumbnail: bool = False,
    thumb_max_side: int = THUMB_MAX_SIDE,
) -> str:
    """클래스명·점수 없이 crop 슬롯 ↔ 원본 bbox 만 안내 (11주차-2 라벨 제거 유지)."""
    if not bboxes_orig:
        return ""

    lines = [f"Original image size: {orig_w}×{orig_h} px (width×height)."]
    if has_thumbnail:
        lines.append(
            f"Image #0: low-resolution thumbnail of the full scene "
            f"(longest edge capped at {thumb_max_side} px)."
        )
    lines.append(
        "The following image slots are axis-aligned crops from the original image. "
        "Use the coordinates below to locate each patch (no object class names or detector scores):"
    )
    for i, (x1, y1, x2, y2) in enumerate(bboxes_orig):
        slot = first_crop_image_index + i
        lines.append(
            f"  · Image #{slot}: [x1,y1,x2,y2]=[{x1},{y1},{x2},{y2}] in original pixels"
        )
    return "\n".join(lines)


def merge_crops_naive_grid(crop_images: list[Image.Image]) -> Image.Image:
    """rectpack 대신 sqrt-grid 로 crop 타일 배치 (9주차·11주차-2 CNN_MERGE 개선안)."""
    if not crop_images:
        return Image.new("RGB", (1, 1), MERGED_BACKGROUND)

    crops = [img.convert("RGB") for img in crop_images]
    n = len(crops)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    col_widths = [0] * cols
    row_heights = [0] * rows
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        col_widths[c] = max(col_widths[c], crop.width)
        row_heights[r] = max(row_heights[r], crop.height)

    merged = Image.new("RGB", (sum(col_widths), sum(row_heights)), MERGED_BACKGROUND)
    y = 0
    for r in range(rows):
        x = 0
        for c in range(cols):
            idx = r * cols + c
            if idx >= n:
                break
            merged.paste(crops[idx], (x, y))
            x += col_widths[c]
        y += row_heights[r]
    return merged


def build_grid_merge_preamble_en(
    *,
    n_crops: int,
    merged_image_index: int,
    grid_cols: int,
    grid_rows: int,
    has_thumbnail: bool = False,
    thumb_max_side: int = THUMB_MAX_SIDE,
) -> str:
    lines: list[str] = []
    if has_thumbnail:
        lines.append(
            f"Image #0: low-resolution thumbnail of the full scene "
            f"(longest edge capped at {thumb_max_side} px)."
        )
    lines.append(
        f"Image #{merged_image_index}: a mosaic grid ({grid_rows}×{grid_cols}) of {n_crops} "
        "region crops from the original image, placed in row-major order "
        "(left-to-right, top-to-bottom). Per-region bounding-box coordinates are omitted "
        "because there are many crops."
    )
    return "\n".join(lines)


def pack_vlm_crops_for_qwen(
    vlm_crops: list[Image.Image],
) -> tuple[list[Image.Image], bool, float]:
    """crop 수가 많으면 naive grid 1장으로 병합. (images, used_grid_merge, merge_time_sec)."""
    if len(vlm_crops) <= BBOX_PROMPT_MAX_CROPS:
        return list(vlm_crops), False, 0.0
    t0 = time.perf_counter()
    merged = merge_crops_naive_grid(vlm_crops)
    return [merged], True, time.perf_counter() - t0


# --- YOLO crop prep ---
def prepare_yolo_vlm_crops(
    image: Image.Image,
    *,
    yolo_model,
    max_crops: int,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]], int, float, str]:
    """저해상도 ctx 에서 YOLO 탐지 → CROP8 필터 후 crop (11주차-3, 추가 리사이즈 없음)."""
    full = image.convert("RGB")
    ctx = resize_uniform_scale(full, YOLO_CONTEXT_SCALE)
    t0 = time.perf_counter()
    crops, bboxes_ctx, summary, n_det, _all_cls = run_yolo_crops(
        ctx,
        model=yolo_model,
        yolo_device=YOLO_DEVICE,
        vlm_budget=True,
        max_crops=max_crops,
        original_full_image_size=full.size,
        vlm_overview_max_side_for_budget=THUMB_MAX_SIDE,
        crop_max_side_for_budget=0,
        max_bbox_area_numerator=CROP_AREA_NUMERATOR,
        max_bbox_area_denominator=CROP_AREA_DENOMINATOR,
        context_resize_scale=YOLO_CONTEXT_SCALE,
    )
    t_yolo = time.perf_counter() - t0
    bboxes_orig = scale_bboxes_ctx_to_original(bboxes_ctx, ctx.size, full.size)
    return list(crops), bboxes_orig, n_det, t_yolo, summary


def build_method_payload(
    method: str,
    *,
    image: Image.Image,
    vlm_crops: list[Image.Image],
    crop_bboxes_orig: list[tuple[int, int, int, int]],
    question: str,
    smol_desc: str,
    smol_time: float,
    t_yolo: float,
) -> tuple[str, str, list[Image.Image], float, float, float, bool, int]:
    instruction = (
        "Choose the best answer from the given options. Respond with only one letter from the given options."
        if is_multiple_choice(question)
        else "Answer the question concisely."
    )
    full = image.convert("RGB")
    ow, oh = full.size
    raw_crop_count = len(vlm_crops)
    crop_images, used_grid_merge, t_merge = pack_vlm_crops_for_qwen(vlm_crops)
    grid_cols = (
        max(1, math.ceil(math.sqrt(raw_crop_count))) if used_grid_merge else 0
    )
    grid_rows = math.ceil(raw_crop_count / grid_cols) if used_grid_merge else 0

    def _compose_user_text(
        *,
        include_smol: bool,
        bbox_first_index: int,
        has_thumbnail: bool,
        merged_image_index: int,
    ) -> str:
        parts: list[str] = []
        if include_smol and smol_desc:
            parts.append(f"[Image Description from SmolVLM]:\n{smol_desc}")
        if used_grid_merge and raw_crop_count > 0:
            grid_block = build_grid_merge_preamble_en(
                n_crops=raw_crop_count,
                merged_image_index=merged_image_index,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                has_thumbnail=has_thumbnail,
            )
            parts.append(f"[Crop regions]\n{grid_block}")
        else:
            bboxes_for_prompt = (
                crop_bboxes_orig
                if len(crop_bboxes_orig) <= BBOX_PROMPT_MAX_CROPS
                else []
            )
            bbox_block = build_yolo_crop_bbox_preamble_en(
                orig_w=ow,
                orig_h=oh,
                bboxes_orig=bboxes_for_prompt,
                first_crop_image_index=bbox_first_index,
                has_thumbnail=has_thumbnail,
            )
            if bbox_block:
                parts.append(f"[Crop regions]\n{bbox_block}")
        parts.append(f"[Question]:\n{question}")
        parts.append(instruction)
        return "\n\n".join(parts)

    if method == "Smol_YOLO_Qwen_4B":
        system_text = (
            "You are an AI assistant. You receive a detailed image description from a lightweight VLM "
            "and cropped patches for fine detail. Crop slots are identified by original-image pixel "
            "coordinates (no detector class names). Answer from both."
        )
        if used_grid_merge and raw_crop_count > 0:
            system_text = (
                "You are an AI assistant. You receive a detailed image description from a lightweight VLM "
                "and a mosaic grid of many region crops from the original image. Answer from both."
            )
        if vlm_crops:
            vlm_images = list(crop_images)
            user_text = _compose_user_text(
                include_smol=True,
                bbox_first_index=0,
                has_thumbnail=False,
                merged_image_index=0,
            )
        else:
            vlm_images = [resize_max_side(full.copy(), THUMB_MAX_SIDE)]
            user_text = _compose_user_text(
                include_smol=True,
                bbox_first_index=0,
                has_thumbnail=False,
                merged_image_index=0,
            )
        preprocess = smol_time + t_yolo + t_merge
        yolo_time, smol_time_out = t_yolo, smol_time
    elif method == "Single_VLM_Qwen_4B":
        system_text = "You are an AI assistant. You will be provided with an image to help you answer the question."
        user_text = f"[Question]:\n{question}\n\n{instruction}"
        vlm_images = [full.copy()]
        preprocess = 0.0
        yolo_time, smol_time_out = 0.0, 0.0
        used_grid_merge = False
        raw_crop_count = 0
    elif method == "Thumb_Smol_YOLO_Qwen_4B":
        system_text = (
            "You are an AI assistant. You receive a low-res thumbnail, a SmolVLM description, "
            "and cropped patches. Crop slots are identified by original-image pixel coordinates "
            "(no detector class names). Use all sources to answer."
        )
        if used_grid_merge and raw_crop_count > 0:
            system_text = (
                "You are an AI assistant. You receive a low-res thumbnail, a SmolVLM description, "
                "and a mosaic grid of many region crops. Use all sources to answer."
            )
        vlm_images = [resize_max_side(full.copy(), THUMB_MAX_SIDE), *crop_images]
        user_text = _compose_user_text(
            include_smol=True,
            bbox_first_index=1,
            has_thumbnail=True,
            merged_image_index=1,
        )
        preprocess = smol_time + t_yolo + t_merge
        yolo_time, smol_time_out = t_yolo, smol_time
    else:
        raise ValueError(f"unknown method: {method}")

    return (
        system_text,
        user_text,
        vlm_images,
        preprocess,
        yolo_time,
        smol_time_out,
        used_grid_merge,
        raw_crop_count,
    )


# --- Phase 1: SmolVLM ---
def run_smol_phase(
    slices: list[tuple[str, Any, int, int]],
) -> tuple[dict[str, dict[int, str]], dict[str, dict[int, dict[str, float]]]]:
    """벤치마크별 SmolVLM 설명 생성. ``slices``: (benchmark, dataset, start, end)."""
    print("=== Phase 1: Loading SmolVLM ===")
    device = get_device()
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Loading model for {model_id}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto",
    )
    print("Model loaded.")

    descriptions: dict[str, dict[int, str]] = {}
    smol_metrics: dict[str, dict[int, dict[str, float]]] = {}

    for benchmark, dataset, start, end in slices:
        descriptions.setdefault(benchmark, {})
        smol_metrics.setdefault(benchmark, {})
        print(f"\n[SmolVLM] {benchmark} samples {start}:{end}", flush=True)
        for idx in range(start, end):
            example = dataset[idx]
            image = extract_image(example)
            if image is None:
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail. Pay attention to small objects, "
                                "text, relationships, and colors."
                            ),
                        },
                    ],
                }
            ]

            t0 = time.perf_counter()
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=256)

            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            t1 = time.perf_counter()

            smol_peak_mem = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if device.startswith("cuda")
                else 0.0
            )
            smol_metrics[benchmark][idx] = {
                "smol_time_sec": t1 - t0,
                "smol_peak_mem_mb": smol_peak_mem,
            }

            output_text = generated_texts[0]
            if "Assistant:" in output_text:
                desc = output_text.split("Assistant:")[-1].strip()
            else:
                desc = output_text.strip()

            descriptions[benchmark][idx] = desc
            print(f"[SmolVLM] {benchmark}[{idx}] {t1 - t0:.2f}s", flush=True)

    print("=== Phase 1 Complete. Unloading SmolVLM ===")
    del model
    del processor
    free_memory()

    return descriptions, smol_metrics

# --- Phase 2: Qwen + YOLO ---
def run_qwen_phase(
    slices: list[tuple[str, Any, int, int]],
    descriptions: dict[str, dict[int, str]],
    smol_metrics: dict[str, dict[int, dict[str, float]]],
    *,
    max_crops: int = DEFAULT_YOLO_MAX_CROPS,
) -> dict[str, list[dict]]:
    print("=== Phase 2: Loading YOLO and Qwen ===")
    device = get_device()

    yolo_model = load_yolo()

    model_id = "Qwen/Qwen3-VL-4B-Instruct"

    print(f"Loading processor for {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model for {model_id}...", flush=True)
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
    )
    print("Model loaded.", flush=True)

    from summary import make_sample_record

    all_results: dict[str, list[dict]] = {}

    for benchmark, dataset, start, end in slices:
        print(f"\n=== Qwen phase: {benchmark} samples {start}:{end} ===", flush=True)
        for idx in range(start, end):
            print(f"\n--- {benchmark} sample {idx} ---")
            example = dataset[idx]
            image = extract_image(example)
            if image is None:
                continue

            question = extract_question(example)
            gt_answer = extract_gt_answer(example)
            smol_desc = descriptions.get(benchmark, {}).get(idx, "")

            vlm_crops, crop_bboxes_orig, n_det, t_yolo, yolo_summary = prepare_yolo_vlm_crops(
                image,
                yolo_model=yolo_model,
                max_crops=max_crops,
            )
            smol_info = smol_metrics.get(benchmark, {}).get(idx, {})
            smol_time = smol_info.get("smol_time_sec", 0.0)
            smol_peak_mem = smol_info.get("smol_peak_mem_mb", 0.0)

            print(
                f"[YOLO] detections={n_det}, vlm_crops={len(vlm_crops)}, "
                f"summary={yolo_summary[:120]}...",
                flush=True,
            )

            for method in METHODS:
                print(f"\n[Running Method: {method}]", flush=True)

                system_text, user_text, vlm_images, cur_preprocess_time, cur_yolo_time, cur_smol_time, used_grid_merge, raw_crop_count = (
                build_method_payload(
                    method,
                    image=image,
                    vlm_crops=vlm_crops,
                    crop_bboxes_orig=crop_bboxes_orig,
                    question=question,
                    smol_desc=smol_desc,
                    smol_time=smol_time,
                    t_yolo=t_yolo,
                )
                )
                content: list[dict] = [{"type": "image", "image": im} for im in vlm_images]
                content.append({"type": "text", "text": user_text})
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": content},
                ]

                if method == "Smol_YOLO_Qwen_4B":
                    print("\n[DEBUG] SmolVLM Description:")
                    print(smol_desc[:500] + ("..." if len(smol_desc) > 500 else ""))
                    if used_grid_merge:
                        print(
                            f"[DEBUG] VLM: naive grid merge of {raw_crop_count} crops → "
                            f"{len(vlm_images)} image(s)"
                        )
                    else:
                        print(
                            f"[DEBUG] VLM images: {len(vlm_images)} crops "
                            f"(CROP8, ctx scale {YOLO_CONTEXT_SCALE})"
                        )

                t_proc_start = time.perf_counter()
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                t_processor = time.perf_counter() - t_proc_start

                text_token_ids = processor.tokenizer(
                    processor.apply_chat_template(
                        [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                text_tokens = len(text_token_ids)
                total_input_tokens = inputs.input_ids.shape[-1]
                image_tokens = max(0, total_input_tokens - text_tokens)

                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                t_gen_start = time.perf_counter()
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_generate = time.perf_counter() - t_gen_start

                peak_mb = (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if device.startswith("cuda")
                    else 0.0
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_tokens = (
                    generated_ids_trimmed[0].shape[-1] if generated_ids_trimmed else 0
                )

                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                pred, is_correct = compute_correctness(output_text, gt_answer, question)

                input_time_with_preprocess_sec = cur_preprocess_time + t_processor
                total_time_sec = input_time_with_preprocess_sec + t_generate
                n_crops_sent = raw_crop_count if method != "Single_VLM_Qwen_4B" else 0

                print(
                    f"[Result] Pred: {pred} | GT: {gt_answer} | Correct: {is_correct} | "
                    f"images={len(vlm_images)} tok={total_input_tokens} peak={peak_mb:.0f}MB | "
                    f"Total: {total_time_sec:.2f}s",
                    flush=True,
                )

                record = make_sample_record(
                    benchmark=benchmark,
                    method=method,
                    sample_index=idx,
                    correct=is_correct,
                    predicted_answer=pred,
                    gt_answer=gt_answer,
                    model_response=output_text,
                    detector_summary=yolo_summary,
                    preprocess_time_sec=cur_preprocess_time,
                    yolo_time_sec=cur_yolo_time,
                    processor_time_sec=t_processor,
                    prefill_time_sec=t_processor,
                    decode_time_sec=t_generate,
                    input_time_with_preprocess_sec=input_time_with_preprocess_sec,
                    total_time_sec=total_time_sec,
                    text_tokens=float(text_tokens),
                    image_tokens=float(image_tokens),
                    total_tokens=float(total_input_tokens),
                    prefill_tokens_per_sec=total_input_tokens / max(t_processor, 0.001),
                    decode_tokens_per_sec=output_tokens / max(t_generate, 0.001),
                    prefill_mem_peak_allocated_mb=peak_mb,
                    decode_mem_peak_allocated_mb=peak_mb,
                    overall_peak_allocated_mb=peak_mb,
                    n_crops_sent=n_crops_sent,
                    n_detections=n_det,
                    smol_time_sec=cur_smol_time if method != "Single_VLM_Qwen_4B" else 0.0,
                    smol_peak_mem_mb=smol_peak_mem if method != "Single_VLM_Qwen_4B" else 0.0,
                    smol_description=smol_desc if method != "Single_VLM_Qwen_4B" else "",
                    question=question,
                    num_vlm_images=len(vlm_images),
                )

                result_key = result_file_stem(benchmark, method)
                all_results.setdefault(result_key, []).append(record)

                del inputs, generated_ids, generated_ids_trimmed
                free_memory()

    print("=== Phase 2 Complete ===")
    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="Smol+YOLO+Qwen 벤치 (11주차-3 와 동일 벤치마크 세트 지원)",
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help=f"쉼표 구분 (기본: {','.join(DEFAULT_BENCHMARKS)})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="벤치마크당 샘플 수 (0=전체)",
    )
    parser.add_argument("--start", type=int, default=0, help="시작 샘플 인덱스")
    parser.add_argument(
        "--max-crops",
        type=int,
        default=DEFAULT_YOLO_MAX_CROPS,
        help="YOLO→Qwen crop 상한 (0=CROP8·VLM 예산 내 무제한, 기본 0)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="결과 폴더 (기본: qwen_smol/qwen_smol_YYYYMMDD_HHMMSS/)",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="(레거시) 루트에 benchmark_results_*.csv 저장. --output 과 동시 사용 불가",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="summary.csv 기반 PNG 그래프 생성 (기본: on)",
    )
    args = parser.parse_args()

    if args.output.strip() and args.output_prefix.strip():
        parser.error("--output 과 --output-prefix 는 동시에 쓸 수 없습니다.")

    benchmarks = parse_str_list(args.benchmarks)
    if not benchmarks:
        parser.error("--benchmarks 가 비어 있습니다.")

    legacy_prefix = args.output_prefix.strip() or None
    run_dir = resolve_run_dir(args.output) if not legacy_prefix else None

    slices: list[tuple[str, Any, int, int]] = []
    for bench in benchmarks:
        ds = load_benchmark_dataset(bench)
        start, end = sample_range(len(ds), args.start, args.max_samples)
        if start >= end:
            print(f"[경고] {bench}: 샘플 범위가 비어 있음 (start={start}, end={end})", flush=True)
            continue
        slices.append((bench, ds, start, end))
        print(f"{bench}: rows {start}:{end} / {len(ds)}", flush=True)

    if not slices:
        parser.error("실행할 벤치마크 샘플이 없습니다.")

    if run_dir is not None:
        print(f"결과 폴더: {run_dir}", flush=True)
        run_config = {
            "benchmarks": benchmarks,
            "max_samples": args.max_samples,
            "start_index": args.start,
            "methods": METHODS,
            "smol_model": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "qwen_model": "Qwen/Qwen3-VL-4B-Instruct",
            "thumb_max_side": THUMB_MAX_SIDE,
            "single_vlm_resize": "none",
            "yolo_crop_resize": "none",
            "crop_area_filter": f"{CROP_AREA_NUMERATOR}/{CROP_AREA_DENOMINATOR}",
            "yolo_context_scale": YOLO_CONTEXT_SCALE,
            "yolo_max_crops": args.max_crops,
            "bbox_prompt_max_crops": BBOX_PROMPT_MAX_CROPS,
            "grid_merge_policy": (
                f"crop>{BBOX_PROMPT_MAX_CROPS}: bbox 생략 + naive sqrt-grid 1장 병합"
            ),
            "yolo_device": YOLO_DEVICE,
            "note": (
                "YOLO crop: 11주차-3 CROP8. crop≤{n} 개별 crop+bbox 좌표. "
                "crop>{n} naive grid merge. Thumb 512px. Single_VLM 원본 전송."
            ).format(n=BBOX_PROMPT_MAX_CROPS),
            "run_stamp": format_run_stamp(),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    descriptions, smol_metrics = run_smol_phase(slices)

    all_results = run_qwen_phase(
        slices,
        descriptions,
        smol_metrics,
        max_crops=args.max_crops,
    )

    from summary import write_sample_csv, write_sample_jsonl, write_summary_csvs

    for result_key, results in all_results.items():
        if not results:
            continue

        if run_dir is not None:
            output_file = run_dir / f"{result_key}.csv"
            samples_dir = run_dir / "samples"
            sample_path = samples_dir / f"{result_key}.jsonl"
            write_sample_csv(results, output_file)
            write_sample_jsonl(results, sample_path)
        else:
            output_file = Path(f"{legacy_prefix}_{result_key}.csv")
            write_sample_csv(results, output_file)

        scored = [r for r in results if r.get("correct") is not None]
        acc = (
            sum(int(r.get("correct") or 0) for r in scored) / len(scored) * 100
            if scored
            else 0.0
        )
        print(f"[{result_key}] Results saved to {output_file}")
        print(f"[{result_key}] Accuracy: {acc:.2f}%")

    if run_dir is not None:
        summary_path, comparison_path = write_summary_csvs(run_dir, all_results)
        print(f"summary.csv (11주차-3 호환): {summary_path}")
        print(f"summary_comparison.csv: {comparison_path}")

        if args.plots:
            from week3.plots import generate_plots_from_summary_csv

            plot_paths = generate_plots_from_summary_csv(summary_path, plots_dir=run_dir)
            print(f"그래프 {len(plot_paths)}개 저장:", flush=True)
            for p in plot_paths:
                print(f"  {p}", flush=True)
    elif legacy_prefix:
        from summary import build_summary_rows
        from week3.metrics import SUMMARY_COLUMNS

        rows = build_summary_rows(all_results)
        df = pd.DataFrame(rows)
        for c in SUMMARY_COLUMNS:
            if c not in df.columns:
                df[c] = None
        comparison_path = Path(f"{legacy_prefix}_summary_comparison.csv")
        df[SUMMARY_COLUMNS].to_csv(comparison_path, index=False, encoding="utf-8-sig")
        print(f"Summary saved to {comparison_path}")

if __name__ == "__main__":
    main()
