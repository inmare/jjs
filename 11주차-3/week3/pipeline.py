"""THUMB1280 + CROP8 + 가변 Detection CNN 파이프라인."""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from qwen_vlm.utils.image_resize import resize_max_side, resize_uniform_scale

from week3.config import (
    CROP_AREA_DENOMINATOR,
    CROP_AREA_NUMERATOR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_QWEN_MODEL_ID,
    THUMB_MAX_SIDE,
    YOLO_CONTEXT_SCALE,
)
from week3.datasets import compute_correctness, extract_gt_answer, extract_image, extract_question
from week3.detection_crops import run_detector_crops
from week3.detectors.base import DetectorSpec, load_detector_backend
from week3.metrics import SampleMetrics


def prepare_cnn_inputs(
    image: Image.Image,
    *,
    spec: DetectorSpec,
    backend: Any,
    device: str,
    max_crops: int = 0,
) -> dict[str, Any]:
    """썸네일 + 탐지 crop 준비 (VLM 없이 전처리만)."""
    full = image.convert("RGB")
    thumb = resize_max_side(full.copy(), THUMB_MAX_SIDE)
    ctx = resize_uniform_scale(full, YOLO_CONTEXT_SCALE)

    t0 = time.perf_counter()
    crops, bboxes, summary, n_det, _ = run_detector_crops(
        ctx,
        spec=spec,
        backend=backend,
        device=device,
        original_full_image_size=full.size,
        vlm_overview_max_side_for_budget=THUMB_MAX_SIDE,
        max_crops=max_crops,
    )
    det_sec = time.perf_counter() - t0

    vlm_images: list[Image.Image] = [thumb]
    vlm_images.extend(crops)

    return {
        "thumb": thumb,
        "crops": crops,
        "bboxes": bboxes,
        "summary": summary,
        "n_det": n_det,
        "n_crops": len(crops),
        "yolo_time_sec": det_sec,
        "vlm_images": vlm_images,
        "preprocess_time_sec": det_sec,
    }


class QwenVLMRunner:
    """Qwen3-VL-4B (transformers, 4bit 가능)."""

    def __init__(self, model_id: str = DEFAULT_QWEN_MODEL_ID) -> None:
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"Loading VLM {self.model_id}…", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        load_kw: dict[str, Any] = {"device_map": "auto"}
        try:
            from transformers import BitsAndBytesConfig

            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except ImportError:
            load_kw["torch_dtype"] = (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, **load_kw
        )
        print("VLM loaded.", flush=True)

    def unload(self) -> None:
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(
        self,
        images: list[Image.Image],
        question: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> tuple[str, SampleMetrics]:
        assert self.processor is not None and self.model is not None
        instruction = (
            "Choose the best answer from the given options. "
            "Respond with only one letter from the given options."
        )
        user_text = f"{question}\n\n{instruction}"
        content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": user_text})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant. You receive a thumbnail of the scene "
                    "and cropped regions from a detector. Answer the multiple-choice question."
                ),
            },
            {"role": "user", "content": content},
        ]

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        t_proc = time.perf_counter()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        proc_sec = time.perf_counter() - t_proc

        t_gen0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen_sec = time.perf_counter() - t_gen0

        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if self.device.startswith("cuda")
            else 0.0
        )
        in_ids = inputs.input_ids.shape[1]
        out_len = trimmed[0].shape[0]
        total_tok = float(in_ids + out_len)

        m = SampleMetrics(
            processor_time_sec=proc_sec,
            prefill_time_sec=proc_sec,
            decode_time_sec=gen_sec,
            output_time_sec=gen_sec,
            input_time_with_preprocess_sec=proc_sec,
            total_time_sec=proc_sec + gen_sec,
            text_tokens=float(out_len),
            image_tokens=max(0.0, float(in_ids - 64)),
            total_tokens=total_tok,
            prefill_tokens_per_sec=in_ids / proc_sec if proc_sec > 0 else 0.0,
            decode_tokens_per_sec=out_len / gen_sec if gen_sec > 0 else 0.0,
            prefill_mem_peak_allocated_mb=peak_mb,
            decode_mem_peak_allocated_mb=peak_mb,
            overall_peak_allocated_mb=peak_mb,
        )
        return response.strip(), m


def run_sample(
    example: dict[str, Any],
    *,
    spec: DetectorSpec,
    backend: Any,
    device: str,
    vlm: QwenVLMRunner | None,
    detect_only: bool,
    max_crops: int = 0,
    sample_index: int = -1,
) -> SampleMetrics | None:
    image = extract_image(example)
    if image is None:
        return None

    question = extract_question(example)
    gt = extract_gt_answer(example)

    pack = prepare_cnn_inputs(
        image, spec=spec, backend=backend, device=device, max_crops=max_crops
    )

    out = SampleMetrics(
        sample_index=sample_index,
        gt_answer=gt,
        detector_summary=str(pack.get("summary") or ""),
        preprocess_time_sec=pack["preprocess_time_sec"],
        yolo_time_sec=pack["yolo_time_sec"],
        n_detections=int(pack["n_det"]),
        n_crops_sent=int(pack["n_crops"]),
        num_objects=float(pack["n_crops"]),
    )

    if detect_only:
        out.total_time_sec = pack["preprocess_time_sec"]
        return out

    if vlm is None:
        raise ValueError("vlm required when not detect_only")

    t_wall = time.perf_counter()
    response, vm = vlm.infer(pack["vlm_images"], question)
    wall = time.perf_counter() - t_wall

    pred_display, correct = compute_correctness(response, gt, question)
    out.correct = correct
    out.model_response = response.strip()
    out.predicted_answer = pred_display
    out.processor_time_sec = vm.processor_time_sec
    out.prefill_time_sec = vm.prefill_time_sec
    out.decode_time_sec = vm.decode_time_sec
    out.output_time_sec = vm.output_time_sec
    out.input_time_with_preprocess_sec = pack["preprocess_time_sec"] + vm.processor_time_sec
    out.total_time_sec = pack["preprocess_time_sec"] + wall
    out.text_tokens = vm.text_tokens
    out.image_tokens = vm.image_tokens
    out.total_tokens = vm.total_tokens
    out.prefill_tokens_per_sec = vm.prefill_tokens_per_sec
    out.decode_tokens_per_sec = vm.decode_tokens_per_sec
    out.prefill_mem_peak_allocated_mb = vm.prefill_mem_peak_allocated_mb
    out.decode_mem_peak_allocated_mb = vm.decode_mem_peak_allocated_mb
    out.overall_peak_allocated_mb = vm.overall_peak_allocated_mb
    return out
