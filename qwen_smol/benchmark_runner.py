import argparse
import json
import gc
import re
import time
from datetime import datetime
from pathlib import Path
import os
import sys

QWEN_SMOL_DIR = Path(__file__).resolve().parent
RUN_DIR_PREFIX = "qwen_smol"
RUN_STAMP_FMT = "%Y%m%d_%H%M%S"


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

import pyarrow
import faulthandler
faulthandler.enable()

# Windows DLL conflict workaround: import transformers before datasets
print("Loading transformers...", flush=True)
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, Qwen3VLForConditionalGeneration

print("Loading torch...", flush=True)
import torch
print("Loading pandas...", flush=True)
import pandas as pd
print("Loading PIL...", flush=True)
from PIL import Image
print("Loading datasets...", flush=True)
from datasets import load_dataset

print("Loading local modules...", flush=True)
from qwen_vlm.pipeline.experiment import prep_yolo_crop_for_vlm
from qwen_vlm.utils.image_resize import resize_max_side, resize_uniform_scale
from qwen_vlm.vision.yolo import load_yolo, run_yolo_crops

# Qwen 입력 정책 (experiment.py parallel-yolo-smol · 11주차-3 CROP8 정렬)
THUMB_MAX_SIDE = 512
SINGLE_VLM_MAX_SIDE = 1280
CROP_MAX_SIDE = 640
YOLO_CONTEXT_SCALE = 0.5
DEFAULT_YOLO_MAX_CROPS = 6
CROP_AREA_NUMERATOR = 8
CROP_AREA_DENOMINATOR = 100
YOLO_DEVICE = "cpu"  # Qwen GPU VRAM 과 분리
DEFAULT_MAX_NEW_TOKENS = 64


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Utility functions for HR Bench parsing ---
def extract_image(example) -> Image.Image:
    from qwen_vlm.hr_bench.io import to_pil
    if "image" in example:
        return to_pil(example["image"])
    elif "bytes" in example:
        import io
        return Image.open(io.BytesIO(example["bytes"])).convert("RGB")
    return None

def is_multiple_choice(question: str) -> bool:
    if not question:
        return False
    return bool(re.search(r"\bA[\.\)]", question)) or "Options:" in question

def extract_options(example) -> list[str]:
    options = []
    for letter in list("ABCDEFGHIJK"):
        for key in [letter, letter.lower(), f"option_{letter}", f"option_{letter.lower()}"]:
            if key in example and example[key] is not None:
                value = str(example[key]).strip()
                if value and value.lower() != "nan":
                    options.append(f"{letter}. {value}")
                break
    return options

def extract_question(example) -> str:
    question = example.get("question", "")
    if not question:
        # Fallback to prompt or other keys if needed
        pass
    options = extract_options(example)
    if options and "Options:" not in question and not re.search(r"\bA[\.\)]", question):
        question = question + "\n\nOptions:\n" + "\n".join(options)
    return question.strip()

def normalize_answer_letter(text) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    upper = raw.upper()
    match = re.search(r"\b([A-K])\b", upper)
    if match:
        return match.group(1)
    return upper[:1]

def parse_predicted_answer(text: str) -> str:
    if not text:
        return ""
    upper = str(text).upper()
    patterns = [
        r"answer\s*[:：]\s*([A-K])",
        r"정답\s*[:：]\s*([A-K])",
        r"\(([A-K])\)",
        r"\b([A-K])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return ""

def load_hrbench_4k():
    print("Loading HR-Bench 4K directly from parquet...")
    url = "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.parquet"
    ds = load_dataset("parquet", data_files={"data": url}, split="data")
    return ds


def prepare_yolo_vlm_crops(
    image: Image.Image,
    *,
    yolo_model,
    max_crops: int,
) -> tuple[list[Image.Image], int, float, str]:
    """저해상도 ctx 에서 YOLO 탐지 후 VLM 용 리사이즈 crop 만 반환."""
    full = image.convert("RGB")
    ctx = resize_uniform_scale(full, YOLO_CONTEXT_SCALE)
    t0 = time.perf_counter()
    crops, _bboxes, summary, n_det, _all_cls = run_yolo_crops(
        ctx,
        model=yolo_model,
        yolo_device=YOLO_DEVICE,
        vlm_budget=True,
        max_crops=max_crops,
        original_full_image_size=full.size,
        vlm_overview_max_side_for_budget=THUMB_MAX_SIDE,
        crop_max_side_for_budget=CROP_MAX_SIDE,
        max_bbox_area_numerator=CROP_AREA_NUMERATOR,
        max_bbox_area_denominator=CROP_AREA_DENOMINATOR,
        context_resize_scale=YOLO_CONTEXT_SCALE,
    )
    t_yolo = time.perf_counter() - t0
    vlm_crops = [prep_yolo_crop_for_vlm(c, CROP_MAX_SIDE) for c in crops]
    return vlm_crops, n_det, t_yolo, summary


def build_method_payload(
    method: str,
    *,
    image: Image.Image,
    vlm_crops: list[Image.Image],
    question: str,
    smol_desc: str,
    smol_time: float,
    t_yolo: float,
) -> tuple[str, str, list[Image.Image], float, float, float]:
    instruction = (
        "Choose the best answer from the given options. Respond with only one letter from the given options."
        if is_multiple_choice(question)
        else "Answer the question concisely."
    )
    full = image.convert("RGB")

    if method == "Smol_YOLO_Qwen_4B":
        # Smol 텍스트가 전망 역할 → 리사이즈 crop 만 전송 (원본 HR crop 금지)
        system_text = (
            "You are an AI assistant. You receive a detailed image description from a lightweight VLM "
            "and cropped patches (longest edge capped) for fine detail. Answer from both."
        )
        user_text = f"[Image Description from SmolVLM]:\n{smol_desc}\n\n[Question]:\n{question}\n\n{instruction}"
        vlm_images = list(vlm_crops) if vlm_crops else [resize_max_side(full.copy(), THUMB_MAX_SIDE)]
        preprocess = smol_time + t_yolo
        yolo_time, smol_time_out = t_yolo, smol_time
    elif method == "Single_VLM_Qwen_4B":
        system_text = "You are an AI assistant. You will be provided with an image to help you answer the question."
        user_text = f"[Question]:\n{question}\n\n{instruction}"
        vlm_images = [resize_max_side(full.copy(), SINGLE_VLM_MAX_SIDE)]
        preprocess = 0.0
        yolo_time, smol_time_out = 0.0, 0.0
    elif method == "Thumb_Smol_YOLO_Qwen_4B":
        system_text = (
            "You are an AI assistant. You receive a low-res thumbnail, a SmolVLM description, "
            "and resized crop patches. Use all sources to answer."
        )
        user_text = f"[Image Description from SmolVLM]:\n{smol_desc}\n\n[Question]:\n{question}\n\n{instruction}"
        vlm_images = [resize_max_side(full.copy(), THUMB_MAX_SIDE), *vlm_crops]
        preprocess = smol_time + t_yolo
        yolo_time, smol_time_out = t_yolo, smol_time
    else:
        raise ValueError(f"unknown method: {method}")

    return system_text, user_text, vlm_images, preprocess, yolo_time, smol_time_out


# --- Phase 1: SmolVLM ---
def run_smol_phase(dataset, max_samples=None):
    print("=== Phase 1: Loading SmolVLM ===")
    device = get_device()
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    
    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Loading model for {model_id}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto"
    )
    print("Model loaded.")

    descriptions = {}
    smol_metrics = {}
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    
    print(f"Generating descriptions for {total} samples...")
    for idx in range(total):
        example = dataset[idx]
        image = extract_image(example)
        if image is None:
            continue
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail. Pay attention to small objects, text, relationships, and colors."}
                ]
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
        
        smol_peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if device.startswith("cuda") else 0
        smol_metrics[idx] = {
            "smol_time_sec": t1 - t0,
            "smol_peak_mem_mb": smol_peak_mem
        }
        
        # The output often includes the prompt, so we take the assistant's part
        output_text = generated_texts[0]
        if "Assistant:" in output_text:
            desc = output_text.split("Assistant:")[-1].strip()
        else:
            desc = output_text.strip()
            
        descriptions[idx] = desc
        print(f"[SmolVLM] Sample {idx} processed in {t1-t0:.2f}s.")
        
    print("=== Phase 1 Complete. Unloading SmolVLM ===")
    del model
    del processor
    free_memory()
    
    return descriptions, smol_metrics

# --- Phase 2: Qwen + YOLO ---
def run_qwen_phase(dataset, descriptions, smol_metrics, max_samples=None, max_crops: int = DEFAULT_YOLO_MAX_CROPS):
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
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )
    print("Model loaded.", flush=True)
    
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    methods = ["Smol_YOLO_Qwen_4B", "Single_VLM_Qwen_4B", "Thumb_Smol_YOLO_Qwen_4B"]
    all_results = {m: [] for m in methods}

    if str(QWEN_SMOL_DIR) not in sys.path:
        sys.path.insert(0, str(QWEN_SMOL_DIR))
    from summary import make_sample_record
    
    for idx in range(total):
        print(f"\n--- Processing Sample {idx} ---")
        example = dataset[idx]
        image = extract_image(example)
        if image is None:
            continue
            
        question = extract_question(example)
        gt_answer = str(example.get("answer", example.get("gt_answer", "")))
        smol_desc = descriptions.get(idx, "")
        
        vlm_crops, n_det, t_yolo, yolo_summary = prepare_yolo_vlm_crops(
            image, yolo_model=yolo_model, max_crops=max_crops,
        )
        smol_info = smol_metrics.get(idx, {})
        smol_time = smol_info.get("smol_time_sec", 0.0)
        smol_peak_mem = smol_info.get("smol_peak_mem_mb", 0.0)

        print(f"[YOLO] detections={n_det}, vlm_crops={len(vlm_crops)}, summary={yolo_summary[:120]}...")
        
        for method in methods:
            print(f"\n[Running Method: {method}]")
            
            system_text, user_text, vlm_images, cur_preprocess_time, cur_yolo_time, cur_smol_time = (
                build_method_payload(
                    method,
                    image=image,
                    vlm_crops=vlm_crops,
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
                print(f"[DEBUG] VLM images: {len(vlm_images)} crops (max side {CROP_MAX_SIDE}px)")
            
            t_proc_start = time.perf_counter()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.startswith("cuda") else 0
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_tokens = generated_ids_trimmed[0].shape[-1] if generated_ids_trimmed else 0
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            pred = parse_predicted_answer(output_text)
            gt = normalize_answer_letter(gt_answer) if is_multiple_choice(question) else gt_answer.strip()
            is_correct = int(pred == gt) if pred else 0
            
            input_time_with_preprocess_sec = cur_preprocess_time + t_processor
            total_time_sec = input_time_with_preprocess_sec + t_generate
            n_crops_sent = len(vlm_crops) if method != "Single_VLM_Qwen_4B" else 0
            
            print(
                f"[Result] Pred: {pred} | GT: {gt} | Correct: {is_correct} | "
                f"images={len(vlm_images)} tok={total_input_tokens} peak={peak_mb:.0f}MB | "
                f"Total: {total_time_sec:.2f}s"
            )

            all_results[method].append(
                make_sample_record(
                    benchmark="hrbench_4k",
                    method=method,
                    sample_index=idx,
                    correct=is_correct,
                    predicted_answer=pred,
                    gt_answer=gt,
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
            )

            del inputs, generated_ids, generated_ids_trimmed
            free_memory()
            
    print("=== Phase 2 Complete ===")
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=5, help="Number of samples to run")
    parser.add_argument(
        "--max-crops",
        type=int,
        default=DEFAULT_YOLO_MAX_CROPS,
        help=f"YOLO→Qwen crop 상한 (기본 {DEFAULT_YOLO_MAX_CROPS})",
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

    legacy_prefix = args.output_prefix.strip() or None
    run_dir = resolve_run_dir(args.output) if not legacy_prefix else None
    if run_dir is not None:
        print(f"결과 폴더: {run_dir}", flush=True)
        run_config = {
            "benchmark": "hrbench_4k",
            "max_samples": args.max_samples,
            "methods": ["Smol_YOLO_Qwen_4B", "Single_VLM_Qwen_4B", "Thumb_Smol_YOLO_Qwen_4B"],
            "smol_model": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "qwen_model": "Qwen/Qwen3-VL-4B-Instruct",
            "thumb_max_side": THUMB_MAX_SIDE,
            "single_vlm_max_side": SINGLE_VLM_MAX_SIDE,
            "crop_max_side": CROP_MAX_SIDE,
            "yolo_context_scale": YOLO_CONTEXT_SCALE,
            "yolo_max_crops": args.max_crops,
            "crop_area_filter": f"{CROP_AREA_NUMERATOR}/{CROP_AREA_DENOMINATOR}",
            "yolo_device": YOLO_DEVICE,
            "note": (
                "Smol_YOLO: Smol 텍스트 + 리사이즈 crop 만 (전망 없음). "
                "prefill_time_sec=processor_time; decode_time_sec=generate 전체 (11주차-3 동일)."
            ),
            "run_stamp": format_run_stamp(),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    ds = load_hrbench_4k()

    descriptions, smol_metrics = run_smol_phase(ds, max_samples=args.max_samples)

    all_results = run_qwen_phase(
        ds,
        descriptions,
        smol_metrics,
        max_samples=args.max_samples,
        max_crops=args.max_crops,
    )

    benchmark_name = "hrbench_4k"
    if str(QWEN_SMOL_DIR) not in sys.path:
        sys.path.insert(0, str(QWEN_SMOL_DIR))
    from summary import write_sample_csv, write_sample_jsonl, write_summary_csvs

    for method, results in all_results.items():
        if not results:
            continue

        if run_dir is not None:
            output_file = run_dir / f"{method}.csv"
            samples_dir = run_dir / "samples"
            sample_path = samples_dir / f"{method}.jsonl"
            write_sample_csv(results, output_file)
            write_sample_jsonl(results, sample_path)
        else:
            output_file = Path(f"{legacy_prefix}_{method}.csv")
            write_sample_csv(results, output_file)
        acc = sum(r.get("correct", 0) for r in results) / len(results) * 100
        print(f"[{method}] Results saved to {output_file}")
        print(f"[{method}] Accuracy: {acc:.2f}%")

    if run_dir is not None:
        summary_path, comparison_path = write_summary_csvs(
            run_dir, all_results, benchmark=benchmark_name,
        )
        print(f"summary.csv (11주차-3 호환): {summary_path}")
        print(f"summary_comparison.csv: {comparison_path}")

        if args.plots:
            _w3 = QWEN_SMOL_DIR.parent / "11주차-3"
            if str(_w3) not in sys.path:
                sys.path.insert(0, str(_w3))
            from week3.plots import generate_plots_from_summary_csv

            plot_paths = generate_plots_from_summary_csv(summary_path, plots_dir=run_dir)
            print(f"그래프 {len(plot_paths)}개 저장:", flush=True)
            for p in plot_paths:
                print(f"  {p}", flush=True)
    elif legacy_prefix:
        _w3 = QWEN_SMOL_DIR.parent / "11주차-3"
        if str(_w3) not in sys.path:
            sys.path.insert(0, str(_w3))
        from summary import build_summary_rows
        from week3.metrics import SUMMARY_COLUMNS

        rows = build_summary_rows(all_results, benchmark=benchmark_name)
        df = pd.DataFrame(rows)
        for c in SUMMARY_COLUMNS:
            if c not in df.columns:
                df[c] = None
        comparison_path = Path(f"{legacy_prefix}_summary_comparison.csv")
        df[SUMMARY_COLUMNS].to_csv(comparison_path, index=False, encoding="utf-8-sig")
        print(f"Summary saved to {comparison_path}")

if __name__ == "__main__":
    main()
