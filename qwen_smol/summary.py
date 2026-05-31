"""qwen_smol 벤치 결과 → 11주차-3 SampleMetrics / summary.csv 호환."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_W3 = _REPO / "11주차-3"
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))

from week3.metrics import (  # noqa: E402
    SUMMARY_COLUMNS,
    RunAccumulator,
    SampleMetrics,
    sample_to_log_record,
    summarize,
)

# 11주차-3 SampleMetrics + benchmark/method (per-sample CSV·JSONL 공통)
SAMPLE_CSV_COLUMNS: list[str] = [
    "benchmark",
    "method",
    "sample_index",
    "correct",
    "predicted_answer",
    "gt_answer",
    "model_response",
    "detector_summary",
    "preprocess_time_sec",
    "yolo_time_sec",
    "processor_time_sec",
    "prefill_time_sec",
    "decode_time_sec",
    "output_time_sec",
    "input_time_with_preprocess_sec",
    "total_time_sec",
    "text_tokens",
    "image_tokens",
    "total_tokens",
    "prefill_tokens_per_sec",
    "decode_tokens_per_sec",
    "prefill_mem_peak_allocated_mb",
    "decode_mem_peak_allocated_mb",
    "overall_peak_allocated_mb",
    "num_objects",
    "n_detections",
    "n_crops_sent",
]

# qwen_smol 전용 (표준 컬럼 뒤에 추가)
QWEN_SMOL_EXTRA_SAMPLE_COLUMNS: list[str] = [
    "smol_time_sec",
    "smol_peak_mem_mb",
    "smol_description",
    "question",
    "num_vlm_images",
]

ALL_SAMPLE_CSV_COLUMNS = SAMPLE_CSV_COLUMNS + QWEN_SMOL_EXTRA_SAMPLE_COLUMNS


def make_sample_record(
    *,
    benchmark: str,
    method: str,
    sample_index: int,
    correct: int | None,
    predicted_answer: str,
    gt_answer: str,
    model_response: str,
    detector_summary: str = "",
    preprocess_time_sec: float = 0.0,
    yolo_time_sec: float = 0.0,
    processor_time_sec: float = 0.0,
    prefill_time_sec: float = 0.0,
    decode_time_sec: float = 0.0,
    input_time_with_preprocess_sec: float = 0.0,
    total_time_sec: float = 0.0,
    text_tokens: float = 0.0,
    image_tokens: float = 0.0,
    total_tokens: float = 0.0,
    prefill_tokens_per_sec: float = 0.0,
    decode_tokens_per_sec: float = 0.0,
    prefill_mem_peak_allocated_mb: float = 0.0,
    decode_mem_peak_allocated_mb: float = 0.0,
    overall_peak_allocated_mb: float = 0.0,
    n_crops_sent: int = 0,
    n_detections: int = 0,
    smol_time_sec: float = 0.0,
    smol_peak_mem_mb: float = 0.0,
    smol_description: str = "",
    question: str = "",
    num_vlm_images: int = 0,
) -> dict[str, Any]:
    """11주차-3 ``SampleMetrics`` 필드명으로 샘플 한 줄 dict."""
    m = SampleMetrics(
        sample_index=sample_index,
        correct=correct,
        predicted_answer=predicted_answer,
        gt_answer=gt_answer,
        model_response=model_response,
        detector_summary=detector_summary,
        preprocess_time_sec=preprocess_time_sec,
        yolo_time_sec=yolo_time_sec,
        processor_time_sec=processor_time_sec,
        prefill_time_sec=prefill_time_sec,
        decode_time_sec=decode_time_sec,
        output_time_sec=decode_time_sec,
        input_time_with_preprocess_sec=input_time_with_preprocess_sec,
        total_time_sec=total_time_sec,
        text_tokens=text_tokens,
        image_tokens=image_tokens,
        total_tokens=total_tokens,
        prefill_tokens_per_sec=prefill_tokens_per_sec,
        decode_tokens_per_sec=decode_tokens_per_sec,
        prefill_mem_peak_allocated_mb=prefill_mem_peak_allocated_mb,
        decode_mem_peak_allocated_mb=decode_mem_peak_allocated_mb,
        overall_peak_allocated_mb=overall_peak_allocated_mb,
        num_objects=float(n_crops_sent),
        n_detections=n_detections,
        n_crops_sent=n_crops_sent,
    )
    rec = sample_to_log_record(m, benchmark=benchmark, method=method)
    rec.update(
        {
            "smol_time_sec": smol_time_sec,
            "smol_peak_mem_mb": smol_peak_mem_mb,
            "smol_description": smol_description,
            "question": question,
            "num_vlm_images": num_vlm_images,
        }
    )
    return rec


def _row_to_sample_metrics(row: dict[str, Any]) -> SampleMetrics:
    """레거시(is_correct, num_crops 등)·신규(correct, n_crops_sent) 모두 수용."""
    proc = float(row.get("processor_time_sec") or 0.0)
    decode = float(row.get("decode_time_sec") or row.get("generate_time_sec") or 0.0)
    in_tok = float(row.get("total_tokens") or 0.0)

    if "correct" in row and row["correct"] is not None and str(row["correct"]) != "nan":
        correct = int(row["correct"])
    else:
        correct = int(row.get("is_correct") or 0)

    if "sample_index" in row and row["sample_index"] is not None:
        sample_index = int(row["sample_index"])
    else:
        sample_index = int(row.get("dataset_index") or -1)

    n_crops = int(row.get("n_crops_sent") if row.get("n_crops_sent") is not None else row.get("num_crops") or 0)
    n_det = int(row.get("n_detections") if row.get("n_detections") is not None else row.get("num_objects") or 0)

    if "num_objects" in row and row.get("n_crops_sent") is None and row.get("num_crops") is None:
        # 레거시: num_objects 가 탐지 수였을 수 있음 → n_crops 우선 없으면 그대로
        num_objects = float(row.get("num_objects") or 0.0)
    else:
        num_objects = float(n_crops)

    return SampleMetrics(
        sample_index=sample_index,
        gt_answer=str(row.get("gt_answer") or ""),
        predicted_answer=str(row.get("predicted_answer") or ""),
        model_response=str(row.get("model_response") or row.get("raw_output") or ""),
        detector_summary=str(row.get("detector_summary") or ""),
        correct=correct,
        preprocess_time_sec=float(row.get("preprocess_time_sec") or 0.0),
        processor_time_sec=proc,
        prefill_time_sec=float(row.get("prefill_time_sec") if row.get("prefill_time_sec") is not None else proc),
        decode_time_sec=decode,
        output_time_sec=float(row.get("output_time_sec") if row.get("output_time_sec") is not None else decode),
        input_time_with_preprocess_sec=float(row.get("input_time_with_preprocess_sec") or 0.0),
        total_time_sec=float(row.get("total_time_sec") or 0.0),
        text_tokens=float(row.get("text_tokens") or 0.0),
        image_tokens=float(row.get("image_tokens") or 0.0),
        total_tokens=in_tok,
        prefill_tokens_per_sec=float(
            row.get("prefill_tokens_per_sec") or (in_tok / proc if proc > 0 else 0.0)
        ),
        decode_tokens_per_sec=float(row.get("decode_tokens_per_sec") or 0.0),
        prefill_mem_peak_allocated_mb=float(row.get("prefill_mem_peak_allocated_mb") or 0.0),
        decode_mem_peak_allocated_mb=float(row.get("decode_mem_peak_allocated_mb") or 0.0),
        overall_peak_allocated_mb=float(row.get("overall_peak_allocated_mb") or 0.0),
        num_objects=num_objects,
        yolo_time_sec=float(row.get("yolo_time_sec") or 0.0),
        n_detections=n_det,
        n_crops_sent=n_crops,
    )


def write_sample_csv(records: list[dict[str, Any]], path: Path) -> None:
    """per-method CSV — 표준 컬럼 순서 고정."""
    df = pd.DataFrame(records)
    for c in ALL_SAMPLE_CSV_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df[ALL_SAMPLE_CSV_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")


def write_sample_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary_rows(
    all_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """``all_results`` 키는 ``{method}`` 또는 ``{benchmark}_{method}``."""
    summary_rows: list[dict[str, Any]] = []
    for _key, results in all_results.items():
        if not results:
            continue
        benchmark = str(results[0].get("benchmark") or "hrbench_4k")
        method = str(results[0].get("method") or _key)
        acc = RunAccumulator(benchmark=benchmark, method=method)
        for row in results:
            acc.add(_row_to_sample_metrics(row))
        summary_rows.append(summarize(acc))
    return summary_rows


def write_summary_csvs(
    run_dir: Path,
    all_results: dict[str, list[dict[str, Any]]],
) -> tuple[Path, Path]:
    summary_rows = build_summary_rows(all_results)

    summary_path = run_dir / "summary.csv"
    df = pd.DataFrame(summary_rows)
    for c in SUMMARY_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df[SUMMARY_COLUMNS].to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 11주차-3 와 동일 컬럼 (별칭 파일)
    comparison_path = run_dir / "summary_comparison.csv"
    df[SUMMARY_COLUMNS].to_csv(comparison_path, index=False, encoding="utf-8-sig")

    return summary_path, comparison_path


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """레거시 필드명 → ``make_sample_record`` 호환 dict (SampleMetrics 기준)."""
    out: list[dict[str, Any]] = []
    for row in records:
        m = _row_to_sample_metrics(row)
        rec = sample_to_log_record(
            m,
            benchmark=str(row.get("benchmark") or "hrbench_4k"),
            method=str(row.get("method") or ""),
        )
        for c in QWEN_SMOL_EXTRA_SAMPLE_COLUMNS:
            if c in row:
                rec[c] = row[c]
            elif c == "smol_description":
                rec[c] = row.get("smol_description", "")
            elif c == "question":
                rec[c] = row.get("question", "")
            elif c == "num_vlm_images":
                rec[c] = row.get("num_vlm_images", 0)
            elif c == "smol_time_sec":
                rec[c] = row.get("smol_time_sec", 0.0)
            elif c == "smol_peak_mem_mb":
                rec[c] = row.get("smol_peak_mem_mb", 0.0)
        out.append(rec)
    return out


def iter_sample_csv_paths(run_dir: Path) -> list[Path]:
    """``{benchmark}_{method}.csv`` 및 레거시 ``{method}.csv``."""
    seen: set[str] = set()
    paths: list[Path] = []
    for pattern in ("*_*_Qwen_4B.csv", "*_Qwen_4B.csv"):
        for csv_path in sorted(run_dir.glob(pattern)):
            if csv_path.name in seen:
                continue
            seen.add(csv_path.name)
            paths.append(csv_path)
    return paths


def normalize_run_dir(run_dir: Path) -> None:
    """run 폴더 내 method CSV·jsonl·summary 를 11주차-3 컬럼명으로 재저장."""
    run_dir = run_dir.resolve()
    all_results: dict[str, list[dict[str, Any]]] = {}
    for csv_path in iter_sample_csv_paths(run_dir):
        key = csv_path.stem
        raw = pd.read_csv(csv_path).to_dict(orient="records")
        records = normalize_records(raw)
        all_results[key] = records
        write_sample_csv(records, csv_path)
        write_sample_jsonl(records, run_dir / "samples" / f"{key}.jsonl")
    if all_results:
        write_summary_csvs(run_dir, all_results)


def load_results_from_run_dir(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for csv_path in iter_sample_csv_paths(run_dir):
        key = csv_path.stem
        df = pd.read_csv(csv_path)
        out[key] = df.to_dict(orient="records")
    return out
