"""11주차-2 CSV 와 동일한 요약 컬럼·집계."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# 11주차-2 / 9주차 all_benchmarks_summary 와 동일
SUMMARY_COLUMNS: list[str] = [
    "benchmark",
    "method",
    "rows",
    "correct",
    "accuracy",
    "avg_preprocess_time_sec",
    "min_preprocess_time_sec",
    "max_preprocess_time_sec",
    "avg_processor_time_sec",
    "min_processor_time_sec",
    "max_processor_time_sec",
    "avg_prefill_time_sec",
    "min_prefill_time_sec",
    "max_prefill_time_sec",
    "avg_decode_time_sec",
    "min_decode_time_sec",
    "max_decode_time_sec",
    "avg_output_time_sec",
    "min_output_time_sec",
    "max_output_time_sec",
    "avg_input_time_with_preprocess_sec",
    "min_input_time_with_preprocess_sec",
    "max_input_time_with_preprocess_sec",
    "avg_total_time_sec",
    "min_total_time_sec",
    "max_total_time_sec",
    "avg_text_tokens",
    "min_text_tokens",
    "max_text_tokens",
    "avg_image_tokens",
    "min_image_tokens",
    "max_image_tokens",
    "avg_total_tokens",
    "min_total_tokens",
    "max_total_tokens",
    "avg_prefill_tokens_per_sec",
    "min_prefill_tokens_per_sec",
    "max_prefill_tokens_per_sec",
    "avg_decode_tokens_per_sec",
    "min_decode_tokens_per_sec",
    "max_decode_tokens_per_sec",
    "avg_prefill_mem_peak_allocated_mb",
    "min_prefill_mem_peak_allocated_mb",
    "max_prefill_mem_peak_allocated_mb",
    "avg_decode_mem_peak_allocated_mb",
    "min_decode_mem_peak_allocated_mb",
    "max_decode_mem_peak_allocated_mb",
    "avg_overall_peak_allocated_mb",
    "min_overall_peak_allocated_mb",
    "max_overall_peak_allocated_mb",
    "avg_num_objects",
    "min_num_objects",
    "max_num_objects",
    "accuracy_percent",
]


@dataclass
class SampleMetrics:
    sample_index: int = -1
    gt_answer: str = ""
    predicted_answer: str = ""
    model_response: str = ""
    detector_summary: str = ""
    correct: int | None = None
    preprocess_time_sec: float = 0.0
    processor_time_sec: float = 0.0
    prefill_time_sec: float = 0.0
    decode_time_sec: float = 0.0
    output_time_sec: float = 0.0
    input_time_with_preprocess_sec: float = 0.0
    total_time_sec: float = 0.0
    text_tokens: float = 0.0
    image_tokens: float = 0.0
    total_tokens: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    prefill_mem_peak_allocated_mb: float = 0.0
    decode_mem_peak_allocated_mb: float = 0.0
    overall_peak_allocated_mb: float = 0.0
    num_objects: float = 0.0
    yolo_time_sec: float = 0.0
    n_detections: int = 0
    n_crops_sent: int = 0


@dataclass
class RunAccumulator:
    benchmark: str
    method: str
    rows: list[SampleMetrics] = field(default_factory=list)

    def add(self, m: SampleMetrics) -> None:
        self.rows.append(m)


def _agg(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    return float(sum(vals) / len(vals)), float(min(vals)), float(max(vals))


def summarize(acc: RunAccumulator) -> dict[str, Any]:
    r = acc.rows
    n = len(r)
    correct_vals = [x.correct for x in r if x.correct is not None]
    correct_sum = sum(correct_vals) if correct_vals else 0
    acc_rows = len(correct_vals) if correct_vals else n
    accuracy = (correct_sum / acc_rows) if acc_rows else 0.0

    def col(getter: Any) -> tuple[float, float, float]:
        return _agg([getter(x) for x in r])

    pre_avg, pre_min, pre_max = col(lambda x: x.preprocess_time_sec)
    proc_avg, proc_min, proc_max = col(lambda x: x.processor_time_sec)
    pref_avg, pref_min, pref_max = col(lambda x: x.prefill_time_sec)
    dec_avg, dec_min, dec_max = col(lambda x: x.decode_time_sec)
    out_avg, out_min, out_max = col(lambda x: x.output_time_sec)
    inp_avg, inp_min, inp_max = col(lambda x: x.input_time_with_preprocess_sec)
    tot_avg, tot_min, tot_max = col(lambda x: x.total_time_sec)
    txt_avg, txt_min, txt_max = col(lambda x: x.text_tokens)
    img_avg, img_min, img_max = col(lambda x: x.image_tokens)
    all_avg, all_min, all_max = col(lambda x: x.total_tokens)
    pps_avg, pps_min, pps_max = col(lambda x: x.prefill_tokens_per_sec)
    dps_avg, dps_min, dps_max = col(lambda x: x.decode_tokens_per_sec)
    pm_avg, pm_min, pm_max = col(lambda x: x.prefill_mem_peak_allocated_mb)
    dm_avg, dm_min, dm_max = col(lambda x: x.decode_mem_peak_allocated_mb)
    om_avg, om_min, om_max = col(lambda x: x.overall_peak_allocated_mb)
    obj_avg, obj_min, obj_max = col(lambda x: x.num_objects)

    return {
        "benchmark": acc.benchmark,
        "method": acc.method,
        "rows": n,
        "correct": correct_sum,
        "accuracy": accuracy,
        "avg_preprocess_time_sec": pre_avg,
        "min_preprocess_time_sec": pre_min,
        "max_preprocess_time_sec": pre_max,
        "avg_processor_time_sec": proc_avg,
        "min_processor_time_sec": proc_min,
        "max_processor_time_sec": proc_max,
        "avg_prefill_time_sec": pref_avg,
        "min_prefill_time_sec": pref_min,
        "max_prefill_time_sec": pref_max,
        "avg_decode_time_sec": dec_avg,
        "min_decode_time_sec": dec_min,
        "max_decode_time_sec": dec_max,
        "avg_output_time_sec": out_avg,
        "min_output_time_sec": out_min,
        "max_output_time_sec": out_max,
        "avg_input_time_with_preprocess_sec": inp_avg,
        "min_input_time_with_preprocess_sec": inp_min,
        "max_input_time_with_preprocess_sec": inp_max,
        "avg_total_time_sec": tot_avg,
        "min_total_time_sec": tot_min,
        "max_total_time_sec": tot_max,
        "avg_text_tokens": txt_avg,
        "min_text_tokens": txt_min,
        "max_text_tokens": txt_max,
        "avg_image_tokens": img_avg,
        "min_image_tokens": img_min,
        "max_image_tokens": img_max,
        "avg_total_tokens": all_avg,
        "min_total_tokens": all_min,
        "max_total_tokens": all_max,
        "avg_prefill_tokens_per_sec": pps_avg,
        "min_prefill_tokens_per_sec": pps_min,
        "max_prefill_tokens_per_sec": pps_max,
        "avg_decode_tokens_per_sec": dps_avg,
        "min_decode_tokens_per_sec": dps_min,
        "max_decode_tokens_per_sec": dps_max,
        "avg_prefill_mem_peak_allocated_mb": pm_avg,
        "min_prefill_mem_peak_allocated_mb": pm_min,
        "max_prefill_mem_peak_allocated_mb": pm_max,
        "avg_decode_mem_peak_allocated_mb": dm_avg,
        "min_decode_mem_peak_allocated_mb": dm_min,
        "max_decode_mem_peak_allocated_mb": dm_max,
        "avg_overall_peak_allocated_mb": om_avg,
        "min_overall_peak_allocated_mb": om_min,
        "max_overall_peak_allocated_mb": om_max,
        "avg_num_objects": obj_avg,
        "min_num_objects": obj_min,
        "max_num_objects": obj_max,
        "accuracy_percent": accuracy * 100.0,
    }


def sample_to_log_record(
    m: SampleMetrics,
    *,
    benchmark: str,
    method: str,
) -> dict[str, Any]:
    """샘플별 분석용 JSONL 한 줄."""
    d = asdict(m)
    d["benchmark"] = benchmark
    d["method"] = method
    return d


def append_sample_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_csv(rows: list[dict[str, Any]], path: Any) -> None:
    df = pd.DataFrame(rows)
    for c in SUMMARY_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[SUMMARY_COLUMNS]
    path = Path(path) if not isinstance(path, type(Path)) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Excel 에서 한글이 깨지지 않도록 UTF-8 BOM
    df.to_csv(path, index=False, encoding="utf-8-sig")
