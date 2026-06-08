#!/usr/bin/env python3
"""Ablation 해상도/스케일 스윕 벤치.

대조군 (control): Single_VLM Qwen 입력 긴 변 px 스윕 (360→1280)
Ablation (나머지 고정, 해당 축만 변화):
  1. smol      — SmolVLM 입력 긴 변 px
  2. crop_qwen — YOLO crop → Qwen 전송 전 균일 축소 배율
  3. thumb     — Thumb_Smol 썸네일 긴 변 px
  4. yolo_ctx  — YOLO 탐지 ctx 축소 배율
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

QWEN_SMOL_DIR = Path(__file__).resolve().parent
_REPO = QWEN_SMOL_DIR.parent
_W3 = _REPO / "11주차-3"
for p in (_REPO, _W3, QWEN_SMOL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_runner import (  # noqa: E402
    DEFAULT_BENCHMARKS,
    DEFAULT_CROP_QWEN_SCALE,
    DEFAULT_SMOL_MAX_SIDE,
    METHODS,
    SINGLE_VLM_MAX_SIDE,
    THUMB_MAX_SIDE,
    YOLO_CONTEXT_SCALE,
    format_run_stamp,
    load_qwen_bundle,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    resolve_run_dir,
    run_qwen_phase,
    run_smol_phase,
    sample_range,
    unload_qwen_bundle,
)
from qwen_vlm.vision.yolo import load_yolo  # noqa: E402
from summary import (  # noqa: E402
    build_summary_rows,
    enrich_dual_model_summary,
    find_results_for_summary,
    write_sample_csv,
    write_sample_jsonl,
    write_summary_csvs,
)
from week3.datasets import load_benchmark_dataset  # noqa: E402

DEFAULT_BASE_CONFIG = QWEN_SMOL_DIR / "qwen_smol_20260531_143238" / "run_config.json"
RUN_DIR_PREFIX = "qwen_smol_resolution_sweep"

DEFAULT_SWEEP_SIDES = [360, 480, 540, 640, 720, 800, 960, 1080, 1280]
DEFAULT_SCALE_VALUES = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 1.0]

# ablation 시 고정 baseline (스윕하지 않는 축)
SWEEP_BASELINE_SMOL_SIDE = 1280
SWEEP_BASELINE_THUMB_SIDE = THUMB_MAX_SIDE
SWEEP_BASELINE_CROP_SCALE = DEFAULT_CROP_QWEN_SCALE
SWEEP_BASELINE_YOLO_CTX = YOLO_CONTEXT_SCALE

SMOL_METHODS = ["Smol_YOLO_Qwen_4B", "Thumb_Smol_YOLO_Qwen_4B"]
SINGLE_METHOD = "Single_VLM_Qwen_4B"
THUMB_METHOD = "Thumb_Smol_YOLO_Qwen_4B"

SWEEP_CHOICES = ["control", "smol", "crop_qwen", "thumb", "yolo_ctx", "all"]
LEGACY_SWEEP_ALIASES = {
    "single_vlm": "control",
    "both": "control,smol",
}


def side_tag(side: int, *, prefix: str) -> str:
    return f"_{prefix}{side}" if side > 0 else f"_{prefix}Orig"


def scale_tag(scale: float, *, prefix: str) -> str:
    pct = int(round(scale * 100))
    return f"_{prefix}{pct:03d}"


def load_base_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_slices(
    benchmarks: list[str],
    start: int,
    max_samples: int,
) -> list[tuple[str, Any, int, int]]:
    slices: list[tuple[str, Any, int, int]] = []
    for bench in benchmarks:
        ds = load_benchmark_dataset(bench)
        s, e = sample_range(len(ds), start, max_samples)
        if s >= e:
            print(f"[경고] {bench}: 샘플 범위 비어 있음 (start={s}, end={e})", flush=True)
            continue
        slices.append((bench, ds, s, e))
        print(f"{bench}: rows {s}:{e} / {len(ds)}", flush=True)
    return slices


def empty_smol_cache(benchmarks: list[str]) -> tuple[dict, dict]:
    return ({b: {} for b in benchmarks}, {b: {} for b in benchmarks})


def merge_results(dst: dict[str, list[dict]], src: dict[str, list[dict]]) -> None:
    for key, rows in src.items():
        dst.setdefault(key, []).extend(rows)


def sweep_value_from_record(meta: dict[str, Any]) -> float | int | None:
    if meta.get("sweep_value") is not None:
        return meta.get("sweep_value")
    axis = str(meta.get("sweep_axis") or "")
    if axis in ("control", "single_vlm"):
        return meta.get("single_vlm_max_side")
    if axis == "smol":
        return meta.get("smol_max_side")
    if axis == "crop_qwen":
        return meta.get("crop_qwen_scale")
    if axis == "thumb":
        return meta.get("thumb_max_side")
    if axis == "yolo_ctx":
        return meta.get("yolo_context_scale")
    return None


def write_sweep_summary(run_dir: Path, all_results: dict[str, list[dict]]) -> Path:
    summary_rows = build_summary_rows(all_results)
    fixed_rows: list[dict[str, Any]] = []
    for s in summary_rows:
        method = str(s.get("method") or "")
        benchmark = str(s.get("benchmark") or "")
        matching = find_results_for_summary(
            all_results, benchmark=benchmark, method=method
        )
        meta = matching[0] if matching else {}
        axis = str(meta.get("sweep_axis") or "")
        sweep_val = sweep_value_from_record(meta)
        enriched = enrich_dual_model_summary(s, matching)
        fixed_rows.append(
            {
                "sweep_axis": axis,
                "sweep_value": sweep_val,
                "max_side": sweep_val if axis in ("control", "single_vlm", "smol", "thumb") else None,
                "benchmark": benchmark,
                "method": method,
                "rows": enriched.get("rows"),
                "correct": enriched.get("correct"),
                "accuracy": enriched.get("accuracy"),
                "accuracy_percent": enriched.get("accuracy_percent"),
                "avg_total_time_sec": enriched.get("avg_total_time_sec"),
                "avg_smol_image_tokens": enriched.get("avg_smol_image_tokens"),
                "avg_smol_peak_mem_mb": enriched.get("avg_smol_peak_mem_mb"),
                "avg_qwen_image_tokens": enriched.get("avg_qwen_image_tokens"),
                "avg_qwen_peak_mem_mb": enriched.get("avg_qwen_peak_mem_mb"),
                # 레거시 alias
                "avg_image_tokens": enriched.get("avg_qwen_image_tokens"),
                "avg_overall_peak_allocated_mb": enriched.get("avg_qwen_peak_mem_mb"),
            }
        )
    out = run_dir / "resolution_sweep_summary.csv"
    pd.DataFrame(fixed_rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def run_control_sweep(
    slices: list[tuple[str, Any, int, int]],
    *,
    sides: list[int],
    max_crops: int,
    benchmarks: list[str],
) -> dict[str, list[dict]]:
    print("\n========== Control: Single_VLM (Qwen 입력 px) ==========", flush=True)
    device = "cuda:0"
    qwen_bundle = load_qwen_bundle(device)
    yolo_model = load_yolo()
    all_results: dict[str, list[dict]] = {}
    empty_desc, empty_metrics = empty_smol_cache(benchmarks)

    try:
        for side in sides:
            tag = side_tag(side, prefix="S")
            print(f"\n--- Single_VLM max_side={side} ({tag}) ---", flush=True)
            chunk = run_qwen_phase(
                slices,
                empty_desc,
                empty_metrics,
                max_crops=max_crops,
                single_vlm_max_side=side,
                smol_max_side=SWEEP_BASELINE_SMOL_SIDE,
                thumb_max_side=SWEEP_BASELINE_THUMB_SIDE,
                crop_qwen_scale=SWEEP_BASELINE_CROP_SCALE,
                yolo_context_scale=SWEEP_BASELINE_YOLO_CTX,
                methods=[SINGLE_METHOD],
                method_tag=tag,
                sweep_axis="control",
                qwen_bundle=qwen_bundle,
                yolo_model=yolo_model,
                unload_at_end=False,
            )
            merge_results(all_results, chunk)
    finally:
        unload_qwen_bundle(qwen_bundle)
        del yolo_model
    return all_results


def run_smol_sweep(
    slices: list[tuple[str, Any, int, int]],
    *,
    sides: list[int],
    max_crops: int,
) -> dict[str, list[dict]]:
    print("\n========== Ablation 1: SmolVLM 입력 px (나머지 baseline 고정) ==========", flush=True)
    # 2-pass: Smol 전 side를 Qwen 미로드 상태에서 먼저 실행 → VRAM·시간 공정 비교
    yolo_model = load_yolo()
    qwen_bundle = None
    all_results: dict[str, list[dict]] = {}
    smol_runs: list[tuple[int, str, dict, dict]] = []

    try:
        print("\n--- Pass 1/2: SmolVLM 모든 side (GPU에 Qwen 없음) ---", flush=True)
        for smol_side in sides:
            tag = side_tag(smol_side, prefix="smol")
            label = "original" if smol_side <= 0 else f"{smol_side}px"
            print(f"\n--- SmolVLM max_side={label} ({tag}) ---", flush=True)
            descriptions, smol_metrics = run_smol_phase(
                slices,
                smol_max_side=smol_side,
            )
            smol_runs.append((smol_side, tag, descriptions, smol_metrics))

        print("\n--- Pass 2/2: Qwen+YOLO 모든 side ---", flush=True)
        qwen_bundle = load_qwen_bundle("cuda:0")
        for smol_side, tag, descriptions, smol_metrics in smol_runs:
            label = "original" if smol_side <= 0 else f"{smol_side}px"
            print(f"\n--- Qwen eval (smol_max_side={label}, {tag}) ---", flush=True)
            chunk = run_qwen_phase(
                slices,
                descriptions,
                smol_metrics,
                max_crops=max_crops,
                single_vlm_max_side=SINGLE_VLM_MAX_SIDE,
                smol_max_side=smol_side,
                thumb_max_side=SWEEP_BASELINE_THUMB_SIDE,
                crop_qwen_scale=SWEEP_BASELINE_CROP_SCALE,
                yolo_context_scale=SWEEP_BASELINE_YOLO_CTX,
                methods=SMOL_METHODS,
                method_tag=tag,
                sweep_axis="smol",
                qwen_bundle=qwen_bundle,
                yolo_model=yolo_model,
                unload_at_end=False,
            )
            merge_results(all_results, chunk)
    finally:
        unload_qwen_bundle(qwen_bundle)
        del yolo_model
    return all_results


def run_crop_qwen_sweep(
    slices: list[tuple[str, Any, int, int]],
    *,
    scales: list[float],
    max_crops: int,
) -> dict[str, list[dict]]:
    print("\n========== Ablation 2: crop → Qwen 축소 배율 (나머지 baseline 고정) ==========", flush=True)
    descriptions, smol_metrics = run_smol_phase(
        slices,
        smol_max_side=SWEEP_BASELINE_SMOL_SIDE,
    )
    qwen_bundle = load_qwen_bundle("cuda:0")
    yolo_model = load_yolo()
    all_results: dict[str, list[dict]] = {}

    try:
        for scale in scales:
            tag = scale_tag(scale, prefix="crop")
            print(f"\n--- crop_qwen_scale={scale} ({tag}) ---", flush=True)
            chunk = run_qwen_phase(
                slices,
                descriptions,
                smol_metrics,
                max_crops=max_crops,
                single_vlm_max_side=SINGLE_VLM_MAX_SIDE,
                smol_max_side=SWEEP_BASELINE_SMOL_SIDE,
                thumb_max_side=SWEEP_BASELINE_THUMB_SIDE,
                crop_qwen_scale=scale,
                yolo_context_scale=SWEEP_BASELINE_YOLO_CTX,
                methods=SMOL_METHODS,
                method_tag=tag,
                sweep_axis="crop_qwen",
                qwen_bundle=qwen_bundle,
                yolo_model=yolo_model,
                unload_at_end=False,
            )
            merge_results(all_results, chunk)
    finally:
        unload_qwen_bundle(qwen_bundle)
        del yolo_model
    return all_results


def run_thumb_sweep(
    slices: list[tuple[str, Any, int, int]],
    *,
    sides: list[int],
    max_crops: int,
) -> dict[str, list[dict]]:
    print("\n========== Ablation 3: Thumb_Smol 썸네일 px (나머지 baseline 고정) ==========", flush=True)
    descriptions, smol_metrics = run_smol_phase(
        slices,
        smol_max_side=SWEEP_BASELINE_SMOL_SIDE,
    )
    qwen_bundle = load_qwen_bundle("cuda:0")
    yolo_model = load_yolo()
    all_results: dict[str, list[dict]] = {}

    try:
        for thumb_side in sides:
            tag = side_tag(thumb_side, prefix="thumb")
            print(f"\n--- thumb_max_side={thumb_side} ({tag}) ---", flush=True)
            chunk = run_qwen_phase(
                slices,
                descriptions,
                smol_metrics,
                max_crops=max_crops,
                single_vlm_max_side=SINGLE_VLM_MAX_SIDE,
                smol_max_side=SWEEP_BASELINE_SMOL_SIDE,
                thumb_max_side=thumb_side,
                crop_qwen_scale=SWEEP_BASELINE_CROP_SCALE,
                yolo_context_scale=SWEEP_BASELINE_YOLO_CTX,
                methods=SMOL_METHODS,
                method_tag=tag,
                sweep_axis="thumb",
                qwen_bundle=qwen_bundle,
                yolo_model=yolo_model,
                unload_at_end=False,
            )
            merge_results(all_results, chunk)
    finally:
        unload_qwen_bundle(qwen_bundle)
        del yolo_model
    return all_results


def run_yolo_ctx_sweep(
    slices: list[tuple[str, Any, int, int]],
    *,
    scales: list[float],
    max_crops: int,
) -> dict[str, list[dict]]:
    print("\n========== Ablation 4: YOLO ctx 축소 배율 (나머지 baseline 고정) ==========", flush=True)
    descriptions, smol_metrics = run_smol_phase(
        slices,
        smol_max_side=SWEEP_BASELINE_SMOL_SIDE,
    )
    qwen_bundle = load_qwen_bundle("cuda:0")
    yolo_model = load_yolo()
    all_results: dict[str, list[dict]] = {}

    try:
        for ctx_scale in scales:
            tag = scale_tag(ctx_scale, prefix="ctx")
            print(f"\n--- yolo_context_scale={ctx_scale} ({tag}) ---", flush=True)
            chunk = run_qwen_phase(
                slices,
                descriptions,
                smol_metrics,
                max_crops=max_crops,
                single_vlm_max_side=SINGLE_VLM_MAX_SIDE,
                smol_max_side=SWEEP_BASELINE_SMOL_SIDE,
                thumb_max_side=SWEEP_BASELINE_THUMB_SIDE,
                crop_qwen_scale=SWEEP_BASELINE_CROP_SCALE,
                yolo_context_scale=ctx_scale,
                methods=SMOL_METHODS,
                method_tag=tag,
                sweep_axis="yolo_ctx",
                qwen_bundle=qwen_bundle,
                yolo_model=yolo_model,
                unload_at_end=False,
            )
            merge_results(all_results, chunk)
    finally:
        unload_qwen_bundle(qwen_bundle)
        del yolo_model
    return all_results


def save_all_results(run_dir: Path, all_results: dict[str, list[dict]]) -> None:
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in all_results.items():
        if not rows:
            continue
        write_sample_csv(rows, run_dir / f"{key}.csv")
        write_sample_jsonl(rows, samples_dir / f"{key}.jsonl")
        scored = [r for r in rows if r.get("correct") is not None]
        acc = (
            sum(int(r.get("correct") or 0) for r in scored) / len(scored) * 100
            if scored
            else 0.0
        )
        print(f"[{key}] accuracy={acc:.2f}% n={len(scored)}", flush=True)
    write_summary_csvs(run_dir, all_results)


def parse_sweep_modes(raw: str) -> list[str]:
    text = LEGACY_SWEEP_ALIASES.get(raw.strip(), raw.strip())
    modes: list[str] = []
    for part in text.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if part == "all":
            return ["control", "smol", "crop_qwen", "thumb", "yolo_ctx"]
        if part not in SWEEP_CHOICES:
            raise ValueError(f"알 수 없는 sweep 모드: {part} (choices: {SWEEP_CHOICES})")
        if part not in modes:
            modes.append(part)
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single_VLM 대조군 + Smol/crop/thumb/yolo_ctx ablation 스윕",
    )
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_BASE_CONFIG),
        help="기준 run_config.json (벤치·샘플 수)",
    )
    parser.add_argument(
        "--sweep",
        default="all",
        help=(
            "실행할 스윕: control, smol, crop_qwen, thumb, yolo_ctx, all "
            "(쉼표 구분). legacy: single_vlm, both"
        ),
    )
    parser.add_argument(
        "--sweep-sides",
        default=",".join(str(s) for s in DEFAULT_SWEEP_SIDES),
        help=f"px 스윕 목록 — control/smol/thumb (기본: {DEFAULT_SWEEP_SIDES})",
    )
    parser.add_argument(
        "--sweep-scales",
        default=",".join(str(s) for s in DEFAULT_SCALE_VALUES),
        help=f"배율 스윕 목록 — crop_qwen/yolo_ctx (기본: {DEFAULT_SCALE_VALUES})",
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="쉼표 구분 (비우면 base-config 의 benchmarks)",
    )
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--max-crops", type=int, default=-1)
    parser.add_argument(
        "--output",
        default="",
        help="결과 폴더 (기본: qwen_smol/qwen_smol_resolution_sweep_YYYYMMDD_HHMMSS/)",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        sweep_modes = parse_sweep_modes(args.sweep)
    except ValueError as exc:
        parser.error(str(exc))
    if not sweep_modes:
        parser.error("--sweep 가 비어 있습니다.")

    base = load_base_config(Path(args.base_config))
    benchmarks = parse_str_list(args.benchmarks) or list(base.get("benchmarks") or DEFAULT_BENCHMARKS)
    max_samples = args.max_samples if args.max_samples >= 0 else int(base.get("max_samples") or 50)
    start = args.start if args.start >= 0 else int(base.get("start_index") or 0)
    max_crops = args.max_crops if args.max_crops >= 0 else int(base.get("yolo_max_crops") or 0)

    sweep_sides = sorted(parse_int_list(args.sweep_sides))
    sweep_scales = sorted(parse_float_list(args.sweep_scales))
    if not sweep_sides:
        parser.error("--sweep-sides 가 비어 있습니다.")
    if not sweep_scales:
        parser.error("--sweep-scales 가 비어 있습니다.")
    if any(s <= 0 for s in sweep_sides):
        parser.error("--sweep-sides 는 양의 정수만 허용합니다.")
    if any(s <= 0 or s > 1.0 + 1e-6 for s in sweep_scales):
        parser.error("--sweep-scales 는 (0, 1] 범위의 배율만 허용합니다.")

    run_dir = resolve_run_dir(args.output) if args.output.strip() else (
        QWEN_SMOL_DIR / f"{RUN_DIR_PREFIX}_{format_run_stamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 폴더: {run_dir}", flush=True)
    print(f"스윕 모드: {sweep_modes}", flush=True)
    print(f"px 목록: {sweep_sides}", flush=True)
    print(f"배율 목록: {sweep_scales}", flush=True)

    slices = build_slices(benchmarks, start, max_samples)
    if not slices:
        parser.error("실행할 벤치마크 샘플이 없습니다.")

    run_config = {
        **base,
        "sweep_modes": sweep_modes,
        "sweep_sides_px": sweep_sides,
        "sweep_scales": sweep_scales,
        "ablation_baselines": {
            "smol_max_side": SWEEP_BASELINE_SMOL_SIDE,
            "thumb_max_side": SWEEP_BASELINE_THUMB_SIDE,
            "crop_qwen_scale": SWEEP_BASELINE_CROP_SCALE,
            "yolo_context_scale": SWEEP_BASELINE_YOLO_CTX,
        },
        "base_config_path": str(Path(args.base_config).resolve()),
        "methods_full": METHODS,
        "run_stamp": format_run_stamp(),
        "note": (
            "control=Single_VLM px. ablation: smol px | crop_qwen scale | thumb px | yolo_ctx scale. "
            "각 ablation 은 해당 축만 변화, 나머지 baseline 고정."
        ),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_results: dict[str, list[dict]] = {}

    if "control" in sweep_modes:
        merge_results(
            all_results,
            run_control_sweep(
                slices,
                sides=sweep_sides,
                max_crops=max_crops,
                benchmarks=benchmarks,
            ),
        )
    if "smol" in sweep_modes:
        merge_results(
            all_results,
            run_smol_sweep(slices, sides=sweep_sides, max_crops=max_crops),
        )
    if "crop_qwen" in sweep_modes:
        merge_results(
            all_results,
            run_crop_qwen_sweep(slices, scales=sweep_scales, max_crops=max_crops),
        )
    if "thumb" in sweep_modes:
        merge_results(
            all_results,
            run_thumb_sweep(slices, sides=sweep_sides, max_crops=max_crops),
        )
    if "yolo_ctx" in sweep_modes:
        merge_results(
            all_results,
            run_yolo_ctx_sweep(slices, scales=sweep_scales, max_crops=max_crops),
        )

    save_all_results(run_dir, all_results)
    sweep_summary = write_sweep_summary(run_dir, all_results)
    print(f"resolution_sweep_summary.csv: {sweep_summary}", flush=True)

    if args.plots and sweep_summary.is_file():
        from plot_resolution_sweep import generate_resolution_sweep_plots

        paths = generate_resolution_sweep_plots(sweep_summary, plots_dir=run_dir)
        print(f"스윕 그래프 {len(paths)}개 저장", flush=True)


if __name__ == "__main__":
    main()
