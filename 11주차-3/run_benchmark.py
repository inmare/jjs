#!/usr/bin/env python3
"""
11주차-3 — Detection CNN 스윕 (YOLO + Torchvision).

기본 탐지기:
  - yolo:yolov8n.pt @ 1536, 640
  - torchvision: fasterrcnn / retinanet / ssd300 @ 1536

예:
  uv run python 11주차-3/run_benchmark.py --detect-only --max-samples 10
  uv run python 11주차-3/run_benchmark.py --detectors yolo:yolov8n.pt:640
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_W3 = Path(__file__).resolve().parent
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))

from week3.config import (  # noqa: E402
    DEFAULT_BENCHMARKS,
    DEFAULT_MAX_CROPS_VLM,
    DEFAULT_YOLO_IMGSZ_MAX_LONG,
    DEFAULT_YOLO_MODELS,
    build_run_config,
    format_summary_stamp,
    resolve_summary_run_paths,
)
from week3.datasets import load_benchmark_dataset  # noqa: E402
from week3.detectors.base import (  # noqa: E402
    DEFAULT_DETECTOR_SPECS,
    DetectorSpec,
    expand_legacy_yolo_args,
    load_detector_backend,
    specs_from_strings,
)
from week3.metrics import (  # noqa: E402
    RunAccumulator,
    append_sample_log,
    sample_to_log_record,
    summarize,
    write_summary_csv,
)
from week3.pipeline import QwenVLMRunner, run_sample  # noqa: E402


def parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.replace(",", " ").split():
        p = part.strip()
        if p:
            out.append(int(p))
    return out


def parse_str_list(s: str) -> list[str]:
    return [p.strip() for p in s.replace(",", " ").split() if p.strip()]


def resolve_detector_specs(args: argparse.Namespace) -> list[DetectorSpec]:
    if args.detectors.strip():
        return specs_from_strings(parse_str_list(args.detectors))
    # 레거시: --yolo-models × --yolo-imgsz-max-long
    yolo_models = parse_str_list(args.yolo_models) if args.yolo_models else []
    imgsz = parse_int_list(args.yolo_imgsz_max_long) if args.yolo_imgsz_max_long else []
    if yolo_models and imgsz:
        return expand_legacy_yolo_args(yolo_models, imgsz)
    return list(DEFAULT_DETECTOR_SPECS)


def main() -> int:
    p = argparse.ArgumentParser(description="11주차-3 Detection CNN 벤치마크")
    p.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help=f"쉼표 구분 (기본: {','.join(DEFAULT_BENCHMARKS)})",
    )
    p.add_argument(
        "--detectors",
        default="",
        help=(
            "탐지기 목록. 형식 backend:model:imgsz — "
            "예: yolo:yolov8n.pt:640, torchvision:ssd300_vgg16:1536. "
            "비우면 기본 5종(YOLO 2 + TV 3)."
        ),
    )
    p.add_argument(
        "--yolo-models",
        default="",
        help=f"(레거시) YOLO만 지정 시 --yolo-imgsz-max-long 과 곱함. 기본 YOLO: {DEFAULT_YOLO_MODELS[0]}",
    )
    p.add_argument(
        "--yolo-imgsz-max-long",
        default="",
        help=f"(레거시) 기본: {','.join(str(x) for x in DEFAULT_YOLO_IMGSZ_MAX_LONG)}",
    )
    p.add_argument("--max-samples", type=int, default=0, help="0=전체")
    p.add_argument(
        "--max-crops",
        type=int,
        default=-1,
        help=(
            f"VLM 시 샘플당 crop 상한 (0=무제한). "
            f"기본: detect-only=0, VLM={DEFAULT_MAX_CROPS_VLM}"
        ),
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--detect-only",
        action="store_true",
        help="탐지·crop만 (Qwen 생략). --yolo-only 와 동일",
    )
    p.add_argument("--yolo-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda:0", help="탐지기 디바이스 cpu 또는 cuda:0")
    p.add_argument("--yolo-device", default="", help=argparse.SUPPRESS)
    p.add_argument(
        "--output",
        default="",
        help="결과 폴더 (기본: results/summary_YYYYMMDD_HHMMSS/)",
    )
    p.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="종료 후 summary CSV 로 PNG 그래프 생성 (기본 켜짐)",
    )
    args = p.parse_args()

    detect_only = args.detect_only or args.yolo_only
    device = (args.yolo_device or args.device).strip()
    if args.max_crops < 0:
        max_crops = 0 if detect_only else DEFAULT_MAX_CROPS_VLM
    else:
        max_crops = args.max_crops

    benchmarks = parse_str_list(args.benchmarks)
    detector_specs = resolve_detector_specs(args)

    print(f"벤치마크 ({len(benchmarks)}개): {', '.join(benchmarks)}", flush=True)
    if not detect_only:
        print(f"VLM crop 상한: {max_crops if max_crops > 0 else '무제한'}", flush=True)

    if not detect_only:
        import torch

        if not torch.cuda.is_available():
            print(
                "[경고] CUDA 없음 — VLM 벤치는 느립니다. --detect-only 권장.",
                flush=True,
            )

    print("Detectors:", flush=True)
    for s in detector_specs:
        print(f"  - {s.method_name()}", flush=True)

    run_dir, out_path, detail_dir, samples_dir = resolve_summary_run_paths(args.output)
    print(f"결과 폴더: {run_dir}", flush=True)

    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            build_run_config(
                benchmarks=benchmarks,
                detector_methods=[s.method_name() for s in detector_specs],
                detect_only=detect_only,
                max_crops=max_crops,
                max_samples=args.max_samples,
                start=args.start,
                device=device,
            )
            | {"run_stamp": format_summary_stamp()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    backend_cache: dict = {}
    vlm: QwenVLMRunner | None = None
    if not detect_only:
        vlm = QwenVLMRunner()
        vlm.load()

    summary_rows: list[dict] = []

    try:
        for bench in benchmarks:
            ds = load_benchmark_dataset(bench)
            n_total = len(ds)
            end = n_total if args.max_samples <= 0 else min(
                n_total, args.start + args.max_samples
            )

            for spec in detector_specs:
                backend = load_detector_backend(spec, device, backend_cache)
                mname = spec.method_name()
                acc = RunAccumulator(benchmark=bench, method=mname)
                safe_name = mname.replace("/", "_")
                sample_log_path = samples_dir / f"{bench}_{safe_name}.jsonl"
                if sample_log_path.is_file():
                    sample_log_path.unlink()
                print(f"\n=== {bench} | {mname} | rows {args.start}:{end} ===", flush=True)

                for idx in range(args.start, end):
                    if idx % 10 == 0:
                        print(f"  sample {idx}/{end}", flush=True)
                    sm = run_sample(
                        ds[idx],
                        spec=spec,
                        backend=backend,
                        device=device,
                        vlm=vlm,
                        detect_only=detect_only,
                        max_crops=max_crops,
                        sample_index=idx,
                    )
                    if sm is not None:
                        acc.add(sm)
                        append_sample_log(
                            sample_log_path,
                            sample_to_log_record(sm, benchmark=bench, method=mname),
                        )

                row = summarize(acc)
                summary_rows.append(row)
                print(
                    f"  done accuracy={row.get('accuracy_percent', 0):.2f}% "
                    f"avg_preprocess={row.get('avg_preprocess_time_sec', 0):.3f}s "
                    f"avg_objects={row.get('avg_num_objects', 0):.1f}",
                    flush=True,
                )
                detail_path = detail_dir / f"{bench}_{safe_name}.json"
                detail_path.write_text(
                    json.dumps(row, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
    finally:
        if vlm is not None:
            vlm.unload()

    write_summary_csv(summary_rows, out_path)
    print(f"\nWrote summary: {out_path}", flush=True)

    if args.plots and summary_rows:
        try:
            from week3.plots import generate_plots_from_summary_csv

            pngs = generate_plots_from_summary_csv(out_path, plots_dir=run_dir)
            print(f"Plots ({len(pngs)} files) → {run_dir}", flush=True)
            md = run_dir / "summary_table.md"
            if md.is_file():
                print(f"  표(한글): {md}", flush=True)
            for pth in pngs[:6]:
                print(f"  {pth}", flush=True)
            if len(pngs) > 6:
                print(f"  … 외 {len(pngs) - 6}개", flush=True)
        except Exception as exc:
            print(f"[경고] 그래프 생성 실패: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
