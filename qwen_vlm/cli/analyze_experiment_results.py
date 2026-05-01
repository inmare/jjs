"""
``docs/experiment_results_last.json`` 집계를 재계산해 ``latency_compare`` 와 대조.

  uv run python -m qwen_vlm.cli.analyze_experiment_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qwen_vlm.main import ROOT
from qwen_vlm.pipeline.week import (
    _bench_aggregate,
    _latency_summary,
    _two_stage_aggregate,
)
from qwen_vlm.utils.stdio_utf8 import configure_stdio_utf8

_DEFAULT_JSON = ROOT / "docs" / "experiment_results_last.json"


def _fclose(x: object, y: object, label: str, tol: float = 1e-3) -> bool:
    a = x is y if x is None or y is None else abs(float(x) - float(y)) < tol
    if not a:
        print(f"  [불일치] {label}: 파일={x!r}  재계산={y!r}", flush=True)
    return a


def main() -> None:
    p = argparse.ArgumentParser(
        description="실험 JSON의 latency_compare·bench 집계를 재계산해 파일 값과 대조합니다."
    )
    p.add_argument(
        "--json",
        type=Path,
        default=_DEFAULT_JSON,
        help="실험 결과 JSON 경로 (기본: docs/experiment_results_last.json)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="프레임 수·필드 요약을 추가 출력",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="집계 불일치 시 종료 코드 1",
    )
    args = p.parse_args()
    path: Path = args.json
    if not path.is_file():
        print(f"파일 없음: {path}", file=sys.stderr)
        sys.exit(2)
    with path.open(encoding="utf-8") as f:
        payload: dict = json.load(f)

    bench = payload.get("bench") or []
    ysp = payload.get("yolo_smol_parallel_qwen") or []
    tss = payload.get("two_stage_skip_low") or []
    file_lc = payload.get("latency_compare") or {}

    b = _bench_aggregate(bench)
    recomputed = _latency_summary(
        bench_rows=bench, two_stage_rows=tss, parallel_rows=ysp or None
    )
    t_agg = _two_stage_aggregate(tss)

    if args.verbose:
        print("--- 메타 ---", flush=True)
        print(
            f"  generated_at: {payload.get('generated_at')!r}  frame_count: {payload.get('frame_count')}",
            flush=True,
        )
        print(
            f"  frames: bench={len(bench)}  yolo_smol_parallel_qwen={len(ysp)}  two_stage_skip_low={len(tss)}",
            flush=True,
        )
        if b:
            print("--- bench 재집계 ---", flush=True)
            for k, v in b.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}", flush=True)
                else:
                    print(f"  {k}: {v}", flush=True)

    print("--- latency_compare vs 재계산 ---", flush=True)
    ok = True
    fqo = file_lc.get("bench_avg_qwen_only_s")
    fyq = file_lc.get("bench_avg_yolo_qwen_total_s")
    rqo = recomputed.get("bench_avg_qwen_only_s")
    ryq = recomputed.get("bench_avg_yolo_qwen_total_s")
    ok &= _fclose(fqo, rqo, "bench_avg_qwen_only_s", tol=1e-2)
    ok &= _fclose(fyq, ryq, "bench_avg_yolo_qwen_total_s", tol=1e-2)

    fp = file_lc.get("yolo_smol_parallel")
    rp = recomputed.get("yolo_smol_parallel")
    if fp or rp:
        if fp and rp:
            ok &= _fclose(
                fp.get("avg_wall_parallel_s"),
                rp.get("avg_wall_parallel_s"),
                "yolo_smol avg_wall_parallel_s",
            )
            ok &= _fclose(
                fp.get("avg_total_s"),
                rp.get("avg_total_s"),
                "yolo_smol avg_total_s",
            )
        else:
            print("  [불일치] yolo_smol_parallel 블록 한쪽만 있음", flush=True)
            ok = False

    fts = file_lc.get("two_stage")
    rts = recomputed.get("two_stage")
    if fts and rts is None and len(tss) == 0:
        pass
    elif fts and rts:
        ok &= _fclose(fts.get("avg_stage1_s"), rts.get("avg_stage1_s"), "two_stage avg_stage1_s")
        ok &= _fclose(
            fts.get("avg_qwen_s_when_called"),
            rts.get("avg_qwen_s_when_called"),
            "two_stage avg_qwen_s_when_called",
        )
        ok &= _fclose(fts.get("avg_total_s"), rts.get("avg_total_s"), "two_stage avg_total_s")
    elif tss and not (fts and rts):
        print("  [불일치] two_stage 요약", flush=True)
        ok = False

    t_file = {
        "frames": t_agg.get("frames") if t_agg else None,
        "qwen_skipped": t_agg.get("qwen_skipped_low_risk") if t_agg else None,
        "qwen_ran": t_agg.get("qwen_ran") if t_agg else None,
    }
    if args.verbose and t_file["frames"] is not None:
        print("--- two_stage_skip_low (재집계) ---", flush=True)
        for k, v in t_file.items():
            print(f"  {k}: {v}", flush=True)

    if ok:
        print("  → 집계 일치(허용 오차 내).", flush=True)
    else:
        print(
            "  → 불일치 항목이 있음. JSON 을 다시 생성하거나( run_week_experiments.py ) 집계 로직을 확인하세요.",
            flush=True,
        )
    if args.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    configure_stdio_utf8()
    main()
