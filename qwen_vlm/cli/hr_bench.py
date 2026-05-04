"""
HR-Bench - 네 가지 입력 전략 비교 (Qwen3-VL llama-server OpenAI 호환).

  uv run python -m qwen_vlm.cli.hr_bench --help
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

from openai import OpenAI

from qwen_vlm.hr_bench.io import (
    HR_BENCH_ID,
    HR_BENCH_LLAMA_OPENAI_BASE,
    HR_BENCH_MODEL_QWEN,
    HR_BENCH_MODEL_SMOL,
    HR_BENCH_YOLO_WEIGHTS,
    SPLITS,
    load_hr_split,
    select_indices,
)
from qwen_vlm.hr_bench.metrics import aggregate_strategy_rows
from qwen_vlm.hr_bench.report import write_hr_compare_charts_png, write_hr_compare_html
from qwen_vlm.hr_bench.single_llama import SingleLlamaSwapController
from qwen_vlm.hr_bench.strategies import (
    HRBenchStrategyConfig,
    STRATEGIES_ALL,
    run_strategy_on_indices,
)
from qwen_vlm.pipeline.week import SMOL_GGUF, SMOL_MMPROJ
from qwen_vlm.main import (
    DEFAULT_GGUF,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_MMPROJ,
    ROOT,
    host_port_from_openai_base_url,
    start_llama_server,
    wait_for_server,
)
from qwen_vlm.utils.openai_compat import normalize_openai_base_url
from qwen_vlm.utils.stdio_utf8 import configure_stdio_utf8

_SMOL_STRATEGIES = frozenset({"yolo_smol_parallel", "yolo_smol_sequential"})


def _paired_smol_log_path(qwen_log: Path) -> Path:
    """``--log-file`` 과 같은 디렉터리에 Smol 전용 로그 경로."""
    return qwen_log.parent / f"{qwen_log.stem}-smol{qwen_log.suffix}"


def _parse_strategies_arg(s: str) -> list[str]:
    if (s or "").strip().lower() in ("all", "*"):
        return list(STRATEGIES_ALL)
    out: list[str] = []
    for part in s.replace(",", " ").split():
        p = part.strip()
        if not p:
            continue
        if p not in STRATEGIES_ALL:
            print(
                f"[오류] 알 수 없는 전략 {p!r}. 허용: {', '.join(STRATEGIES_ALL)}, all",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if p not in out:
            out.append(p)
    if not out:
        print("[오류] --strategies 에 최소 한 개 전략 또는 all", file=sys.stderr)
        raise SystemExit(1)
    return out


def main() -> int:
    configure_stdio_utf8()
    p = argparse.ArgumentParser(
        description="HR-Bench 객관식 - 다중 입력 전략 비교"
    )
    p.add_argument("--split", default="hrbench_4k", choices=SPLITS)
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument(
        "--sample-mode",
        choices=("sequential", "random"),
        default="sequential",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--strategies",
        default="all",
        help=f"all 또는 공백/쉼표 구분: {' '.join(STRATEGIES_ALL)}",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--context-max-side",
        type=int,
        default=960,
        help="저해상 전망(긴 변 캡). --context-by-long-edge 일 때만 전망 크기로 쓰임",
    )
    p.add_argument(
        "--context-resize-scale",
        type=float,
        default=0.5,
        help="전망 가로·세로 각 이 비율 축소(YOLO+VLM 외부 스크립트와 동일). --context-by-long-edge 이면 무시",
    )
    p.add_argument(
        "--context-by-long-edge",
        action="store_true",
        help="전망을 비율 축소 대신 --context-max-side 만 사용",
    )
    p.add_argument(
        "--crop-max-side",
        type=int,
        default=0,
        help="크롭 VLM 입력 긴 변 상한. 0=원본 크롭 그대로",
    )
    p.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="YOLO→VLM 크롭 개수 상한. 0=예산·필터 후 전부",
    )
    p.add_argument(
        "--yolo-weights",
        type=str,
        default="",
        help=f"Ultralytics 가중치 경로. 비우면 {HR_BENCH_YOLO_WEIGHTS!r}",
    )
    p.add_argument(
        "--yolo-surveillance-classes-only",
        action="store_true",
        help="감시용 COCO 부분 클래스만. 기본은 전 클래스",
    )
    p.add_argument(
        "--yolo-max-bbox-area-num",
        type=int,
        default=1,
        metavar="N",
        help="박스 면적 ≤ 원본×N/M (분자)",
    )
    p.add_argument(
        "--yolo-max-bbox-area-den",
        type=int,
        default=4,
        metavar="M",
        help="박스 면적 분모(기본 1/4)",
    )
    p.add_argument("--min-crop-short-side", type=int, default=0)
    p.add_argument("--min-crop-area", type=int, default=0)
    p.add_argument(
        "--qwen-image-max-long-side",
        type=int,
        default=0,
        help="qwen_only 전용: 0이면 다른 전략과 동일 전망(비율 또는 긴 변 캡). >0이면 이 긴 변 캡만",
    )
    p.add_argument(
        "--qwen-only-use-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "qwen_only 전략에서 원본 이미지를 그대로 전송(대조군). "
            "--no-qwen-only-use-original 이면 context-resize-scale/context-max-side 적용"
        ),
    )
    p.add_argument(
        "--yolo-overview-max-side",
        type=int,
        default=960,
        help=(
            "YOLO 전략 전용 전망 긴 변 상한(픽셀). "
            "0이면 context-resize-scale/context-max-side 와 동일. "
            "> 0이면 원본에서 긴 변을 이 크기 이하로 축소한 전망(#0)·Smol 입력 (기본 960)"
        ),
    )
    p.add_argument(
        "--llama-server",
        default=str(DEFAULT_LLAMA_SERVER),
    )
    p.add_argument("--gguf", default=str(DEFAULT_GGUF))
    p.add_argument("--mmproj", default=str(DEFAULT_MMPROJ))
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    p.add_argument("--ctx-size", type=int, default=8192)
    p.add_argument("--server-timeout", type=float, default=180.0)
    p.add_argument(
        "--log-file",
        type=Path,
        default=ROOT / "vendor" / "llama-server-hr-bench.log",
    )
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--smol-max-tokens", type=int, default=256)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--html-out", type=Path, default=None)
    p.add_argument(
        "--png-out",
        type=Path,
        default=None,
        help="전략 비교 막대 그래프 PNG (matplotlib)",
    )
    p.add_argument(
        "--no-sample-tables",
        action="store_true",
        help="HTML에서 샘플별 긴 표 생략(요약만)",
    )
    p.add_argument(
        "--omit-smol-if-unavailable",
        action="store_true",
        help=(
            "Smol GGUF/mmproj 파일이 없으면 yolo_smol_* 전략만 빼고 나머지만 실행"
        ),
    )
    p.add_argument(
        "--smol-gguf",
        type=Path,
        default=SMOL_GGUF,
        help="단일 포트 스왑 시 Smol GGUF 경로",
    )
    p.add_argument(
        "--smol-mmproj",
        type=Path,
        default=SMOL_MMPROJ,
        help="단일 포트 스왑 시 Smol mmproj 경로",
    )
    p.add_argument(
        "--disable-yolo-vlm-budget",
        action="store_true",
        help="YOLO→VLM 박스 픽셀 예산·필터 끄기(기본: 켜짐)",
    )
    args = p.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "sk-local")
    if args.max_samples < 0:
        print("[오류] --max-samples 는 0 이상", file=sys.stderr)
        return 1

    strategies = _parse_strategies_arg(args.strategies)
    need_smol = any(s in _SMOL_STRATEGIES for s in strategies)

    if need_smol and args.omit_smol_if_unavailable:
        smol_ok = Path(args.smol_gguf).is_file() and Path(args.smol_mmproj).is_file()
        if not smol_ok:
            strategies = [s for s in strategies if s not in _SMOL_STRATEGIES]
            need_smol = False
            if not strategies:
                print(
                    "[오류] Smol 전략만 선택했는데 Smol GGUF/mmproj 가 없습니다.",
                    file=sys.stderr,
                )
                return 1
            print(
                f"[경고] Smol 가중치 없음 → Smol 전략 생략, 실행: {strategies}",
                flush=True,
            )

    base = normalize_openai_base_url(HR_BENCH_LLAMA_OPENAI_BASE)
    proc: subprocess.Popen | None = None
    log_f: TextIO | None = None
    own_server = False
    swap_ctrl: SingleLlamaSwapController | None = None
    client_qwen: OpenAI
    client_small: OpenAI | None = None

    if need_smol:
        try:
            wait_for_server(base, timeout_s=2.0)
            print(
                f"[오류] {base} 에 이미 프로세스가 응답합니다. "
                "Smol↔Qwen 스왑은 **비어 있는 포트**가 필요합니다(해당 포트의 llama-server 종료).",
                file=sys.stderr,
            )
            return 1
        except RuntimeError:
            pass
        for label, pth in (
            ("--smol-gguf", args.smol_gguf),
            ("--smol-mmproj", args.smol_mmproj),
            ("--gguf", args.gguf),
            ("--mmproj", args.mmproj),
        ):
            if not Path(pth).is_file():
                print(f"[오류] {label} 파일 없음: {pth}", file=sys.stderr)
                return 1
        try:
            host, port = host_port_from_openai_base_url(base)
        except ValueError as e:
            print(f"[오류] {e}", file=sys.stderr)
            return 1
        print(
            f"[HR-Bench] 단일 GPU: {host}:{port} 에서 Smol GGUF → Qwen GGUF 순차 스왑",
            flush=True,
        )
        swap_ctrl = SingleLlamaSwapController(
            llama_server=args.llama_server,
            host=host,
            port=port,
            qwen_gguf=args.gguf,
            qwen_mmproj=args.mmproj,
            qwen_model=HR_BENCH_MODEL_QWEN,
            smol_gguf=args.smol_gguf,
            smol_mmproj=args.smol_mmproj,
            smol_model=HR_BENCH_MODEL_SMOL,
            ngl=args.ngl,
            flash_attn=args.flash_attn,
            ctx_size=args.ctx_size,
            qwen_log=args.log_file,
            smol_log=_paired_smol_log_path(args.log_file),
            api_key=api_key,
            server_timeout=args.server_timeout,
        )
        client_qwen = OpenAI(base_url=swap_ctrl.base, api_key=api_key)
        client_small = client_qwen
    else:
        try:
            wait_for_server(base, timeout_s=2.0)
            print(f"[HR-Bench] 기존 llama-server 사용: {base}", flush=True)
        except RuntimeError:
            try:
                host, port = host_port_from_openai_base_url(base)
            except ValueError as e:
                print(f"[오류] {e}", file=sys.stderr)
                return 1
            print(
                f"[HR-Bench] llama-server 기동: {host}:{port} (stderr → {args.log_file})",
                flush=True,
            )
            try:
                proc, base, log_f = start_llama_server(
                    llama_server=args.llama_server,
                    gguf=args.gguf,
                    mmproj=args.mmproj,
                    host=host,
                    port=port,
                    ngl=args.ngl,
                    flash_attn=args.flash_attn,
                    model=HR_BENCH_MODEL_QWEN,
                    ctx_size=args.ctx_size,
                    log_file=args.log_file,
                )
            except (FileNotFoundError, OSError) as e:
                print(str(e), file=sys.stderr)
                return 1
            own_server = True
            try:
                wait_for_server(base, args.server_timeout, child=proc)
            except RuntimeError as e:
                if log_f is not None:
                    log_f.close()
                if proc is not None:
                    proc.terminate()
                print(f"[오류] 서버 대기 실패: {e}", file=sys.stderr)
                return 1
        client_qwen = OpenAI(base_url=base, api_key=api_key)

    yolo_w = (args.yolo_weights or "").strip() or HR_BENCH_YOLO_WEIGHTS
    ctx_scale: float | None = (
        None if args.context_by_long_edge else args.context_resize_scale
    )
    cfg = HRBenchStrategyConfig(
        context_max_side=args.context_max_side,
        context_resize_scale=ctx_scale,
        crop_max_side=args.crop_max_side,
        max_crops=args.max_crops,
        min_crop_short_side=args.min_crop_short_side,
        min_crop_area=args.min_crop_area,
        yolo_vlm_budget=not args.disable_yolo_vlm_budget,
        yolo_model_path=yolo_w,
        yolo_surveillance_classes_only=args.yolo_surveillance_classes_only,
        yolo_max_bbox_area_numerator=args.yolo_max_bbox_area_num,
        yolo_max_bbox_area_denominator=args.yolo_max_bbox_area_den,
        yolo_overview_max_side=args.yolo_overview_max_side,
        qwen_only_use_original=args.qwen_only_use_original,
        qwen_image_max_long_side=args.qwen_image_max_long_side,
        max_tokens=args.max_tokens,
        smol_max_tokens=args.smol_max_tokens,
    )

    def log_fn(msg: str) -> None:
        print(msg, flush=True)

    try:
        print(f"[HR-Bench] 로딩: {HR_BENCH_ID} / {args.split} …", flush=True)
        ds = load_hr_split(args.split)
        n = len(ds)
        if n == 0:
            print("[오류] split 비어 있음", file=sys.stderr)
            return 1
        if args.sample_mode == "random" and args.start != 0:
            print(
                "[힌트] random 모드에서는 --start 가 무시됩니다.",
                file=sys.stderr,
            )
        indices = select_indices(
            n=n,
            max_samples=args.max_samples,
            sample_mode=args.sample_mode,
            start=args.start,
            seed=args.seed,
        )
        if not indices:
            print("[오류] 평가할 샘플이 없음", file=sys.stderr)
            return 1
        n_strat = len(strategies)
        print(
            f"[HR-Bench] N={n} · 이번 인덱스 {len(indices)}개 · 전략 {n_strat}개 {strategies}",
            flush=True,
        )

        strategy_summaries: list[dict[str, Any]] = []
        per_strategy_rows: dict[str, list[dict[str, Any]]] = {}

        for si, strat in enumerate(strategies):
            print(
                f"\n======== 전략 {si + 1}/{n_strat}: {strat} ========",
                flush=True,
            )
            if swap_ctrl is not None and strat not in _SMOL_STRATEGIES:
                swap_ctrl.ensure_qwen()
            smol_hooks = (
                swap_ctrl.hooks()
                if swap_ctrl is not None and strat in _SMOL_STRATEGIES
                else None
            )
            rows = run_strategy_on_indices(
                strategy=strat,
                ds=ds,
                indices=indices,
                client_qwen=client_qwen,
                model_qwen=HR_BENCH_MODEL_QWEN,
                client_small=client_small,
                model_small=HR_BENCH_MODEL_SMOL if need_smol else None,
                cfg=cfg,
                log=log_fn,
                smol_single_server_hooks=smol_hooks,
            )
            per_strategy_rows[strat] = rows
            strategy_summaries.append(aggregate_strategy_rows(strat, rows))

        summary: dict[str, Any] = {
            "hr_bench": HR_BENCH_ID,
            "split": args.split,
            "sample_mode": args.sample_mode,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "start": args.start if args.sample_mode == "sequential" else None,
            "dataset_size": n,
            "selected_dataset_indices": indices,
            "strategies": strategy_summaries,
            "model": HR_BENCH_MODEL_QWEN,
            "model_small": HR_BENCH_MODEL_SMOL if need_smol else None,
            "base_url": base,
            "single_llama_swap": swap_ctrl is not None,
            "spawned_llama": own_server,
            "llama_server_log": str(args.log_file) if own_server else None,
            "smol_log_file": str(_paired_smol_log_path(args.log_file))
            if swap_ctrl is not None
            else None,
            "config": {
                "context_max_side": cfg.context_max_side,
                "context_resize_scale": cfg.context_resize_scale,
                "context_by_long_edge_only": args.context_by_long_edge,
                "crop_max_side": cfg.crop_max_side,
                "max_crops": cfg.max_crops,
                "min_crop_short_side": cfg.min_crop_short_side,
                "min_crop_area": cfg.min_crop_area,
                "qwen_image_max_long_side": cfg.qwen_image_max_long_side,
                "yolo_model_path": cfg.yolo_model_path,
                "yolo_surveillance_classes_only": cfg.yolo_surveillance_classes_only,
                "yolo_max_bbox_area_numerator": cfg.yolo_max_bbox_area_numerator,
                "yolo_max_bbox_area_denominator": cfg.yolo_max_bbox_area_denominator,
                "yolo_vlm_budget": cfg.yolo_vlm_budget,
                "yolo_overview_max_side": cfg.yolo_overview_max_side,
                "qwen_only_use_original": cfg.qwen_only_use_original,
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **summary,
                "samples_by_strategy": per_strategy_rows,
            }
            args.json_out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"저장: {args.json_out}", flush=True)
        if args.html_out:
            write_hr_compare_html(
                args.html_out,
                summary=summary,
                strategy_summaries=strategy_summaries,
                per_strategy_rows=per_strategy_rows,
                include_sample_tables=not args.no_sample_tables,
            )
            print(f"저장: {args.html_out}", flush=True)
        if args.png_out:
            try:
                write_hr_compare_charts_png(args.png_out, strategy_summaries)
                print(f"저장: {args.png_out}", flush=True)
            except RuntimeError as e:
                print(f"[경고] PNG 생략: {e}", file=sys.stderr)
        return 0
    finally:
        if swap_ctrl is not None:
            swap_ctrl.stop()
        if own_server and proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=25.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            try:
                if log_f is not None:
                    log_f.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
