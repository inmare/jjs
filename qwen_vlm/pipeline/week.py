"""
Qwen3-VL + SmolVLM-256M 서버를 로컬에서 띄운 뒤 벤치·2단계·(**YOLO∥Smol**)+Qwen 을 돌리고 JSON 을 저장합니다.

단계 요약(VRAM 절약을 위해 llama-server 를 번갈아 켠다):

1. **Phase 1:** Qwen 서버만 기동 → ``bench`` (Qwen-only vs YOLO+Qwen).
2. **Phase 2:** Smol 서버만 기동 → 프레임별 Smol 1단계 응답을 ``precached_stage1`` 에 캐시.
3. **Phase 3:** Qwen + Smol **동시** 기동 →

   - ``run_yolo_smol_parallel_qwen``: 스레드 풀로 YOLO 와 Smol 을 **병렬** 수행 후 Qwen.
   - ``run_two_stage``: 캐시된 Smol 로 two-stage(``--skip-qwen-if-low``).

Phase 3 은 GPU 메모리를 많이 쓴다(두 VLM + YOLO). 8GB 환경에서는 OOM 시 ``--ngl`` 축소 등이 필요할 수 있다.

  uv sync --group dev
  uv run python scripts/download_smolvlm_256m.py
  uv run python run_week_experiments.py
  # 또는: uv run python -m qwen_vlm.pipeline.week

ShanghaiTech 프레임은 `data/datasets/shanghaitech_frames/` 등에 두고 `--frames-dir` 로 지정합니다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

from qwen_vlm.main import ROOT, chat_vlm_multi, pil_to_data_url, wait_for_server
from qwen_vlm.pipeline.experiment import (
    DEFAULT_PROMPT_KO,
    STAGE1_PROMPT_EN,
    bench,
    bench_with_frame_gating,
    resolve_input_frames,
    resize_max_side,
    run_two_stage,
    run_yolo_smol_parallel_qwen,
)
from qwen_vlm.reporting.experiment_metrics import (
    resource_snapshot,
    usage_to_dict,
    write_experiment_html_report,
    write_single_image_experiment_html,
)
from qwen_vlm.utils.stdio_utf8 import configure_stdio_utf8

DEFAULT_LLAMA = ROOT / "vendor" / "llama-cpp-win-cuda" / "llama-server.exe"
QWEN_GGUF = ROOT / "vendor" / "qwen3-vl-4b-q8-gguf" / "Qwen3VL-4B-Instruct-Q8_0.gguf"
QWEN_MMPROJ = ROOT / "vendor" / "qwen3-vl-4b-q8-gguf" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
SMOL_GGUF = ROOT / "vendor" / "smolvlm-256m-q8" / "SmolVLM-256M-Instruct-Q8_0.gguf"
SMOL_MMPROJ = ROOT / "vendor" / "smolvlm-256m-q8" / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

QWEN_PORT = 8765
SMOL_PORT = 8766
QWEN_MODEL = "qwen3-vl-4b-q8"
SMOL_MODEL = "smolvlm-256m-q8"


def _popen_server(cmd: list[str], log_path: Path) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=log_f,
        creationflags=creationflags,
    )
    return proc, log_f


def _bench_aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    pt = lambda key: [r[key]["prompt_tokens"] for r in rows if r[key].get("prompt_tokens") is not None]
    q_pt = pt("qwen_only")
    y_pt = pt("yolo_qwen")
    return {
        "frames": n,
        "qwen_only_avg_s": sum(r["qwen_only"]["seconds"] for r in rows) / n,
        "qwen_only_avg_prompt_tokens": sum(q_pt) / len(q_pt) if q_pt else None,
        "qwen_only_avg_approx_vision": sum(
            r["approx_vision_tokens"]["qwen_only"] for r in rows
        )
        / n,
        "yolo_qwen_avg_total_s": sum(r["yolo_qwen"]["total_seconds"] for r in rows) / n,
        "yolo_qwen_avg_prompt_tokens": sum(y_pt) / len(y_pt) if y_pt else None,
        "yolo_qwen_avg_approx_vision": sum(
            r["approx_vision_tokens"]["yolo_qwen_images"] for r in rows
        )
        / n,
    }


def _two_stage_aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    skipped = sum(1 for r in rows if not r.get("qwen_called", True))
    return {
        "frames": n,
        "qwen_skipped_low_risk": skipped,
        "qwen_ran": n - skipped,
    }


def _latency_summary(
    bench_rows: list[dict],
    two_stage_rows: list[dict],
    parallel_rows: list[dict] | None = None,
) -> dict:
    """bench 평균, (YOLO∥Smol)+Qwen 평균, two-stage(Smol 1단계 + Qwen 2단계) 평균을 한 dict 에 담는다."""
    ba = _bench_aggregate(bench_rows)
    n = len(two_stage_rows)
    out: dict = {
        "bench_avg_qwen_only_s": ba.get("qwen_only_avg_s"),
        "bench_avg_yolo_qwen_total_s": ba.get("yolo_qwen_avg_total_s"),
    }
    if parallel_rows:
        np = len(parallel_rows)
        if np:
            wavg = sum(
                r["yolo_smol_parallel"]["wall_parallel_seconds"] for r in parallel_rows
            ) / np
            tavg = sum(r["total_seconds"] for r in parallel_rows) / np
            out["yolo_smol_parallel"] = {
                "avg_wall_parallel_s": round(wavg, 4),
                "avg_total_s": round(tavg, 4),
                "frames": np,
                "note": (
                    "wall_parallel: ThreadPool YOLO∥Smol 구간, total: wall+qwen(또는 low 스킵시 wall만)"
                ),
            }
    if n == 0:
        out["two_stage"] = None
        return out
    avg_s1 = sum(r["stage1_seconds"] for r in two_stage_rows) / n
    called = [r for r in two_stage_rows if r.get("qwen_called")]
    avg_q = (
        sum(r["qwen_seconds"] for r in called) / len(called) if called else None
    )
    avg_tot = sum(r["total_seconds"] for r in two_stage_rows) / n
    out["two_stage"] = {
        "avg_stage1_s": round(avg_s1, 4),
        "avg_qwen_s_when_called": round(avg_q, 4) if avg_q is not None else None,
        "avg_total_s": round(avg_tot, 4),
        "frames": n,
        "qwen_calls": len(called),
        "note": "total = Phase2에서 측정한 Smol stage1_seconds + Phase3 qwen_seconds(또는 low면 stage1만).",
    }
    return out


def _write_auto_results_md(root: Path, payload: dict) -> None:
    md_path = root / "docs" / "week-demo-pipeline.md"
    if not md_path.is_file():
        return
    try:
        import torch

        gpu = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU (CUDA 없음)"
        )
    except Exception:
        gpu = "(확인 불가)"
    b = _bench_aggregate(payload.get("bench") or [])
    t = _two_stage_aggregate(payload.get("two_stage_skip_low") or [])
    lines = [
        "",
        f"**자동 기록** (`run_week_experiments.py`): {payload.get('generated_at', '')}",
        "",
        f"- **프레임 디렉터리**: `{payload.get('frames_dir', '')}` ({payload.get('frame_count', 0)}장)",
        "- **경량 VLM**: [ggml-org/SmolVLM-256M-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF) — "
        "`SmolVLM-256M-Instruct-Q8_0.gguf` + `mmproj-SmolVLM-256M-Instruct-Q8_0.gguf`",
        f"- **Qwen**: 로컬 `Qwen3VL-4B-Instruct-Q8_0` / 별칭 `{payload.get('qwen_model', '')}`",
        f"- **Ultralytics/torch 백엔드**: {gpu} — VLM·Smol·Qwen은 `llama-server` 별도 프로세스(VRAM/속도는 `resource_sampling_note`·행별 스냅샷 참고)",
        "",
        "### 집계 (bench)",
        "",
        "| 지표 | Qwen-only | YOLO+Qwen |",
        "|------|-----------|-----------|",
    ]
    if b:
        lines.append(
            f"| 평균 VLM 시간(s) | {b['qwen_only_avg_s']:.3f} | {b['yolo_qwen_avg_total_s']:.3f} (YOLO+추론) |"
        )
        qpt = b["qwen_only_avg_prompt_tokens"]
        ypt = b["yolo_qwen_avg_prompt_tokens"]
        qpt_s = f"{qpt:.1f}" if qpt is not None else "—"
        ypt_s = f"{ypt:.1f}" if ypt is not None else "—"
        lines.append(f"| 평균 prompt_tokens | {qpt_s} | {ypt_s} |")
        lines.append(
            f"| 평균 approx 비전 토큰(합) | {b['qwen_only_avg_approx_vision']:.1f} | "
            f"{b['yolo_qwen_avg_approx_vision']:.1f} |"
        )
    lines += ["", "### 지연 비교 (평균, 초)", ""]
    lc = payload.get("latency_compare") or {}
    ts = lc.get("two_stage") or {}
    if lc.get("bench_avg_qwen_only_s") is not None:
        qo = lc["bench_avg_qwen_only_s"]
        yq = lc.get("bench_avg_yolo_qwen_total_s")
        yq_s = f"{yq:.3f}" if yq is not None else "—"
        lines.append(
            f"- **Qwen-only (bench)**: 평균 **{qo:.3f}** s/프레임"
        )
        lines.append(
            f"- **YOLO+Qwen (bench)**: 평균 **{yq_s}** s/프레임 (YOLO + VLM)"
        )
    ppar = (payload.get("latency_compare") or {}).get("yolo_smol_parallel")
    if ppar:
        aw, at = ppar.get("avg_wall_parallel_s"), ppar.get("avg_total_s")
        aw_s = f"{aw:.3f}" if isinstance(aw, (int, float)) else str(aw)
        at_s = f"{at:.3f}" if isinstance(at, (int, float)) else str(at)
        lines.append(
            f"- **YOLO∥Smol→Qwen (병렬)**: wall 병렬 평균 **{aw_s}** s, "
            f"E2E 합산 평균 **{at_s}** s — ({ppar.get('note', '')})"
        )
    if ts:
        lines.append(
            f"- **Smol→Qwen (two-stage)**: Smol 평균 **{ts.get('avg_stage1_s', '—')}** s + "
            f"Qwen(호출 시) 평균 **{ts.get('avg_qwen_s_when_called', '—')}** s → "
            f"프레임당 평균 **{ts.get('avg_total_s', '—')}** s (end-to-end 합산)"
        )
        lines.append(f"  - ({ts.get('note', '')})")
    lines += ["", "### 집계 (two-stage, `--skip-qwen-if-low`)", ""]
    if t:
        lines.append(
            f"- Qwen 호출 생략: **{t['qwen_skipped_low_risk']}** / {t['frames']} 프레임 "
            f"(1단계 `risk=low`)"
        )
        lines.append(f"- Qwen 실행: **{t['qwen_ran']}** 회")
    lines += ["", "### 프레임별 bench (요약)", "", "| 프레임 | approx Q-only | approx YOLO+Q | p.tok Q | p.tok Y |", "|--------|---------------|---------------|---------|---------|"]
    for r in payload.get("bench") or []:
        qo = r.get("qwen_only") or {}
        yq = r.get("yolo_qwen") or {}
        av = r.get("approx_vision_tokens") or {}
        lines.append(
            f"| {r.get('frame', '')} | {av.get('qwen_only', '')} | {av.get('yolo_qwen_images', '')} | "
            f"{qo.get('prompt_tokens', '')} | {yq.get('prompt_tokens', '')} |"
        )
    lines += ["", "> ShanghaiTech 등 별도 프레임 디렉터리로 동일 스크립트를 다시 실행하면 이 블록을 덮어씁니다.", ""]
    block = "\n".join(lines)
    text = md_path.read_text(encoding="utf-8")
    start = "<!-- AUTO_RESULTS_START -->"
    end = "<!-- AUTO_RESULTS_END -->"
    if start not in text or end not in text:
        insert = f"\n{start}\n{block}{end}\n\n## 한계·다음 단계"
        if "## 한계·다음 단계" in text:
            text = text.replace("## 한계·다음 단계", insert, 1)
        else:
            text = text.rstrip() + "\n\n" + start + "\n" + block + end + "\n"
    else:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        text = pre + start + "\n" + block + end + post
    md_path.write_text(text, encoding="utf-8")
    print(f"[실험] 문서 갱신: {md_path}")


def _terminate(proc: subprocess.Popen | None, log_f: object | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_f is not None:
        try:
            log_f.close()
        except OSError:
            pass


def main() -> None:
    configure_stdio_utf8()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (OSError, ValueError, AttributeError):
        pass
    p = argparse.ArgumentParser(description="주간 실험: 서버 기동 + bench + two-stage")
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=ROOT / "data" / "datasets" / "demo" / "frames",
    )
    p.add_argument(
        "--single-image",
        type=Path,
        default=None,
        help=(
            "이미지 1장만 전체 Phase에 사용(고해상도 단일). "
            "지정 시 --frames-dir 로 목록을 만들지 않고 이 파일만 씀."
        ),
    )
    p.add_argument("--max-frames", type=int, default=6)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="스모크: 프레임 1장만 사용(빠른 검증). --max-frames 보다 우선합니다.",
    )
    p.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA)
    p.add_argument("--qwen-timeout", type=float, default=420.0)
    p.add_argument("--smol-timeout", type=float, default=240.0)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "docs" / "experiment_results_last.json",
    )
    p.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Chart.js 리포트 HTML (기본: --json-out 과 같은 stem, 확장자 .html)",
    )
    p.add_argument(
        "--no-frame-gate",
        action="store_true",
        help="Phase1 bench 를 프레임마다 풀 추론(게이트 비활성). 기본은 mse_then_yolo.",
    )
    p.add_argument(
        "--frame-gate",
        default="mse_then_yolo",
        help=(
            "Phase1 bench: mse | yolo | mse_then_yolo. "
            "mse=휘도로만 '전부 스킵' 판정 / yolo=매 프레임 YOLO 후 서명(또는 크롭게이트)로 VLM만 스킵, MSE 안 씀 / "
            "mse_then_yolo=둘 다."
        ),
    )
    p.add_argument(
        "--mse-threshold",
        type=float,
        default=2.5,
        help="휘도 유사 임계(작을수록 '같은 장면'으로 봄). mse·mse_then_yolo의 MSE 전체스킵에만 사용. yolo 전용이면 의미 없음",
    )
    p.add_argument(
        "--min-crop-short-side",
        type=int,
        default=0,
        metavar="PX",
        help=(
            "탐지 상자(원본) **짧은 쪽** 길이(픽셀)가 이 값 미만이면 VLM에 안 보냄. "
            "0=끔. 너무 작은 박스(토큰 낭비·노이즈) 제거"
        ),
    )
    p.add_argument(
        "--min-crop-area",
        type=int,
        default=0,
        metavar="PX2",
        help=(
            "탐지 상자 **면적**(원본 가로×세로, 픽셀²)이 이 값 미만이면 제외. 0=끔"
        ),
    )
    p.add_argument(
        "--use-crop-layout-gate",
        action="store_true",
        help=(
            "Phase1: 클래스 대신 '크롭 개수+박스 중심 위치'로 이전 프레임과 비교해 VLM만 생략"
        ),
    )
    p.add_argument(
        "--crop-gate-max-shift",
        type=float,
        default=0.02,
        metavar="RATIO",
        help=(
            "크롭 위치 게이트 켰을 때: 이전과 박스 **중심**이 화면 대비 최대 **몇 %**까지 "
            "움직여도 '같다'로 볼지(0.02≈2%). **작을수록** VLM을 더 자주 돌림"
        ),
    )
    args = p.parse_args()
    if args.smoke:
        args.max_frames = 1
    if args.single_image is not None:
        args.max_frames = 1

    for label, path in (
        ("llama-server", args.llama_server),
        ("Qwen GGUF", QWEN_GGUF),
        ("Qwen mmproj", QWEN_MMPROJ),
        ("SmolVLM GGUF", SMOL_GGUF),
        ("SmolVLM mmproj", SMOL_MMPROJ),
    ):
        if not path.is_file():
            print(f"[오류] {label} 없음: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        frames = resolve_input_frames(
            args.frames_dir, args.max_frames, args.single_image
        )
    except FileNotFoundError as e:
        print(f"[오류] {e}", file=sys.stderr)
        sys.exit(1)
    if not frames:
        print(f"[오류] 프레임 없음: {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    cmd_qwen = [
        str(args.llama_server),
        "-m",
        str(QWEN_GGUF),
        "--mmproj",
        str(QWEN_MMPROJ),
        "--no-mmproj-offload",
        "--host",
        "127.0.0.1",
        "--port",
        str(QWEN_PORT),
        "-np",
        "1",
        "-ngl",
        str(args.ngl),
        "-fa",
        "on",
        "-a",
        QWEN_MODEL,
        "--ctx-size",
        "8192",
    ]
    cmd_smol = [
        str(args.llama_server),
        "-m",
        str(SMOL_GGUF),
        "--mmproj",
        str(SMOL_MMPROJ),
        # SmolVLM 내장 Jinja는 OpenAI image_url과 맞지 않아 토큰화 실패할 수 있음 (#21634).
        "--no-jinja",
        "--host",
        "127.0.0.1",
        "--port",
        str(SMOL_PORT),
        "-np",
        "1",
        "-ngl",
        str(args.ngl),
        "-fa",
        "auto",
        "-a",
        SMOL_MODEL,
        "--ctx-size",
        "4096",
    ]

    # Phase1(Qwen) / Phase2(Smol) / Phase3(Qwen+Smol 동시) 마다 별도 로그로 분리해 스트림 뒤섞임을 줄인다.
    log_q = ROOT / "vendor" / "llama-server-qwen-experiment.log"
    log_q3 = ROOT / "vendor" / "llama-server-qwen-experiment-phase3.log"
    log_s = ROOT / "vendor" / "llama-server-smolvlm-experiment.log"
    log_s3 = ROOT / "vendor" / "llama-server-smolvlm-experiment-phase3.log"
    api_key = os.environ.get("OPENAI_API_KEY", "sk-local")

    # VRAM 8GB 대: Qwen+mmproj 동시 GPU 적재 시 OOM → 서버를 단계별로만 기동.
    fgate = "off" if args.no_frame_gate else args.frame_gate
    print(
        f"[실험] 프레임 {len(frames)}장, Phase 1: bench (게이트={fgate})…",
        flush=True,
    )
    pq, fq = _popen_server(cmd_qwen, log_q)
    try:
        wait_for_server(f"http://127.0.0.1:{QWEN_PORT}/v1", args.qwen_timeout)
        client_q = OpenAI(
            base_url=f"http://127.0.0.1:{QWEN_PORT}/v1", api_key=api_key
        )
        if fgate in ("", "off", "none"):
            rows_bench = bench(
                client=client_q,
                model=QWEN_MODEL,
                frames=frames,
                prompt=DEFAULT_PROMPT_KO,
                max_tokens=256,
                context_max_side=960,
                crop_max_side=640,
                yolo_model_path="yolov8n.pt",
                max_crops=3,
                min_crop_short_side=args.min_crop_short_side,
                min_crop_area=args.min_crop_area,
            )
        else:
            rows_bench = bench_with_frame_gating(
                client=client_q,
                model=QWEN_MODEL,
                frames=frames,
                prompt=DEFAULT_PROMPT_KO,
                max_tokens=256,
                context_max_side=960,
                crop_max_side=640,
                yolo_model_path="yolov8n.pt",
                max_crops=3,
                frame_gate=fgate,
                mse_threshold=args.mse_threshold,
                min_crop_short_side=args.min_crop_short_side,
                min_crop_area=args.min_crop_area,
                use_crop_layout_gate=args.use_crop_layout_gate,
                crop_gate_max_shift=args.crop_gate_max_shift,
            )
    finally:
        _terminate(pq, fq)

    print("[실험] Phase 2: SmolVLM — 1단계 응답만 수집…", flush=True)
    precached_stage1: dict[str, dict] = {}
    ps, fs = _popen_server(cmd_smol, log_s)
    try:
        wait_for_server(f"http://127.0.0.1:{SMOL_PORT}/v1", args.smol_timeout)
        client_s = OpenAI(
            base_url=f"http://127.0.0.1:{SMOL_PORT}/v1", api_key=api_key
        )
        for i, fp in enumerate(frames):
            print(
                f"[실험] Smol 1단계 {i + 1}/{len(frames)}: {fp.name}",
                flush=True,
            )
            img = Image.open(fp).convert("RGB")
            low = resize_max_side(img, 960)
            low_url = pil_to_data_url(low)
            t0 = time.perf_counter()
            txt, usage_s = chat_vlm_multi(
                client=client_s,
                model=SMOL_MODEL,
                prompt=STAGE1_PROMPT_EN,
                image_data_urls=[low_url],
                max_tokens=256,
            )
            precached_stage1[fp.name] = {
                "text": txt,
                "seconds": round(time.perf_counter() - t0, 3),
                "usage": usage_to_dict(usage_s),
                "image_sent": {
                    "source_frame": fp.name,
                    "path": str(fp.resolve()),
                    "width": low.size[0],
                    "height": low.size[1],
                    "context_max_side": 960,
                },
                "resources_after": resource_snapshot(label="phase2_smol_after"),
            }
    finally:
        _terminate(ps, fs)

    print(
        "[실험] Phase 3: Qwen+Smol 동시 기동 — (YOLO∥Smol)+Qwen, then two-stage (캐시)…",
        flush=True,
    )
    pq3, fq3 = _popen_server(cmd_qwen, log_q3)
    ps3, fs3 = _popen_server(cmd_smol, log_s3)
    rows_parallel: list[dict] = []
    try:
        wait_for_server(f"http://127.0.0.1:{QWEN_PORT}/v1", args.qwen_timeout)
        wait_for_server(f"http://127.0.0.1:{SMOL_PORT}/v1", args.smol_timeout)
        client_q = OpenAI(
            base_url=f"http://127.0.0.1:{QWEN_PORT}/v1", api_key=api_key
        )
        client_s = OpenAI(
            base_url=f"http://127.0.0.1:{SMOL_PORT}/v1", api_key=api_key
        )
        rows_parallel = run_yolo_smol_parallel_qwen(
            client_qwen=client_q,
            model_qwen=QWEN_MODEL,
            client_small=client_s,
            model_small=SMOL_MODEL,
            frames=frames,
            user_prompt=DEFAULT_PROMPT_KO,
            max_tokens=256,
            context_max_side=960,
            crop_max_side=640,
            yolo_model_path="yolov8n.pt",
            max_crops=3,
            stage1_prompt=STAGE1_PROMPT_EN,
            skip_qwen_if_low=True,
            min_crop_short_side=args.min_crop_short_side,
            min_crop_area=args.min_crop_area,
        )
        rows_2 = run_two_stage(
            client_qwen=client_q,
            model_qwen=QWEN_MODEL,
            client_small=None,
            model_small=None,
            frames=frames,
            user_prompt=DEFAULT_PROMPT_KO,
            max_tokens=256,
            context_max_side=960,
            crop_max_side=640,
            yolo_model_path="yolov8n.pt",
            max_crops=3,
            stage1_prompt=STAGE1_PROMPT_EN,
            skip_qwen_if_low=True,
            precached_stage1=precached_stage1,
            min_crop_short_side=args.min_crop_short_side,
            min_crop_area=args.min_crop_area,
        )
    finally:
        _terminate(pq3, fq3)
        _terminate(ps3, fs3)

    latency_compare = _latency_summary(rows_bench, rows_2, rows_parallel)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "frame_gate": {
            "mode": fgate,
            "mse_threshold": float(args.mse_threshold),
            "use_crop_layout_gate": bool(args.use_crop_layout_gate),
            "crop_gate_max_shift": float(args.crop_gate_max_shift),
        },
        "frames_dir": str(args.frames_dir),
        "input_mode": "single_image" if args.single_image else "frames_sequence",
        "single_image": str(args.single_image.resolve())
        if args.single_image
        else None,
        "frame_count": len(frames),
        "qwen_model": QWEN_MODEL,
        "small_vlm_model": SMOL_MODEL,
        "notes": (
            "8GB VRAM: Phase1/2 는 단일 서버. Phase3 는 Qwen+Smol 동시+YOLO(병렬)로 메모리 부담↑ — OOM 시 ng 감소·프레임 수 축소."
        ),
        "resource_sampling_note": (
            "각 VLM 호출 직후 nvidia-smi로 GPU 전체 memory.used/utilization.gpu 스냅샷. "
            "python_rss_mb는 uv run python 프로세스(실험 스크립트)만 — llama-server.exe VRAM은 GPU 행에 포함."
        ),
        "user_prompt": DEFAULT_PROMPT_KO,
        "bench": rows_bench,
        "yolo_smol_parallel_qwen": rows_parallel,
        "two_stage_skip_low": rows_2,
        "smol_stage1_cache": precached_stage1,
        "latency_compare": latency_compare,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[실험] 결과 저장: {args.json_out}", flush=True)
    print(f"[실험] 지연 요약: {json.dumps(latency_compare, ensure_ascii=False)}", flush=True)
    html_path = args.html_out
    if html_path is None:
        html_path = args.json_out.with_suffix(".html")
    try:
        write_experiment_html_report(args.json_out, html_path)
        print(f"[실험] HTML 리포트: {html_path}", flush=True)
    except OSError as e:
        print(f"[실험] HTML 리포트 생략: {e}", flush=True)
    if args.single_image:
        onepage = args.json_out.with_name(args.json_out.stem + "_singleview.html")
        if write_single_image_experiment_html(args.json_out, onepage):
            print(f"[실험] 단일 이미지 한눈에 보기: {onepage}", flush=True)
    _write_auto_results_md(ROOT, payload)


if __name__ == "__main__":
    main()
