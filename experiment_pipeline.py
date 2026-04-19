"""
VLM 실험: Qwen-only vs YOLO+Qwen 벤치, 초소형 VLM(또는 YOLO 대체) → 조건부 Qwen.

사전에 llama-server(Qwen)가 떠 있거나, main.py 와 동일하게 --base-url 로 연결합니다.
YOLO 는 `uv sync --group dev` 후 사용합니다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

from experiment_metrics import resource_snapshot, usage_to_dict
from main import ROOT, chat_vlm_multi, pil_to_data_url
from qwen3_vl_image_tokens import (
    VENDOR_QWEN3VL_4B_MAX_PIXELS,
    VENDOR_QWEN3VL_4B_MIN_PIXELS,
    smart_resize,
    vision_grid_tokens,
)
from yolo_surveillance import (
    load_yolo,
    run_yolo_crops,
    yolo_heuristic_risk,
)

DEFAULT_PROMPT_KO = (
    "이 감시 화면에서 사람·차량·이상 징후가 있는지 짧게 요약해줘. 한국어로."
)
STAGE1_PROMPT_EN = (
    'Reply with ONLY valid JSON, no markdown: '
    '{"objects":["..."],"risk":"low|med|high","note":"one short English sentence"}'
)


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((max(nw, 1), max(nh, 1)), Image.Resampling.LANCZOS)


def approx_vision_tokens_pil(img: Image.Image) -> int:
    w, h = img.size
    rh, rw = smart_resize(
        h,
        w,
        factor=32,
        min_pixels=VENDOR_QWEN3VL_4B_MIN_PIXELS,
        max_pixels=VENDOR_QWEN3VL_4B_MAX_PIXELS,
    )
    return vision_grid_tokens(rh, rw, 16, 2)


def list_frames(frames_dir: Path, max_frames: int) -> list[Path]:
    cands = sorted(frames_dir.glob("frame_*.jpg"))
    if not cands:
        cands = sorted(frames_dir.glob("*.jpg"))
    return cands[:max_frames]


def parse_risk_from_stage1(text: str) -> str:
    text = text.strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            r = str(d.get("risk", "med")).lower()
            if r in ("low", "med", "high"):
                return r
        elif isinstance(d, str):
            r = d.strip().lower()
            if r in ("low", "med", "high"):
                return r
    except json.JSONDecodeError:
        pass
    m = re.search(r'"risk"\s*:\s*"([^"]+)"', text, re.I)
    if m:
        r = m.group(1).lower()
        if r in ("low", "med", "high"):
            return r
    return "med"


def bench(
    *,
    client: OpenAI,
    model: str,
    frames: list[Path],
    prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
) -> list[dict]:
    yolo = load_yolo(yolo_model_path)
    rows: list[dict] = []
    n_frames = len(frames)

    for i, fp in enumerate(frames):
        print(f"[bench] 프레임 {i + 1}/{n_frames}: {fp.name}", flush=True)
        img = Image.open(fp).convert("RGB")
        ctx = resize_max_side(img, context_max_side)

        row: dict = {"frame": str(fp.name), "approx_vision_tokens": {}}

        t0 = time.perf_counter()
        url = pil_to_data_url(ctx)
        tok_ctx = approx_vision_tokens_pil(ctx)
        row["approx_vision_tokens"]["qwen_only"] = tok_ctx

        text_q, usage_q = chat_vlm_multi(
            client=client,
            model=model,
            prompt=prompt,
            image_data_urls=[url],
            max_tokens=max_tokens,
        )
        t_q = time.perf_counter() - t0
        pt_q = getattr(usage_q, "prompt_tokens", None) if usage_q else None

        t1 = time.perf_counter()
        crops, summary, n_det, all_cls = run_yolo_crops(
            img, model=yolo, max_crops=max_crops
        )
        t_yolo = time.perf_counter() - t1

        urls_y = [url]
        tok_sum = tok_ctx
        crop_dims: list[list[int]] = []
        for c in crops:
            rc = resize_max_side(c, crop_max_side)
            crop_dims.append([rc.width, rc.height])
            urls_y.append(pil_to_data_url(rc))
            tok_sum += approx_vision_tokens_pil(rc)
        row["approx_vision_tokens"]["yolo_qwen_images"] = tok_sum
        row["yolo"] = {"n_detections": n_det, "summary": summary, "n_crops_sent": len(crops)}
        row["images"] = {
            "source_path": str(fp.resolve()),
            "qwen_context": {
                "width": ctx.width,
                "height": ctx.height,
                "max_side_cap": context_max_side,
            },
            "yolo_qwen": {
                "context_width": ctx.width,
                "context_height": ctx.height,
                "crop_max_side_cap": crop_max_side,
                "n_images_sent": len(urls_y),
                "crop_dimensions_wh": crop_dims,
            },
        }

        y_prompt = f"[YOLO 프리셋 탐지: {summary}]\n{prompt}"
        t2 = time.perf_counter()
        text_y, usage_y = chat_vlm_multi(
            client=client,
            model=model,
            prompt=y_prompt,
            image_data_urls=urls_y,
            max_tokens=max_tokens,
        )
        t_y = time.perf_counter() - t2
        pt_y = getattr(usage_y, "prompt_tokens", None) if usage_y else None
        uq_d = usage_to_dict(usage_q)
        uy_d = usage_to_dict(usage_y)

        row["qwen_only"] = {
            "seconds": round(t_q, 3),
            "prompt_tokens": pt_q,
            "completion_tokens": (uq_d or {}).get("completion_tokens"),
            "total_tokens": (uq_d or {}).get("total_tokens"),
            "usage": uq_d,
            "approx_image_tokens": tok_ctx,
            "reply": text_q[:12000],
            "reply_preview": text_q[:500],
            "resources_after": resource_snapshot(label="bench_qwen_only_after"),
        }
        row["yolo_qwen"] = {
            "yolo_seconds": round(t_yolo, 3),
            "vlm_seconds": round(t_y, 3),
            "total_seconds": round(t_yolo + t_y, 3),
            "prompt_tokens": pt_y,
            "completion_tokens": (uy_d or {}).get("completion_tokens"),
            "total_tokens": (uy_d or {}).get("total_tokens"),
            "usage": uy_d,
            "approx_image_tokens": tok_sum,
            "reply": text_y[:12000],
            "reply_preview": text_y[:500],
            "resources_after": resource_snapshot(label="bench_yolo_qwen_after"),
        }
        rows.append(row)

    return rows


def run_two_stage(
    *,
    client_qwen: OpenAI,
    model_qwen: str,
    client_small: OpenAI | None,
    model_small: str | None,
    frames: list[Path],
    user_prompt: str,
    max_tokens: int,
    context_max_side: int,
    crop_max_side: int,
    yolo_model_path: str,
    max_crops: int,
    stage1_prompt: str,
    skip_qwen_if_low: bool,
    precached_stage1: dict[str, dict] | None = None,
) -> list[dict]:
    """precached_stage1: 프레임 파일명 → {\"text\", \"source\", \"seconds\"?} (별도 경량 서버에서 미리 수집)."""
    yolo = load_yolo(yolo_model_path)
    out: list[dict] = []
    n_frames = len(frames)

    for i, fp in enumerate(frames):
        print(f"[two-stage] 프레임 {i + 1}/{n_frames}: {fp.name}", flush=True)
        img = Image.open(fp).convert("RGB")
        low = resize_max_side(img, context_max_side)
        low_url = pil_to_data_url(low)

        t_s0 = time.perf_counter()
        stage1_source = "no_small_vlm"
        stage1_text = ""
        prior_err: str | None = None

        stage1_usage_d: dict | None = None
        stage1_res_after: dict | None = None
        stage1_img_meta: dict | None = None

        if precached_stage1 is not None and fp.name in precached_stage1:
            c = precached_stage1[fp.name]
            stage1_text = str(c["text"])
            stage1_source = "small_vlm"
            stage1_usage_d = c.get("usage")
            stage1_res_after = c.get("resources_after")
            stage1_img_meta = c.get("image_sent")
        elif client_small is not None and model_small:
            try:
                stage1_text, usage_s = chat_vlm_multi(
                    client=client_small,
                    model=model_small,
                    prompt=stage1_prompt,
                    image_data_urls=[low_url],
                    max_tokens=256,
                )
                stage1_source = "small_vlm"
                stage1_usage_d = usage_to_dict(usage_s)
                stage1_res_after = resource_snapshot(label="two_stage_smol_after")
                stage1_img_meta = {
                    "source_frame": fp.name,
                    "path": str(fp.resolve()),
                    "width": low.size[0],
                    "height": low.size[1],
                    "context_max_side": context_max_side,
                }
            except Exception as e:
                prior_err = str(e)
                stage1_source = "small_vlm_error"

        if stage1_source != "small_vlm":
            _, summary, n_det, all_cls = run_yolo_crops(
                img, model=yolo, max_crops=max_crops
            )
            risk = yolo_heuristic_risk(
                num_detections=n_det, class_ids_detected=all_cls
            )
            payload: dict = {
                "source": "yolo_fallback",
                "summary": summary,
                "risk": risk,
                "n_detections": n_det,
            }
            if prior_err is not None:
                payload["small_vlm_error"] = prior_err
            stage1_text = json.dumps(payload, ensure_ascii=False)
            stage1_source = "yolo_fallback"

        if precached_stage1 is not None and fp.name in precached_stage1:
            t_stage1 = float(precached_stage1[fp.name].get("seconds", 0.0))
        else:
            t_stage1 = time.perf_counter() - t_s0
        risk = parse_risk_from_stage1(stage1_text)

        row: dict = {
            "frame": str(fp.name),
            "stage1_source": stage1_source,
            "stage1_seconds": round(t_stage1, 3),
            "stage1_text": stage1_text[:12000],
            "parsed_risk": risk,
            "qwen_called": True,
            "input_image": {
                "source_frame": fp.name,
                "path": str(fp.resolve()),
                "width": low.size[0],
                "height": low.size[1],
                "max_side_cap": context_max_side,
            },
        }
        if stage1_usage_d is not None:
            row["stage1_usage"] = stage1_usage_d
            row["stage1_prompt_tokens"] = stage1_usage_d.get("prompt_tokens")
        if stage1_res_after is not None:
            row["stage1_resources_after"] = stage1_res_after
        if stage1_img_meta is not None:
            row["stage1_image_sent"] = stage1_img_meta

        if skip_qwen_if_low and risk == "low":
            row["qwen_called"] = False
            row["qwen_reply"] = "(skipped: stage1 risk=low)"
            row["total_seconds"] = round(t_stage1, 3)
            out.append(row)
            continue

        q_prompt = (
            f"다음은 엣지(경량) 분석 결과입니다. 참고만 하고, 이미지를 직접 보고 판단해줘.\n"
            f"{stage1_text}\n\n"
            f"{user_prompt}"
        )
        t_q0 = time.perf_counter()
        reply, usage = chat_vlm_multi(
            client=client_qwen,
            model=model_qwen,
            prompt=q_prompt,
            image_data_urls=[low_url],
            max_tokens=max_tokens,
        )
        row["qwen_seconds"] = round(time.perf_counter() - t_q0, 3)
        qu_d = usage_to_dict(usage)
        row["qwen_prompt_tokens"] = getattr(usage, "prompt_tokens", None) if usage else None
        row["qwen_completion_tokens"] = (qu_d or {}).get("completion_tokens")
        row["qwen_total_tokens"] = (qu_d or {}).get("total_tokens")
        row["qwen_usage"] = qu_d
        row["qwen_reply"] = reply[:12000]
        row["qwen_resources_after"] = resource_snapshot(label="two_stage_qwen_after")
        row["total_seconds"] = round(t_stage1 + row["qwen_seconds"], 3)
        out.append(row)

    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="VLM 파이프라인 실험 (벤치 / 2단계)")
    p.add_argument(
        "mode",
        choices=("bench", "two-stage"),
        help="bench=Qwen-only vs YOLO+Qwen, two-stage=경량→조건부 Qwen",
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=ROOT / "data" / "datasets" / "demo" / "frames",
    )
    p.add_argument("--max-frames", type=int, default=6)
    p.add_argument("--context-max-side", type=int, default=960)
    p.add_argument("--crop-max-side", type=int, default=640)
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLAMA_OPENAI_BASE", "http://127.0.0.1:8765/v1"),
        help="Qwen llama-server OpenAI 베이스 (/v1 포함)",
    )
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-local"))
    p.add_argument("--model", default="qwen3-vl-4b-q8")
    p.add_argument("--prompt", default=DEFAULT_PROMPT_KO)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--max-crops", type=int, default=3)
    p.add_argument(
        "--small-vlm-url",
        default=os.environ.get("SMALL_VLM_OPENAI_BASE", ""),
        help="경량 VLM llama-server 베이스 (예: http://127.0.0.1:8766/v1). 비우면 YOLO 대체만",
    )
    p.add_argument(
        "--small-vlm-model",
        default=os.environ.get("SMALL_VLM_MODEL", "small-vlm"),
        help="경량 서버의 -a 모델 별칭",
    )
    p.add_argument("--stage1-prompt", default=STAGE1_PROMPT_EN)
    p.add_argument(
        "--skip-qwen-if-low",
        action="store_true",
        help="two-stage: risk=low 이면 Qwen 호출 생략",
    )
    p.add_argument("--json-out", type=Path, default=None, help="결과 JSON 저장 경로")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    client_qwen = OpenAI(base_url=base, api_key=args.api_key)

    frames = list_frames(args.frames_dir, args.max_frames)
    if not frames:
        print(f"프레임 없음: {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "bench":
        rows = bench(
            client=client_qwen,
            model=args.model,
            frames=frames,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            context_max_side=args.context_max_side,
            crop_max_side=args.crop_max_side,
            yolo_model_path=args.yolo_model,
            max_crops=args.max_crops,
        )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        if args.json_out:
            args.json_out.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return

    small_url = (args.small_vlm_url or "").strip()
    client_small: OpenAI | None = None
    model_small: str | None = None
    if small_url:
        b = small_url.rstrip("/")
        if not b.endswith("/v1"):
            b = f"{b}/v1"
        client_small = OpenAI(base_url=b, api_key=args.api_key)
        model_small = args.small_vlm_model

    rows = run_two_stage(
        client_qwen=client_qwen,
        model_qwen=args.model,
        client_small=client_small,
        model_small=model_small,
        frames=frames,
        user_prompt=args.prompt,
        max_tokens=args.max_tokens,
        context_max_side=args.context_max_side,
        crop_max_side=args.crop_max_side,
        yolo_model_path=args.yolo_model,
        max_crops=args.max_crops,
        stage1_prompt=args.stage1_prompt,
        skip_qwen_if_low=args.skip_qwen_if_low,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
