"""실험용: OpenAI usage 직렬화, nvidia-smi / 프로세스 메모리 스냅샷, HTML 리포트.

- :func:`usage_to_dict` — ``chat.completions`` 의 ``usage``(토큰 수) 를 JSON 친화 dict 로.
- :func:`resource_snapshot` — 추론 **직후** GPU 전체( ``nvidia-smi`` )·선택적 Python RSS.
- :func:`write_experiment_html_report` — ``run_week_experiments`` 가 쓰는
  ``experiment_results_last.json`` 구조( ``bench`` , ``yolo_smol_parallel_qwen`` , ``two_stage_skip_low`` )를
  Chart.js 막대 + 표(시나리오별로 프레임 모음)로 렌더한다.
- :func:`write_single_image_experiment_html` / :func:`write_single_image_experiment_html_from_payload` —
  ``input_mode: single_image`` 일 때 **이미지·질문·모델별 응답** 한 페이지( ``*_singleview.html`` ).

JSON 키는 ``docs/experiment_results_last.json`` 과 동일한 스키마를 가정한다.
"""
from __future__ import annotations

import base64
import copy
import html
import json
import math
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from qwen_vlm.main import ROOT


def usage_to_dict(usage: object | None) -> dict[str, Any] | None:
    """OpenAI SDK ``usage`` 객체를 JSON 직렬화 가능한 얕은 dict 로(프롬프트/완료/합계 토큰)."""
    if usage is None:
        return None
    d: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, name, None)
        if v is not None:
            d[name] = int(v)
    return d or None


def resource_snapshot(*, label: str = "") -> dict[str, Any]:
    """
    한 시점의 **시스템/프로세스** 스냅샷(실험 본문에서 "추론 직후"에 호출).

    - **GPU** 행( ``memory.used`` 등 )은 `nvidia-smi` **첫 번째 카드**의 **전체** 사용량.
      ``llama-server`` 가 잡는 VRAM 이 대부분이나, YOLO·다른 PID 가 있으면 합산에 포함된다.
    - **``python_rss_mb``** 는 **이 Python 프로세스**( ``uv run python …`` )의 RSS — llama-server.exe 는 별도.
    - ``label`` 은 JSON 에서 어느 구간 뒤에 찍혔는지 식별용(디버그/리포트용).
    """
    out: dict[str, Any] = {"label": label, "unix_ts": round(time.time(), 3)}
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=creationflags,
        )
        if r.returncode == 0 and r.stdout.strip():
            line = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    u, t, util = float(parts[0]), float(parts[1]), float(parts[2])
                    if math.isfinite(u):
                        out["gpu_memory_used_mb"] = u
                    if math.isfinite(t):
                        out["gpu_memory_total_mb"] = t
                    if math.isfinite(util):
                        out["gpu_utilization_percent"] = util
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass

    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process()
        out["python_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    return out


def _relpath_for_html(p: str | Path | None, base: Path = ROOT) -> str:
    """HTML 표시용: 절대 경로를 저장소 루트(``ROOT``) 기준 상대 경로(슬래시)."""
    if p is None or str(p).strip() == "":
        return "—"
    s = str(p).strip()
    try:
        a = Path(s).resolve()
        b = base.resolve()
        return os.path.relpath(a, b).replace("\\", "/")
    except (ValueError, OSError, TypeError):
        return s.replace("\\", "/")


def _relpath_if_filesystemish(s: str, base: Path = ROOT) -> str:
    if not s or s == "—":
        return s
    if not any(x in s for x in ("/", "\\", ":")):
        return s
    return _relpath_for_html(s, base)


def _relativize_experiment_json_for_html(
    payload: dict[str, Any], base: Path = ROOT
) -> dict[str, Any]:
    """Chart HTML용: ``frames_dir``·``single_image`` 만 상대화(원본 JSON 파일은 그대로)."""
    out = copy.deepcopy(payload)
    for k in ("frames_dir", "single_image"):
        if k in out and out[k]:
            out[k] = _relpath_for_html(out[k], base)
    return out


def write_experiment_html_report(json_path: Path, html_path: Path) -> None:
    """
    `experiment_results_last.json` 스타일 페이로드를 읽어, Chart.js 막대 3개 + 상세 표 HTML 을 쓴다.

    선택 키: ``bench`` , ``yolo_smol_parallel_qwen`` (없으면 빈 배열로 간주) , ``two_stage_skip_low`` .
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    display = _relativize_experiment_json_for_html(payload, ROOT)
    raw = json.dumps(display, ensure_ascii=False).encode("utf-8")
    b64 = base64.standard_b64encode(raw).decode("ascii")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        EXPERIMENT_HTML_TEMPLATE.replace("__B64_PAYLOAD__", b64),
        encoding="utf-8",
    )


def _is_single_shot_experiment_payload(payload: dict[str, Any]) -> bool:
    if payload.get("input_mode") == "single_image" and payload.get("single_image"):
        return True
    return bool(payload.get("single_image") and payload.get("frame_count") == 1)


def _image_data_url_for_path(image_path: Path) -> str | None:
    if not image_path.is_file():
        return None
    data = image_path.read_bytes()
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def write_single_image_experiment_html_from_payload(
    payload: dict[str, Any], html_path: Path
) -> None:
    """같은 프레임·질문·응답을 한 화면에. ``single_image`` 경로·``user_prompt``·bench/parallel/two 를 쓴다."""
    if not _is_single_shot_experiment_payload(payload):
        raise ValueError("single_image/single_shot JSON 이 아닙니다.")
    sp = Path(str(payload.get("single_image", "")))
    data_url = _image_data_url_for_path(sp)
    prompt = (
        payload.get("user_prompt")
        or payload.get("bench_prompt")
        or "(프롬프트 없음 / 이전 JSON)"
    )
    when = str(payload.get("generated_at", ""))
    bench = list(payload.get("bench") or [])
    ysp = list(payload.get("yolo_smol_parallel_qwen") or [])
    two = list(payload.get("two_stage_skip_low") or [])

    def tcell(s: str) -> str:
        return f'<div class="reply">{html.escape(s, quote=False)}</div>'

    part_bench_q = []
    for r in bench:
        fn = str(r.get("frame", "—"))
        rep = (r.get("qwen_only") or {}).get("reply") or ""
        sec = (r.get("qwen_only") or {}).get("seconds")
        part_bench_q.append(
            f"<tr><td class='f'>{html.escape(fn)}</td>"
            f"<td class='num'>{html.escape(str(sec) if sec is not None else '—')}</td>"
            f"<td>{tcell(str(rep)[:20000])}</td></tr>"
        )
    part_bench_y = []
    for r in bench:
        fn = str(r.get("frame", "—"))
        yq = r.get("yolo_qwen") or {}
        rep = yq.get("reply") or ""
        sec = yq.get("total_seconds")
        part_bench_y.append(
            f"<tr><td class='f'>{html.escape(fn)}</td>"
            f"<td class='num'>{html.escape(str(sec) if sec is not None else '—')}</td>"
            f"<td>{tcell(str(rep)[:20000])}</td></tr>"
        )
    part_bench_t = []
    for r in bench:
        tm = r.get("timings_ms")
        if not tm:
            continue
        fn = str(r.get("frame", "—"))
        fg2 = (r.get("frame_gating") or {}).get("path") or "—"
        m2 = (r.get("frame_gating") or {}).get("mse")
        m2s = f"{m2:.2f}" if m2 is not None else "—"
        line = (
            f"읽기 {tm.get('read_resize_ms', 0) or 0} · mse {tm.get('mse_ms', 0) or 0} · "
            f"yolo {tm.get('yolo_ms', 0) or 0} · 1번VLM {tm.get('qwen_only_ms', 0) or 0} · "
            f"2번VLM {tm.get('yolo_qwen_ms', 0) or 0} ms (합계 {tm.get('total_ms', '—')})"
        )
        fg2d = _relpath_if_filesystemish(str(fg2))
        part_bench_t.append(
            f"<tr><td class='f'>{html.escape(fn)}</td>"
            f"<td class='p' title=\"{html.escape(fg2d)}\">절약 경로: {html.escape(fg2d)} / 흑백 {m2s}</td>"
            f"<td class='p' style=\"font-size:0.8rem\">{html.escape(line)}</td></tr>"
        )
    part_par1 = []
    for r in ysp:
        fn = str(r.get("frame", "—"))
        txt = str(r.get("stage1_merged") or "")
        part_par1.append(
            f"<tr><td class='f'>{html.escape(fn)}</td><td>{tcell(txt[:20000])}</td></tr>"
        )
    part_par2 = []
    for r in ysp:
        fn = str(r.get("frame", "—"))
        txt = str(r.get("qwen_reply") or "")
        part_par2.append(
            f"<tr><td class='f'>{html.escape(fn)}</td><td>{tcell(txt[:20000])}</td></tr>"
        )
    part_two1 = []
    for r in two:
        fn = str(r.get("frame", "—"))
        txt = str(r.get("stage1_text") or "")
        part_two1.append(
            f"<tr><td class='f'>{html.escape(fn)}</td><td>{tcell(txt[:20000])}</td></tr>"
        )
    part_two2 = []
    for r in two:
        fn = str(r.get("frame", "—"))
        if r.get("qwen_called", True):
            txt = str(r.get("qwen_reply") or "")
        else:
            txt = str(r.get("qwen_reply") or "(2단 Qwen 생략)")
        part_two2.append(
            f"<tr><td class='f'>{html.escape(fn)}</td><td>{tcell(str(txt)[:20000])}</td></tr>"
        )

    sp_show = _relpath_for_html(sp)
    img_block = (
        f'<p class="warn">이미지를 불러올 수 없습니다: <code>{html.escape(sp_show)}</code></p>'
        if not data_url
        else f'<div class="imgbox"><img src="{data_url}" alt="입력 이미지"/></div>'
    )

    sections: list[str] = []
    if part_bench_q:
        rows = "\n".join(part_bench_q)
        sections.append(
            f"<h2>벤치: Qwen-only (1장)</h2>"
            f'<table class="t"><thead><tr><th>프레임</th><th>초</th><th>응답</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    if part_bench_y:
        rows = "\n".join(part_bench_y)
        sections.append(
            f"<h2>벤치: YOLO+크롭·Qwen</h2>"
            f'<table class="t"><thead><tr><th>프레임</th><th>초</th><th>응답</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    if part_bench_t:
        rows = "\n".join(part_bench_t)
        sections.append(
            f"<h2>벤치: 절약·시간(구간)</h2>"
            f'<table class="t"><thead><tr><th>프레임</th><th>게이트</th><th>구간(ms)</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    if part_par1:
        rows = "\n".join(part_par1)
        sections.append(
            f"<h2>병렬: 1단 YOLO+Smol</h2>"
            f'<table class="t"><thead><tr><th>프레임</th><th>응답/합성</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    if part_par2:
        rows = "\n".join(part_par2)
        sections.append(
            f"<h2>병렬: 2단 Qwen</h2>"
            f'<table class="t"><thead><tr><th>프레임</th><th>응답</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    if part_two1:
        rows = "\n".join(part_two1)
        sections.append(
            f"<h2>2단계: 1단 Smol</h2>"
            f'<table class="t"><thead><tr><th>프레임</th><th>응답</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    if part_two2:
        rows = "\n".join(part_two2)
        sections.append(
            f"<h2>2단계: 2단 Qwen</h2>"
            f'<table class="t"><thead><tr><th>프레임</th><th>응답</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    if not sections:
        sections.append("<p>표시할 응답이 없습니다.</p>")

    body = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>단일 이미지 — 질문 · 응답</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #f4f4f2; color: #222; max-width: 64rem; }}
    h1 {{ font-size: 1.2rem; }}
    h2 {{ font-size: 1.05rem; margin-top: 1.4rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }}
    .meta {{ font-size: 0.9rem; color: #555; margin: 0.5rem 0; }}
    .imgbox {{ background: #111; text-align: center; padding: 0.5rem; border-radius: 8px; margin: 0.8rem 0; }}
    .imgbox img {{ max-width: 100%; max-height: min(60vh, 900px); height: auto; object-fit: contain; }}
    .prompt {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.8rem; white-space: pre-wrap; font-size: 0.9rem; line-height: 1.45; margin: 0.5rem 0; }}
    table.t {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 0.3rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.5rem; vertical-align: top; }}
    th {{ background: #ebebeb; text-align: left; }}
    td.f {{ font-family: Consolas, monospace; font-size: 0.8rem; width: 9rem; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; width: 4rem; }}
    td.p {{ font-size: 0.8rem; }}
    .reply {{ white-space: pre-wrap; max-height: 20rem; overflow: auto; font-size: 0.82rem; line-height: 1.4; }}
    .warn {{ color: #a40; }}
    code {{ font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>단일 이미지 — 질문과 모델별 응답</h1>
  <p class="meta">생성: {html.escape(when)} · 경로(프로젝트 루트 기준): <code>{html.escape(sp_show)}</code></p>
  {img_block}
  <h2>질문(프롬프트)</h2>
  <div class="prompt">{html.escape(str(prompt), quote=False)}</div>
  {chr(10).join(sections)}
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body, encoding="utf-8")


def write_single_image_experiment_html(json_path: Path, html_path: Path) -> bool:
    """
    `experiment_results_last.json` 스타일을 읽어, `single_image` 모드일 때만
    ``*_singleview.html`` (이미지+질문+모델별 답) 을 쓴다. 해당 없으면 ``False``·파일 없음.
    """
    try:
        raw = json_path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not _is_single_shot_experiment_payload(payload):
        return False
    try:
        write_single_image_experiment_html_from_payload(payload, html_path)
    except (ValueError, OSError, KeyError, TypeError):
        return False
    return True


def _jpeg_thumb_data_url(image_path: Path | str, max_side: int = 280) -> str | None:
    """HTML 삽입용 JPEG 썸네일(data URL). 원본이 없거나 열 수 없으면 ``None``."""
    try:
        import io

        from PIL import Image

        p = Path(str(image_path)).expanduser().resolve()
        if not p.is_file():
            return None
        im = Image.open(p).convert("RGB")
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore[attr-defined]
        im.thumbnail((max_side, max_side), resample)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except (OSError, ValueError, TypeError):
        return None


def write_sequence_compare_html(
    payload: dict[str, Any],
    html_path: Path,
) -> None:
    """``compare_sequence_yolo_qwen_baseline_vs_gated`` 결과 dict 를 요약·프레임 표 HTML 로 저장."""
    meta = payload.get("meta") or {}
    summ = payload.get("summary") or {}
    base_rows: list[dict[str, Any]] = list(payload.get("baseline") or [])
    gated_rows: list[dict[str, Any]] = list(payload.get("gated") or [])
    gated_by_frame = {str(r.get("frame", "")): r for r in gated_rows}
    frames_dir = str(meta.get("frames_dir") or "").strip()

    def _yolo_qwen_vlm_ok(r: dict[str, Any]) -> bool:
        yq = r.get("yolo_qwen") or {}
        if yq.get("vlm_skipped") or yq.get("yolo_skipped"):
            return False
        return True

    def _yq_preview(r: dict[str, Any]) -> str:
        yq = r.get("yolo_qwen") or {}
        prev = yq.get("reply_preview") or (yq.get("reply") or "")[:400]
        return str(prev)

    def _gpath(r: dict[str, Any]) -> str:
        fg = r.get("frame_gating")
        if not fg:
            return "—"
        return str(fg.get("path") or fg.get("mode") or "—")

    def _tok_triplet(
        pt: object | None, ct: object | None, tt: object | None
    ) -> str:
        if pt is None and ct is None and tt is None:
            return "—"
        return f"{pt if pt is not None else '—'}/{ct if ct is not None else '—'}/{tt if tt is not None else '—'}"

    def _fmt_baseline_yq(br: dict[str, Any]) -> tuple[str, str]:
        yq = br.get("yolo_qwen") or {}
        if not _yolo_qwen_vlm_ok(br):
            return ("—", "—")
        sec = yq.get("total_seconds")
        tok = _tok_triplet(
            yq.get("prompt_tokens"),
            yq.get("completion_tokens"),
            yq.get("total_tokens"),
        )
        return (html.escape(str(sec) if sec is not None else "—"), html.escape(tok))

    def _fmt_gated_qo(gr: dict[str, Any]) -> tuple[str, str]:
        qo = gr.get("qwen_only") or {}
        if qo.get("vlm_skipped"):
            return (
                html.escape("생략"),
                html.escape("0 / 0 / 0"),
            )
        sec = qo.get("seconds")
        tok = _tok_triplet(
            qo.get("prompt_tokens"),
            qo.get("completion_tokens"),
            qo.get("total_tokens"),
        )
        return (
            html.escape(str(sec) if sec is not None else "—"),
            html.escape(tok),
        )

    def _fmt_gated_yq(gr: dict[str, Any]) -> tuple[str, str]:
        yq = gr.get("yolo_qwen") or {}
        sec = yq.get("total_seconds")
        sec_s = html.escape(str(sec) if sec is not None else "—")
        if yq.get("yolo_skipped") and yq.get("vlm_skipped"):
            return (sec_s, html.escape("0 / 0 / 0"))
        if yq.get("vlm_skipped"):
            return (sec_s, html.escape("0 / 0 / 0 (VLM 생략)"))
        tok = _tok_triplet(
            yq.get("prompt_tokens"),
            yq.get("completion_tokens"),
            yq.get("total_tokens"),
        )
        return (sec_s, html.escape(tok))

    rows_html: list[str] = []
    for br in base_rows:
        fn = str(br.get("frame", ""))
        gr = gated_by_frame.get(fn, {})
        imgs = gr.get("images") or br.get("images") or {}
        src = imgs.get("source_path") if isinstance(imgs, dict) else None
        thumb = _jpeg_thumb_data_url(src, 300) if src else None
        if thumb:
            img_cell = (
                f'<td class="thumbcell"><img class="thumb" src="{thumb}" '
                f'title="{html.escape(str(src))}" alt=""/></td>'
            )
        else:
            rel = _relpath_for_html(str(src) if src else "", ROOT)
            img_cell = f'<td class="thumbcell"><span class="noimg">{html.escape(rel)}</span></td>'

        b_sec, b_tok = _fmt_baseline_yq(br)
        gq_sec, gq_tok = _fmt_gated_qo(gr)
        gy_sec, gy_tok = _fmt_gated_yq(gr)

        rows_html.append(
            "<tr>"
            f"{img_cell}"
            f"<td class='fname'>{html.escape(fn)}</td>"
            f"<td class='num'>{b_sec}</td><td class='num tok'>{b_tok}</td>"
            f"<td class='num'>{gq_sec}</td><td class='num tok'>{gq_tok}</td>"
            f"<td class='num'>{gy_sec}</td><td class='num tok'>{gy_tok}</td>"
            f"<td class='p'>{html.escape(_gpath(gr))}</td>"
            f"<td class='reply'>{html.escape(_yq_preview(gr))}</td>"
            "</tr>"
        )

    cmp_rows = f"""
  <h2>처리 시간·토큰 비교 (요약)</h2>
  <p class="note">「베이스」는 매 프레임 YOLO+크롭 Qwen만 실행합니다. 「게이트」는 <strong>Qwen-only</strong>(전망 1장)와 <strong>YOLO+Qwen</strong>이 각각 언제 호출·생략됐는지의 합계입니다.
  프레임 합산 시간은 YOLO+게이트 구간 전체 wall time 과는 다를 수 있습니다.</p>
  <table class="cmp">
    <thead>
      <tr>
        <th>구분</th>
        <th>VLM 호출 수</th>
        <th>합계 시간(초)<br/><span class="sub">프레임별 합산</span></th>
        <th>prompt 토큰 합</th>
        <th>completion 합</th>
        <th>total 토큰 합</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>베이스 YOLO+Qwen</td>
        <td class="num">{html.escape(str(summ.get("baseline_yolo_qwen_vlm_calls", "")))}</td>
        <td class="num">{html.escape(str(summ.get("baseline_yolo_qwen_wall_seconds_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("baseline_yolo_qwen_prompt_tokens_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("baseline_yolo_qwen_completion_tokens_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("baseline_yolo_qwen_total_tokens_sum", "")))}</td>
      </tr>
      <tr>
        <td>게이트 Qwen-only</td>
        <td class="num">{html.escape(str(summ.get("gated_qwen_only_vlm_calls", "")))}</td>
        <td class="num">{html.escape(str(summ.get("gated_qwen_only_wall_seconds_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("gated_qwen_only_prompt_tokens_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("gated_qwen_only_completion_tokens_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("gated_qwen_only_total_tokens_sum", "")))}</td>
      </tr>
      <tr>
        <td>게이트 YOLO+Qwen (VLM 호출분만 토큰 합산)</td>
        <td class="num">{html.escape(str(summ.get("gated_yolo_qwen_vlm_calls", "")))}</td>
        <td class="num">{html.escape(str(summ.get("gated_yolo_qwen_wall_seconds_sum", "")))}<br/><span class="sub">YOLO 포함·프레임별 total</span></td>
        <td class="num">{html.escape(str(summ.get("gated_yolo_qwen_prompt_tokens_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("gated_yolo_qwen_completion_tokens_sum", "")))}</td>
        <td class="num">{html.escape(str(summ.get("gated_yolo_qwen_total_tokens_sum", "")))}</td>
      </tr>
    </tbody>
  </table>
"""

    body = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>연속 프레임 — YOLO+Qwen 베이스라인 vs 게이트</title>
  <style>
    body {{ font-family: system-ui, "Malgun Gothic", sans-serif; margin: 1rem; background: #fafafa; color: #222; }}
    h1 {{ font-size: 1.2rem; }}
    h2 {{ font-size: 1.05rem; margin-top: 1.25rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin-top: 0.6rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.45rem; vertical-align: top; }}
    th {{ background: #eee; text-align: left; }}
    table.cmp th, table.cmp td {{ font-size: 0.8rem; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    td.tok {{ font-size: 0.75rem; }}
    td.reply {{ white-space: pre-wrap; max-height: 10rem; overflow: auto; font-size: 0.78rem; }}
    td.p {{ font-size: 0.78rem; max-width: 7rem; }}
    td.fname {{ font-family: Consolas, monospace; font-size: 0.78rem; }}
    td.thumbcell {{ width: 10rem; text-align: center; background: #f6f6f6; }}
    img.thumb {{ max-height: 140px; max-width: 180px; object-fit: contain; vertical-align: middle; border-radius: 4px; }}
    span.noimg {{ font-size: 0.72rem; color: #666; }}
    .sum {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem; max-width: 48rem; }}
    .sum dt {{ font-weight: 600; margin-top: 0.35rem; }}
    .sum dd {{ margin: 0.1rem 0 0 0.8rem; }}
    .note {{ font-size: 0.8rem; color: #444; max-width: 52rem; line-height: 1.45; }}
    span.sub {{ font-weight: 400; font-size: 0.72rem; color: #666; }}
  </style>
</head>
<body>
  <h1>YOLO+Qwen 매 프레임(베이스라인) vs 프레임 게이트</h1>
  <p>생성: {html.escape(str(meta.get("generated_at", "")))}
   · <strong>frames_dir</strong>: {html.escape(frames_dir or "—")}
   · frame_gate: {html.escape(str(meta.get("frame_gate", "")))}</p>
  <div class="sum">
    <dl>
      <dt>프레임 수</dt><dd>{html.escape(str(summ.get("frames", "")))}</dd>
      <dt>베이스라인 — YOLO+Qwen VLM 호출 수</dt><dd>{html.escape(str(summ.get("baseline_yolo_qwen_vlm_calls", "")))}</dd>
      <dt>게이트 — YOLO+Qwen VLM 호출 수</dt><dd>{html.escape(str(summ.get("gated_yolo_qwen_vlm_calls", "")))}</dd>
      <dt>절약(베이스 대비 YOLO+Qwen VLM 스킵 수)</dt><dd>{html.escape(str(summ.get("saved_yolo_qwen_vlm_calls_vs_baseline", "")))}</dd>
      <dt>게이트 — Qwen-only VLM 호출 수</dt><dd>{html.escape(str(summ.get("gated_qwen_only_vlm_calls", "")))}</dd>
      <dt>게이트 파이프라인 wall 시간 (베이스 / 게이트 / 합계)</dt><dd>{html.escape(str(meta.get("seconds_baseline", "")))} s · {html.escape(str(meta.get("seconds_gated", "")))} s · {html.escape(str(meta.get("seconds_total", "")))} s</dd>
    </dl>
  </div>
{cmp_rows}
  <h2>프레임별 — 이미지 · 시간·토큰 · 응답</h2>
  <p class="note">썸네일은 리포트 생성 시점에 디스크에서 읽어 JPEG 로 넣습니다. p/c/t = prompt / completion / total (서버 보고).</p>
  <table>
    <thead>
      <tr>
        <th>이미지</th>
        <th>프레임</th>
        <th>베이스 Y+Q<br/>초</th>
        <th>베이스<br/>p/c/t</th>
        <th>게이트 Q-only<br/>초</th>
        <th>게이트 Q-only<br/>p/c/t</th>
        <th>게이트 Y+Q<br/>초</th>
        <th>게이트 Y+Q<br/>p/c/t</th>
        <th>게이트 경로</th>
        <th>게이트 Y+Q 응답(일부)</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body, encoding="utf-8")


EXPERIMENT_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>실험 결과</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1.2rem; background: #fafafa; color: #222; }
    h1 { font-size: 1.25rem; }
    h2 { font-size: 1.05rem; margin-top: 1.5rem; }
    h2 .card-title-note { font-weight: 400; font-size: 0.78rem; color: #666; display: block; margin-top: 0.2rem; }
    .note { font-size: 0.85rem; color: #555; max-width: 52rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem 1rem; }
    .lead { font-size: 0.95rem; max-width: 52rem; line-height: 1.5; color: #333; }
    details.glossary { max-width: 52rem; margin: 0.8rem 0; padding: 0.6rem 0.9rem; background: #f5f5f8; border: 1px solid #d8d8e0; border-radius: 6px; }
    details.glossary summary { cursor: pointer; font-weight: 600; color: #2a2a4a; }
    details.glossary .g-body { margin-top: 0.6rem; font-size: 0.86rem; line-height: 1.5; color: #444; }
    details.glossary dl { margin: 0; }
    details.glossary dt { margin-top: 0.55em; font-weight: 600; color: #222; }
    details.glossary dt:first-child { margin-top: 0; }
    details.glossary dd { margin: 0.15em 0 0 0.8rem; }
    th[title], abbr[title] { cursor: help; border-bottom: 1px dotted #888; text-decoration: none; }
    table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
    th, td { border: 1px solid #ddd; padding: 0.35rem 0.5rem; vertical-align: top; }
    th { background: #f0f0f0; }
    #tables h3 { font-size: 0.95rem; margin: 1rem 0 0.35rem; color: #333; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .reply { white-space: pre-wrap; max-height: 12rem; overflow: auto; font-size: 0.78rem; }
    canvas { max-height: 280px; }
  </style>
</head>
<body>
  <h1 title="VLM=이미지와 글을 같이 이해하는 대형 AI">영상(프레임) 감시 실험 리포트</h1>
  <p class="lead">이 페이지는 <strong>같은 프레임(이미지)에 대해 여러 가지로 AI를 돌렸을 때</strong> 걸리는 시간·GPU·토큰(입력 길이 대략치)을 나란히 보여 줍니다. 용어가 낯설면 아래 <strong>「용어 설명」</strong>을 먼저 펼쳐 보세요. 표·제목에 마우스를 올리면(<span style="border-bottom:1px dotted #888">밑줄 점선</span>) 짧은 설명이 뜨는 곳이 있습니다.</p>
  <p class="note" id="meta"></p>
  <details class="glossary" open>
    <summary>용어 설명 (접기/펼치기)</summary>
    <div class="g-body">
      <dl>
        <dt>Qwen / Qwen-only</dt>
        <dd>이 실험의 <strong>큰 VLM(비전+언어 모델)</strong>. <strong>Qwen-only</strong>는 전체 화면을 작게 맞춘 이미지 <strong>1장</strong>만 넣어 물어본 경우입니다.</dd>
        <dt>YOLO + Qwen (YOLO+Qwen)</dt>
        <dd><strong>YOLO</strong>는 사진 안의 물체(사람, 차 등)를 빠르게 찾는 작은 AI입니다. 찾은 부분을 <strong>잘라낸 그림(크롭)</strong>까지 같이 VLM에 넣어 답을 요청한 파이프라인입니다. 그림 장수가 늘어 <strong>시간·토큰이 Qwen-only보다 클 수</strong> 있습니다.</dd>
        <dt>Smol (SmolVLM)</dt>
        <dd>Qwen보다 가벼운 <strong>작은 VLM</strong>. 먼저 짧은 판단·요약을 하고, 필요하면 Qwen을 부르는 <strong>2단계</strong>에서 쓰입니다.</dd>
        <dt>YOLO∥Smol → Qwen</dt>
        <dd><strong>∥(병렬)</strong>은 YOLO와 Smol을 <strong>동시에</strong> 돌린 뒤, 합쳐서 Qwen에 넘기는 실험 줄입니다. 합쳐진 벽시계 시간은 표에 <code>wall</code>로 적힌 구간이 있습니다.</dd>
        <dt>2단계 (Smol → Qwen)</dt>
        <dd>1단에서 Smol(또는 YOLO만으로 짠 규칙)으로 요약·위험도를 보고, 조건이 맞으면 2단 Qwen을 <strong>건너뛸</strong> 수 있습니다.</dd>
        <dt>입력 토큰 (prompt_tokens, 표에서는 p/c/t)</dt>
        <dd>서버가 알려 주는 <strong>입력 길이</strong> 대략치입니다. <strong>p</strong>=입력, <strong>c</strong>=생성(답), <strong>t</strong>=합계. 이미지는 토큰으로 환산되어 포함될 수 있습니다.</dd>
        <dt>VLM 생략(0) / «비전~N»</dt>
        <dd>프레임이 이전과 비슷해 <strong>이번엔 VLM에 요청을 안 보낸</strong> 경우, p/c/t는 <strong>0/0/0</strong>으로 보입니다. 뒤의 «비전~숫자»는 <strong>보냈을 때의 비전 토큰(규칙으로 추정)</strong>이며, 실제 API에 청구된 값이 아닙니다.</dd>
        <dt>GPU (MiB) / util%</dt>
        <dd>그래픽 메모리 사용량(메비바이트)과 GPU 바쁨 정도(%)입니다. 전체 <strong>그래픽 카드</strong> 기준이라, 다른 프로그램이 쓰는 양이 합쳐질 수 있습니다.</dd>
        <dt>Python RSS</dt>
        <dd>이 <strong>실험을 돌리는 Python 프로세스</strong>가 쓰는 일반 메모리(RAM)입니다. <code>llama-server</code>는 별도 프로세스일 수 있습니다.</dd>
        <dt>프레임 게이트 / MSE / YOLO 서명</dt>
        <dd>연속 사진이 <strong>이전과 거의 같으면</strong> VLM을 다시 안 돌리기 위한 <strong>절약</strong>입니다. <strong>MSE</strong>는 흑백 썸네일끼리 얼마나 닮았는지(작을수록 닮음). <strong>YOLO 서명</strong>은 “몇 개 찾았는지+어떤 종류인지”를 짧은 글로 비교하며, <strong>위치</strong>는 보지 않습니다.</dd>
        <dt>구간·게이트 (ms)</dt>
        <dd>읽기·MSE·YOLO·Qwen 호출마다 <strong>걸린 밀리초</strong>를 나누어 쓴 줄입니다. 스킵한 프레임은 일부가 0에 가깝게 보일 수 있습니다.</dd>
      </dl>
    </div>
  </details>
  <p class="note" title="OpenAI 호환 API 서버(예: llama-server)가 보내온 측정값">GPU·아래 토큰 수는 <strong>서버/드라이버가 보내온 값</strong>입니다. GPU는 <code>nvidia-smi</code>의 <strong>카드 전체</strong>이며, Python의 <code>python_rss_mb</code>는 <strong>이 스크립트만</strong>의 RAM입니다.</p>
  <p class="note" title="병렬 구간+2번째 VLM, 또는 위험도 낮을 때 2번째 VLM 생략">시간: <strong>YOLO∥Smol→Qwen</strong> 행 = YOLO와 Smol을 <strong>동시에</strong> 돌린 뒤 이어서 Qwen을 호출한 흐름(벽시계 <code>wall</code> 포함)입니다. 2단계에서 <code>risk=low</code>이면 Qwen을 생략한 실행도 있습니다. 벤치의 <strong>YOLO+Qwen</strong>은 <strong>크롭 이미지 여러 장</strong>이므로 다른 줄과 토큰·시간을 <strong>직접 비교하기 어려울 수</strong> 있습니다.</p>
  <p class="note" id="fgNote" style="display:none"></p>

  <div class="grid">
    <div class="card"><h2 title="각 막대: 해당 시나리오에서 서버 응답이 돌아오기까지 걸린 초(벽시계)">응답까지 걸린 시간(초)</h2><span class="card-title-note">짧을수록 빠름. 시나리오마다 보내는 그림 수가 달라 직접만 비교는 어려울 수 있음</span><canvas id="chartTime"></canvas></div>
    <div class="card"><h2 title="OpenAI API usage의 prompt/completion 총계 중 입력 쪽(이미지·텍스트 합산)">입력 토큰 수 (서버 보고)</h2><span class="card-title-note">p/c/t = 프롬프트(입력) / 생성(답) / 합계 — 여기서 막대는 주로 prompt 쪽</span><canvas id="chartPromptTok"></canvas></div>
    <div class="card"><h2 title="해당 추론 직후, GPU0 전체 점유(MiB)">추론 직후 GPU 메모리 (MiB)</h2><span class="card-title-note">다른 앱이 GPU를 쓰면 합쳐서 보일 수 있음</span><canvas id="chartGpu"></canvas></div>
  </div>

  <div id="benchMsBlock" style="display:none" class="grid">
    <div class="card" style="grid-column: 1 / -1;">
      <h2 title="한 프레임을 처리하는 동안, 단계마다 쓴 시간을 겹쳐 쌓은 막대">벤치: 단계별 시간 (ms, 누적 막대)</h2>
      <p class="note" title="read=읽기·리사이즈, MSE=흑백 유사도, q1=첫 VLM, q2=크롭 포함 두 번째 VLM">프레임이 <strong>이전과 비슷해 생략</strong>한 경우, 아래 측정에서 일부는 0ms에 가깝게 나올 수 있습니다.</p>
      <canvas id="chartMs" height="200"></canvas>
    </div>
  </div>

  <h2 title="같은 시나리오(같은 모델/줄) 아래에 프레임이 순서대로 모입니다.">시나리오별 상세 (모델·줄마다 모음)</h2>
  <div id="tables"></div>

  <script>
  function decodeB64Utf8(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return JSON.parse(new TextDecoder('utf-8').decode(bytes));
  }
  const P = decodeB64Utf8('__B64_PAYLOAD__');
  if (P.frame_gate) {
    const fn = document.getElementById('fgNote');
    fn.style.display = 'block';
    const md = P.frame_gate.mode || '—';
    const mth = P.frame_gate.mse_threshold != null ? P.frame_gate.mse_threshold : '—';
    fn.textContent = `「프레임 게이트(절약)」켜짐: 방식=${md} · 흑백 유사(MSE) 기준=${mth} (작을수록 «이전과 닮음»을 엄격히 봄).`;
    fn.setAttribute('title', '이전 프레임과 닮으면 VLM(큰 AI)을 다시 안 돌리는 옵션입니다. mse_then_yolo는 먼저 MSE, 그다음 YOLO “서명(개수+종류)”으로 판단합니다.');
  }
  document.getElementById('meta').textContent =
    `생성: ${P.generated_at || ''} · 프레임 ${P.frame_count || 0}장 · ${P.frames_dir || ''}`;

  const bench = P.bench || [];
  const labels = bench.map(r => r.frame || '');
  const two = P.two_stage_skip_low || [];
  const parallel = P.yolo_smol_parallel_qwen || [];
  const twoByF = Object.fromEntries(two.map(x => [x.frame, x]));
  const parByF = Object.fromEntries(parallel.map(x => [x.frame, x]));

  const qOnlyS = bench.map(r => (r.qwen_only && r.qwen_only.seconds) ?? null);
  const yoloS = bench.map(r => (r.yolo_qwen && r.yolo_qwen.total_seconds) ?? null);
  const twoTotal = labels.map(f => (twoByF[f] && twoByF[f].total_seconds) ?? null);
  const parTotal = labels.map(f => (parByF[f] && parByF[f].total_seconds) ?? null);

  const qTok = bench.map(r => (r.qwen_only && r.qwen_only.vlm_skipped) ? 0 : ((r.qwen_only && r.qwen_only.prompt_tokens) ?? null));
  const yTok = bench.map(r => (r.yolo_qwen && r.yolo_qwen.vlm_skipped) ? 0 : ((r.yolo_qwen && r.yolo_qwen.prompt_tokens) ?? null));
  const smolTok = labels.map(f => (twoByF[f] && twoByF[f].stage1_usage && twoByF[f].stage1_usage.prompt_tokens) ?? null);
  const qwen2Tok = labels.map(f => (twoByF[f] && twoByF[f].qwen_prompt_tokens) ?? null);
  const parSmolPTok = labels.map(f => (parByF[f] && parByF[f].stage1_usage && parByF[f].stage1_usage.prompt_tokens) ?? null);
  const parQwenPTok = labels.map(f => (parByF[f] && parByF[f].qwen_prompt_tokens) ?? null);

  const gpuQ = bench.map(r => (r.qwen_only && r.qwen_only.resources_after && r.qwen_only.resources_after.gpu_memory_used_mb) ?? null);
  const gpuY = bench.map(r => (r.yolo_qwen && r.yolo_qwen.resources_after && r.yolo_qwen.resources_after.gpu_memory_used_mb) ?? null);
  const gpuSmol = labels.map(f => (twoByF[f] && twoByF[f].stage1_resources_after && twoByF[f].stage1_resources_after.gpu_memory_used_mb) ?? null);
  const gpuQ2 = labels.map(f => (twoByF[f] && twoByF[f].qwen_resources_after && twoByF[f].qwen_resources_after.gpu_memory_used_mb) ?? null);
  const gpuParEnd = labels.map(f => (parByF[f] && parByF[f].qwen_resources_after && parByF[f].qwen_resources_after.gpu_memory_used_mb) ?? null);

  if (labels.length) {
    new Chart(document.getElementById('chartTime'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Qwen만 (전체 1장)', data: qOnlyS, backgroundColor: 'rgba(54,162,235,0.7)' },
          { label: 'YOLO+크롭·Qwen', data: yoloS, backgroundColor: 'rgba(255,159,64,0.7)' },
          { label: '병렬: YOLO·Smol 후 Qwen', data: parTotal, backgroundColor: 'rgba(144,202,90,0.75)' },
          { label: '2단계: 1단 후 Qwen', data: twoTotal, backgroundColor: 'rgba(75,192,192,0.7)' },
        ]
      },
      options: { responsive: true, scales: { x: { stacked: false } } }
    });
    new Chart(document.getElementById('chartPromptTok'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Qwen만: 입력 토큰', data: qTok, backgroundColor: 'rgba(54,162,235,0.7)' },
          { label: 'YOLO+크롭: 입력 토큰', data: yTok, backgroundColor: 'rgba(255,159,64,0.7)' },
          { label: '병렬·1단(Smol) 입력', data: parSmolPTok, backgroundColor: 'rgba(144,202,90,0.6)' },
          { label: '병렬·2단 Qwen 입력', data: parQwenPTok, backgroundColor: 'rgba(100,180,50,0.6)' },
          { label: '2단계·1단(Smol) 입력', data: smolTok, backgroundColor: 'rgba(153,102,255,0.7)' },
          { label: '2단계·2단 Qwen 입력', data: qwen2Tok, backgroundColor: 'rgba(75,192,192,0.7)' },
        ]
      },
      options: { responsive: true }
    });
    new Chart(document.getElementById('chartGpu'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Qwen만 직후 GPU', data: gpuQ, backgroundColor: 'rgba(54,162,235,0.5)' },
          { label: 'YOLO+크롭 직후 GPU', data: gpuY, backgroundColor: 'rgba(255,159,64,0.5)' },
          { label: '병렬·2단 Qwen 직후', data: gpuParEnd, backgroundColor: 'rgba(120,200,60,0.5)' },
          { label: '2단계·1단 직후', data: gpuSmol, backgroundColor: 'rgba(153,102,255,0.5)' },
          { label: '2단계·2단 Qwen 직후', data: gpuQ2, backgroundColor: 'rgba(75,192,192,0.5)' },
        ]
      },
      options: { responsive: true }
    });
  }

  const fg = P.frame_gate;
  if (fg && (bench[0] && bench[0].timings_ms)) {
    document.getElementById('benchMsBlock').style.display = 'grid';
    const bbf = Object.fromEntries(bench.map(x => [x.frame, x]));
    const mrr = labels.map(f => (bbf[f] && bbf[f].timings_ms && bbf[f].timings_ms.read_resize_ms) || 0);
    const mms = labels.map(f => (bbf[f] && bbf[f].timings_ms && bbf[f].timings_ms.mse_ms) || 0);
    const mjy = labels.map(f => (bbf[f] && bbf[f].timings_ms && bbf[f].timings_ms.yolo_ms) || 0);
    const mq1 = labels.map(f => (bbf[f] && bbf[f].timings_ms && bbf[f].timings_ms.qwen_only_ms) || 0);
    const mq2 = labels.map(f => (bbf[f] && bbf[f].timings_ms && bbf[f].timings_ms.yolo_qwen_ms) || 0);
    new Chart(document.getElementById('chartMs'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: '읽기·리사이즈', data: mrr, backgroundColor: 'rgba(180,180,200,0.9)', stack: 't' },
          { label: '흑백·유사도(MSE)', data: mms, backgroundColor: 'rgba(200,100,200,0.7)', stack: 't' },
          { label: '물체찾기(YOLO)', data: mjy, backgroundColor: 'rgba(255,159,64,0.7)', stack: 't' },
          { label: '1번 VLM (Qwen만)', data: mq1, backgroundColor: 'rgba(54,162,235,0.7)', stack: 't' },
          { label: '2번 VLM (크롭 포함)', data: mq2, backgroundColor: 'rgba(75,90,200,0.6)', stack: 't' },
        ]
      },
      options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true, title: { display: true, text: 'ms' } } } }
    });
  }

  function esc(s) {
    if (s == null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  const head = '<th title="파일 이름(한 장의 이미지)">프레임</th>'
    + '<th title="응답이 돌아오기까지 걸린 시간(초)">시간(초)</th>'
    + '<th title="서버가 알려 준 토큰: 입력/생성/합계 (이미지도 토큰에 포함)"><abbr title="prompt / completion / total tokens">p/c/t</abbr> 토큰</th>'
    + '<th title="전송한 그림의 크기(픽셀) 또는 몇 장을 넣었는지">이미지</th>'
    + '<th title="이 추론이 끝난 직후 GPU0 전체 메모리·부하(%)">GPU(MiB) / %</th>'
    + '<th title="이 Python 스크립트만의 RAM(메가바이트)">Python RAM</th>'
    + '<th title="모델이 돌려 준 답(일부 잘림)">응답 요약</th>';
  let html = '';
  if (bench.length) {
    html += '<h3 title="전체 화면 1장만 큰 VLM(Qwen)">벤치: Qwen-only (1장)</h3><table><thead><tr>' + head + '</tr></thead><tbody>';
    for (const r of bench) {
      const img = r.images && r.images.qwen_context ? `${r.images.qwen_context.width}×${r.images.qwen_context.height}` : '';
      const uq = r.qwen_only && r.qwen_only.usage;
      const aqv = r.approx_vision_tokens && r.approx_vision_tokens.qwen_only;
      const vlmSk0 = r.qwen_only && r.qwen_only.vlm_skipped;
      const tok = vlmSk0
        ? `0 / 0 / 0 (VLM 생략)${(aqv != null) ? ' · 비전~'+aqv : ''}`
        : (uq ? `${uq.prompt_tokens ?? '—'}/${uq.completion_tokens ?? '—'}/${uq.total_tokens ?? '—'}` : '');
      const ra = r.qwen_only && r.qwen_only.resources_after;
      const gpu = ra && ra.gpu_memory_used_mb;
      const util = ra && ra.gpu_utilization_percent;
      const rss = ra && ra.python_rss_mb;
      const reply = (r.qwen_only && r.qwen_only.reply) || '';
      html += `<tr><td>${esc(r.frame)}</td><td class="num">${r.qwen_only && r.qwen_only.seconds}</td><td class="num" title="${vlmSk0 ? '이번엔 API 호출 없음' : ''}">${tok}</td><td>${esc(img)}</td><td class="num">${gpu != null ? gpu : '—'} / ${util != null ? util : '—'}</td><td class="num">${rss ?? '—'}</td><td class="reply">${esc(reply)}</td></tr>`;
    }
    html += '</tbody></table>';
    html += '<h3 title="YOLO로 잘라낸 그림(크롭)까지 VLM">벤치: YOLO+크롭·Qwen</h3><table><thead><tr>' + head + '</tr></thead><tbody>';
    for (const r of bench) {
      const uy = r.yolo_qwen && r.yolo_qwen.usage;
      const aym = r.approx_vision_tokens && r.approx_vision_tokens.yolo_qwen_images;
      const vlmSk1 = r.yolo_qwen && r.yolo_qwen.vlm_skipped;
      const toky = vlmSk1
        ? `0 / 0 / 0 (VLM 생략)${(aym != null) ? ' · 비전(보냈다면)~'+aym : ''}`
        : (uy ? `${uy.prompt_tokens ?? '—'}/${uy.completion_tokens ?? '—'}/${uy.total_tokens ?? '—'}` : '');
      const imgsY = r.images && r.images.yolo_qwen ? `${r.images.yolo_qwen.n_images_sent}장 · ctx ${r.images.yolo_qwen.context_width}×${r.images.yolo_qwen.context_height}` : '';
      const ray = r.yolo_qwen && r.yolo_qwen.resources_after;
      const gpuY = ray && ray.gpu_memory_used_mb;
      const utilY = ray && ray.gpu_utilization_percent;
      const rssY = ray && ray.python_rss_mb;
      const replyY = (r.yolo_qwen && r.yolo_qwen.reply) || '';
      html += `<tr><td>${esc(r.frame)}</td><td class="num">${r.yolo_qwen && r.yolo_qwen.total_seconds}</td><td class="num" title="${vlmSk1 ? (r.yolo_qwen.yolo_skipped ? 'VLM·YOLO 모두 이번에 안 씀' : 'YOLO는 이번에 실행, VLM만 미호출') : ''}">${toky}</td><td>${esc(imgsY)}</td><td class="num">${gpuY != null ? gpuY : '—'} / ${utilY != null ? utilY : '—'}</td><td class="num">${rssY ?? '—'}</td><td class="reply">${esc(replyY)}</td></tr>`;
    }
    html += '</tbody></table>';
    const trows = [];
    for (const r of bench) {
      if (r.timings_ms) {
        const tm = r.timings_ms;
        const fg2 = (r.frame_gating && r.frame_gating.path) || '—';
        const m2 = (r.frame_gating && r.frame_gating.mse != null) ? r.frame_gating.mse.toFixed(2) : '—';
        const totm = (tm.total_ms != null) ? String(tm.total_ms) : '—';
        trows.push(`<tr><td>${esc(r.frame)}</td><td title="단계별로 몇 ms, 스킵 이유">절약·시간(구간)</td><td class="num">${(totm)}ms · ${esc(fg2)}</td><td>—</td><td title="MSE(휘도)">흑백 ${esc(m2)}</td><td>—</td><td>—</td><td class="reply" style="font-size:0.75rem">읽기 ${(tm.read_resize_ms|0)} · mse ${(tm.mse_ms|0)} · yolo ${(tm.yolo_ms|0)} · 1번VLM ${(tm.qwen_only_ms|0)} · 2번VLM ${(tm.yolo_qwen_ms|0)} ms</td></tr>`);
      }
    }
    if (trows.length) {
      html += '<h3>벤치: 절약·시간(구간)</h3><table><thead><tr>'
        + '<th>프레임</th><th>시나리오</th><th>시간</th><th colspan="2">p/c·이미지</th><th colspan="2">GPU</th><th>구간(ms) 요약</th></tr></thead><tbody>' + trows.join('') + '</tbody></table>';
    }
  }
  if (parallel.length) {
    html += '<h3>병렬: 1단 YOLO+Smol (동시)</h3><table><thead><tr>' + head + '</tr></thead><tbody>';
    for (const r of parallel) {
      const w = r.yolo_smol_parallel || {};
      const imgP = r.input_image ? `${r.input_image.width}×${r.input_image.height}` : '';
      const sra = r.stage1_resources_after;
      const g1 = sra && sra.gpu_memory_used_mb;
      const u1 = sra && sra.gpu_utilization_percent;
      const rs1 = sra && sra.python_rss_mb;
      const su = r.stage1_usage;
      const stok1 = su ? `${su.prompt_tokens ?? '—'}/${su.completion_tokens ?? '—'}/${su.total_tokens ?? '—'}` : '—';
      const wallN = w.wall_parallel_seconds != null ? `yolo ${w.yolo_seconds ?? '—'}s + smol ${w.smol_seconds ?? '—'}s → wall ${w.wall_parallel_seconds}s` : '—';
      const sm = r.yolo && r.yolo.summary ? '· '+String(r.yolo.summary).slice(0,80) : '';
      const rep = (r.stage1_merged || '').slice(0, 2000);
      html += `<tr><td>${esc(r.frame)}</td><td class="num">${esc(wallN)}</td><td class="num">${stok1}</td><td title="YOLO 요약(일부)">${esc(imgP)}${esc(sm)}</td><td class="num">${g1 != null ? g1 : '—'} / ${u1 != null ? u1 : '—'}</td><td class="num">${rs1 ?? '—'}</td><td class="reply">${esc(rep)}</td></tr>`;
    }
    html += '</tbody></table><h3>병렬: 2단 Qwen (병렬 파이프)</h3><table><thead><tr>' + head + '</tr></thead><tbody>';
    for (const r of parallel) {
      const imgP = r.input_image ? `${r.input_image.width}×${r.input_image.height}` : '';
      const qu = r.qwen_usage;
      const tokp = qu ? `${qu.prompt_tokens ?? '—'}/${qu.completion_tokens ?? '—'}/${qu.total_tokens ?? '—'}` : (r.qwen_called ? '—' : '스킵');
      const qra = r.qwen_resources_after;
      const g2 = qra && qra.gpu_memory_used_mb;
      const u2 = qra && qra.gpu_utilization_percent;
      const rs2 = qra && qra.python_rss_mb;
      const rep = (r.qwen_reply || '').slice(0,4000);
      const imgCell = imgP + ' 합산 ' + (r.total_seconds != null ? r.total_seconds : '—') + 's';
      html += `<tr><td>${esc(r.frame)}</td><td class="num">${r.qwen_seconds != null ? r.qwen_seconds : '—'}</td><td class="num">${tokp}</td><td>${esc(imgCell)}</td><td class="num">${g2 != null ? g2 : '—'} / ${u2 != null ? u2 : '—'}</td><td class="num">${rs2 ?? '—'}</td><td class="reply">${esc(rep)}</td></tr>`;
    }
    html += '</tbody></table>';
  }
  if (two.length) {
    html += '<h3>2단계: 1단 Smol(요약)</h3><table><thead><tr>' + head + '</tr></thead><tbody>';
    for (const r of two) {
      const su = r.stage1_usage;
      const stok = su ? `${su.prompt_tokens ?? '—'}/${su.completion_tokens ?? '—'}/${su.total_tokens ?? '—'}` : '—';
      const img = r.input_image ? `${r.input_image.width}×${r.input_image.height}` : '';
      const sra = r.stage1_resources_after;
      const g1 = sra && sra.gpu_memory_used_mb;
      const u1 = sra && sra.gpu_utilization_percent;
      const rs1 = sra && sra.python_rss_mb;
      const rep = (r.stage1_text || '').slice(0,4000);
      html += `<tr><td>${esc(r.frame)}</td><td class="num">${r.stage1_seconds}</td><td class="num">${stok}</td><td>${esc(img)}</td><td class="num">${g1 != null ? g1 : '—'} / ${u1 != null ? u1 : '—'}</td><td class="num">${rs1 ?? '—'}</td><td class="reply">${esc(rep)}</td></tr>`;
    }
    html += '</tbody></table><h3>2단계: 2단 Qwen</h3><table><thead><tr>' + head + '</tr></thead><tbody>';
    for (const r of two) {
      const qu = r.qwen_usage;
      const qtok = qu ? `${qu.prompt_tokens ?? '—'}/${qu.completion_tokens ?? '—'}/${qu.total_tokens ?? '—'}` : (r.qwen_called ? '—' : '스킵');
      const img = r.input_image ? `${r.input_image.width}×${r.input_image.height}` : '';
      const qra = r.qwen_resources_after;
      const g2 = qra && qra.gpu_memory_used_mb;
      const u2 = qra && qra.gpu_utilization_percent;
      const rs2 = qra && qra.python_rss_mb;
      const rep = (r.qwen_reply || '') || '';
      html += `<tr><td>${esc(r.frame)}</td><td class="num">${r.qwen_called ? r.qwen_seconds : '—'}</td><td class="num">${qtok}</td><td>${esc(img)}</td><td class="num">${g2 != null ? g2 : '—'} / ${u2 != null ? u2 : '—'}</td><td class="num">${rs2 ?? '—'}</td><td class="reply">${esc(rep)}</td></tr>`;
    }
    html += '</tbody></table>';
  }
  if (!html) html = '<p>상세할 행이 없습니다.</p>';
  document.getElementById('tables').innerHTML = html;
  </script>
</body>
</html>
"""
