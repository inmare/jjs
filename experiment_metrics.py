"""실험용: OpenAI usage 직렬화, nvidia-smi / 프로세스 메모리 스냅샷, HTML 리포트."""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def usage_to_dict(usage: object | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    d: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, name, None)
        if v is not None:
            d[name] = int(v)
    return d or None


def resource_snapshot(*, label: str = "") -> dict[str, Any]:
    """GPU는 nvidia-smi 기준(카드 전체 사용량). Python 프로세스 RSS는 psutil 있을 때만."""
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
                out["gpu_memory_used_mb"] = float(parts[0])
                out["gpu_memory_total_mb"] = float(parts[1])
                out["gpu_utilization_percent"] = float(parts[2])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass

    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process()
        out["python_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    return out


def write_experiment_html_report(json_path: Path, html_path: Path) -> None:
    """JSON 결과를 읽어 Chart.js 막대 그래프 + 표가 있는 단일 HTML 생성."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    b64 = base64.standard_b64encode(raw).decode("ascii")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        EXPERIMENT_HTML_TEMPLATE.replace("__B64_PAYLOAD__", b64),
        encoding="utf-8",
    )


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
    .note { font-size: 0.85rem; color: #555; max-width: 52rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem 1rem; }
    table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
    th, td { border: 1px solid #ddd; padding: 0.35rem 0.5rem; vertical-align: top; }
    th { background: #f0f0f0; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .reply { white-space: pre-wrap; max-height: 12rem; overflow: auto; font-size: 0.78rem; }
    canvas { max-height: 280px; }
  </style>
</head>
<body>
  <h1>VLM 실험 리포트</h1>
  <p class="note" id="meta"></p>
  <p class="note">GPU 메모리·사용률은 <code>nvidia-smi</code>의 <strong>GPU 카드 전체</strong> 값입니다. llama-server가 대부분 점유하지만, 다른 프로세스가 있으면 합산됩니다. Python <code>python_rss_mb</code>는 실험 스크립트 프로세스만 해당합니다.</p>

  <div class="grid">
    <div class="card"><h2>응답 시간 (초)</h2><canvas id="chartTime"></canvas></div>
    <div class="card"><h2>prompt_tokens (서버 보고)</h2><canvas id="chartPromptTok"></canvas></div>
    <div class="card"><h2>추론 직후 GPU 사용량 (MiB)</h2><canvas id="chartGpu"></canvas></div>
  </div>

  <h2>프레임별 상세</h2>
  <div id="tables"></div>

  <script>
  function decodeB64Utf8(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return JSON.parse(new TextDecoder('utf-8').decode(bytes));
  }
  const P = decodeB64Utf8('__B64_PAYLOAD__');
  document.getElementById('meta').textContent =
    `생성: ${P.generated_at || ''} · 프레임 ${P.frame_count || 0}장 · ${P.frames_dir || ''}`;

  const bench = P.bench || [];
  const labels = bench.map(r => r.frame || '');
  const qOnlyS = bench.map(r => (r.qwen_only && r.qwen_only.seconds) ?? null);
  const yoloS = bench.map(r => (r.yolo_qwen && r.yolo_qwen.total_seconds) ?? null);
  const two = P.two_stage_skip_low || [];
  const twoTotal = two.map(r => r.total_seconds ?? null);

  const qTok = bench.map(r => (r.qwen_only && r.qwen_only.prompt_tokens) ?? null);
  const yTok = bench.map(r => (r.yolo_qwen && r.yolo_qwen.prompt_tokens) ?? null);
  const smolTok = two.map(r => (r.stage1_usage && r.stage1_usage.prompt_tokens) ?? null);
  const qwen2Tok = two.map(r => r.qwen_prompt_tokens ?? null);

  const gpuQ = bench.map(r => (r.qwen_only && r.qwen_only.resources_after && r.qwen_only.resources_after.gpu_memory_used_mb) ?? null);
  const gpuY = bench.map(r => (r.yolo_qwen && r.yolo_qwen.resources_after && r.yolo_qwen.resources_after.gpu_memory_used_mb) ?? null);
  const gpuSmol = two.map(r => (r.stage1_resources_after && r.stage1_resources_after.gpu_memory_used_mb) ?? null);
  const gpuQ2 = two.map(r => (r.qwen_resources_after && r.qwen_resources_after.gpu_memory_used_mb) ?? null);

  if (labels.length) {
    new Chart(document.getElementById('chartTime'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Qwen-only', data: qOnlyS, backgroundColor: 'rgba(54,162,235,0.7)' },
          { label: 'YOLO+Qwen', data: yoloS, backgroundColor: 'rgba(255,159,64,0.7)' },
          { label: 'Smol+Qwen 합산', data: twoTotal, backgroundColor: 'rgba(75,192,192,0.7)' },
        ]
      },
      options: { responsive: true, scales: { x: { stacked: false } } }
    });
    new Chart(document.getElementById('chartPromptTok'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Q-only p.tok', data: qTok, backgroundColor: 'rgba(54,162,235,0.7)' },
          { label: 'YOLO+Q p.tok', data: yTok, backgroundColor: 'rgba(255,159,64,0.7)' },
          { label: 'Smol p.tok', data: smolTok, backgroundColor: 'rgba(153,102,255,0.7)' },
          { label: 'Qwen2 p.tok', data: qwen2Tok, backgroundColor: 'rgba(75,192,192,0.7)' },
        ]
      },
      options: { responsive: true }
    });
    new Chart(document.getElementById('chartGpu'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Q-only 후 GPU MiB', data: gpuQ, backgroundColor: 'rgba(54,162,235,0.5)' },
          { label: 'YOLO+Q 후', data: gpuY, backgroundColor: 'rgba(255,159,64,0.5)' },
          { label: 'Smol 후', data: gpuSmol, backgroundColor: 'rgba(153,102,255,0.5)' },
          { label: 'Qwen2 후', data: gpuQ2, backgroundColor: 'rgba(75,192,192,0.5)' },
        ]
      },
      options: { responsive: true }
    });
  }

  let html = '<table><thead><tr><th>프레임</th><th>시나리오</th><th>시간(s)</th><th>p/c/t tok</th><th>이미지</th><th>GPU 후 MiB / util%</th><th>Python RSS MiB</th><th>응답</th></tr></thead><tbody>';
  for (const r of bench) {
    const img = r.images && r.images.qwen_context ? `${r.images.qwen_context.width}×${r.images.qwen_context.height}` : '';
    const uq = r.qwen_only && r.qwen_only.usage;
    const tok = uq ? `${uq.prompt_tokens ?? '—'}/${uq.completion_tokens ?? '—'}/${uq.total_tokens ?? '—'}` : '';
    const ra = r.qwen_only && r.qwen_only.resources_after;
    const gpu = ra && ra.gpu_memory_used_mb;
    const util = ra && ra.gpu_utilization_percent;
    const rss = ra && ra.python_rss_mb;
    html += `<tr><td>${r.frame}</td><td>Qwen-only</td><td class="num">${r.qwen_only && r.qwen_only.seconds}</td><td class="num">${tok}</td><td>${img}</td><td class="num">${gpu != null ? gpu : '—'} / ${util != null ? util : '—'}</td><td class="num">${rss ?? '—'}</td><td class="reply">${(r.qwen_only && r.qwen_only.reply) || ''}</td></tr>`;
    const uy = r.yolo_qwen && r.yolo_qwen.usage;
    const toky = uy ? `${uy.prompt_tokens ?? '—'}/${uy.completion_tokens ?? '—'}/${uy.total_tokens ?? '—'}` : '';
    const imgsY = r.images && r.images.yolo_qwen ? `${r.images.yolo_qwen.n_images_sent}장 · ctx ${r.images.yolo_qwen.context_width}×${r.images.yolo_qwen.context_height}` : '';
    const ray = r.yolo_qwen && r.yolo_qwen.resources_after;
    const gpuY = ray && ray.gpu_memory_used_mb;
    const utilY = ray && ray.gpu_utilization_percent;
    const rssY = ray && ray.python_rss_mb;
    html += `<tr><td>${r.frame}</td><td>YOLO+Qwen</td><td class="num">${r.yolo_qwen && r.yolo_qwen.total_seconds}</td><td class="num">${toky}</td><td>${imgsY}</td><td class="num">${gpuY != null ? gpuY : '—'} / ${utilY != null ? utilY : '—'}</td><td class="num">${rssY ?? '—'}</td><td class="reply">${(r.yolo_qwen && r.yolo_qwen.reply) || ''}</td></tr>`;
  }
  for (const r of two) {
    const su = r.stage1_usage;
    const stok = su ? `${su.prompt_tokens ?? '—'}/${su.completion_tokens ?? '—'}/${su.total_tokens ?? '—'}` : '—';
    const qu = r.qwen_usage;
    const qtok = qu ? `${qu.prompt_tokens ?? '—'}/${qu.completion_tokens ?? '—'}/${qu.total_tokens ?? '—'}` : (r.qwen_called ? '—' : '스킵');
    const img = r.input_image ? `${r.input_image.width}×${r.input_image.height}` : '';
    const sra = r.stage1_resources_after;
    const g1 = sra && sra.gpu_memory_used_mb;
    const u1 = sra && sra.gpu_utilization_percent;
    const rs1 = sra && sra.python_rss_mb;
    const qra = r.qwen_resources_after;
    const g2 = qra && qra.gpu_memory_used_mb;
    const u2 = qra && qra.gpu_utilization_percent;
    const rs2 = qra && qra.python_rss_mb;
    html += `<tr><td>${r.frame}</td><td>Smol 1단계</td><td class="num">${r.stage1_seconds}</td><td class="num">${stok}</td><td>${img}</td><td class="num">${g1 != null ? g1 : '—'} / ${u1 != null ? u1 : '—'}</td><td class="num">${rs1 ?? '—'}</td><td class="reply">${(r.stage1_text || '').slice(0,4000)}</td></tr>`;
    html += `<tr><td>${r.frame}</td><td>Qwen 2단계</td><td class="num">${r.qwen_called ? r.qwen_seconds : '—'}</td><td class="num">${qtok}</td><td>${img}</td><td class="num">${g2 != null ? g2 : '—'} / ${u2 != null ? u2 : '—'}</td><td class="num">${rs2 ?? '—'}</td><td class="reply">${(r.qwen_reply || '')}</td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('tables').innerHTML = html;
  </script>
</body>
</html>
"""
