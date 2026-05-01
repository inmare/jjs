"""HR-Bench 비교 HTML·PNG 차트."""
from __future__ import annotations

import html as html_mod
import json
import math
from pathlib import Path
from typing import Any


def _chart_float(v: Any) -> float | None:
    """Chart.js·브라우저 ``JSON.parse`` 용: 유한 실수만. (Python ``NaN`` 은 JSON 비표준이라 금지.)"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _chart_payload(strategy_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """전략 요약 리스트에서 Chart.js용 레이블·시리즈 값을 뽑는다. None 은 JSON null."""
    names = [str(s.get("strategy", "")) for s in strategy_summaries]

    def col(key: str) -> list[float | None]:
        out: list[float | None] = []
        for s in strategy_summaries:
            out.append(_chart_float(s.get(key)))
        return out

    return {
        "labels": names,
        "accuracy": col("accuracy"),
        "mean_seconds": col("mean_seconds"),
        "mean_prompt_tokens": col("mean_prompt_tokens"),
        "mean_stage1_prompt_tokens": col("mean_stage1_prompt_tokens"),
        "max_gpu_memory_used_mb": col("max_gpu_memory_used_mb"),
    }


def _hr_config_table_html(cfg: dict[str, Any] | None) -> str:
    """JSON에 실린 ``config`` 를 읽기 쉬운 표로."""
    if not cfg:
        return ""
    rows: list[str] = []
    for key in sorted(cfg.keys(), key=str):
        v = cfg[key]
        rows.append(
            "<tr>"
            f"<td>{html_mod.escape(str(key))}</td>"
            f"<td>{html_mod.escape(str(v))}</td>"
            "</tr>"
        )
    return (
        "<h2>실행 설정 (YOLO·전망)</h2>\n"
        "<table><thead><tr><th>설정</th><th>값</th></tr></thead><tbody>\n"
        + "\n".join(rows)
        + "\n</tbody></table>\n"
    )


def _hr_compare_chart_section(strategy_summaries: list[dict[str, Any]]) -> str:
    """요약 표 아래에 붙일 Chart.js(막대) 2×2 블록 HTML."""
    if not strategy_summaries:
        return ""
    payload = _chart_payload(strategy_summaries)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    safe = raw.replace("</", "<\\/")
    return f"""
  <h2>비교 차트 (HTML)</h2>
  <p class="meta">PNG 저장 없이 동일 지표를 브라우저에서 볼 수 있습니다.</p>
  <div class="chart-grid">
    <div class="chart-cell"><canvas id="hrChartAcc"></canvas></div>
    <div class="chart-cell"><canvas id="hrChartSec"></canvas></div>
    <div class="chart-cell"><canvas id="hrChartTok"></canvas></div>
    <div class="chart-cell"><canvas id="hrChartGpu"></canvas></div>
  </div>
  <script type="application/json" id="hr-chart-payload">{safe}</script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
  (function() {{
    const el = document.getElementById('hr-chart-payload');
    if (!el || typeof Chart === 'undefined') return;
    const D = JSON.parse(el.textContent);
    const labels = D.labels || [];
    const nz = (arr) => (arr || []).map(v => (v === null || v === undefined) ? 0 : v);
    function opts(title) {{
      return {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: title }} }},
        scales: {{
          x: {{ ticks: {{ maxRotation: 40, minRotation: 20 }} }},
          y: {{ beginAtZero: true }}
        }}
      }};
    }}
    new Chart(document.getElementById('hrChartAcc'), {{
      type: 'bar',
      data: {{ labels, datasets: [{{ label: 'accuracy', data: nz(D.accuracy), backgroundColor: '#4682b4' }}] }},
      options: opts('Accuracy')
    }});
    new Chart(document.getElementById('hrChartSec'), {{
      type: 'bar',
      data: {{ labels, datasets: [{{ label: 's', data: nz(D.mean_seconds), backgroundColor: '#ff8c00' }}] }},
      options: opts('Mean seconds / sample')
    }});
    new Chart(document.getElementById('hrChartTok'), {{
      type: 'bar',
      data: {{ labels, datasets: [{{ label: 'tok', data: nz(D.mean_prompt_tokens), backgroundColor: '#2e8b57' }}] }},
      options: opts('Mean Qwen prompt tokens')
    }});
    new Chart(document.getElementById('hrChartGpu'), {{
      type: 'bar',
      data: {{ labels, datasets: [{{ label: 'MB', data: nz(D.max_gpu_memory_used_mb), backgroundColor: '#9370db' }}] }},
      options: opts('Max GPU memory (MB)')
    }});
  }})();
  </script>
"""


def write_hr_compare_html(
    path: Path,
    *,
    summary: dict[str, Any],
    strategy_summaries: list[dict[str, Any]],
    per_strategy_rows: dict[str, list[dict[str, Any]]],
    include_sample_tables: bool = True,
    include_charts: bool = True,
) -> None:
    """
    HR-Bench 전략 비교 HTML 을 ``path`` 에 기록한다.

    요약 표 아래에 Chart.js(CDN) 막대 그래프 4종(정확도·시간·토큰·GPU)을 넣고,
    ``include_sample_tables`` 가 참이면 전략별 샘플 표를 덧붙인다.
    """
    acc_rows = []
    for s in strategy_summaries:
        acc = s.get("accuracy")
        acc_s = "—" if acc is None else str(acc)
        mpt = s.get("mean_prompt_tokens")
        mpt_s = "—" if mpt is None else str(mpt)
        ms1 = s.get("mean_stage1_prompt_tokens")
        ms1_s = "—" if ms1 is None else str(ms1)
        msg = s.get("max_gpu_memory_used_mb")
        msg_s = "—" if msg is None else str(msg)
        msec = s.get("mean_seconds")
        msec_s = "—" if msec is None else str(msec)
        acc_rows.append(
            "<tr>"
            f"<td>{html_mod.escape(str(s.get('strategy', '')))}</td>"
            f"<td class='num'>{s.get('evaluated', '—')}</td>"
            f"<td class='num'>{s.get('correct', '—')}</td>"
            f"<td class='num'>{acc_s}</td>"
            f"<td class='num'>{msec_s}</td>"
            f"<td class='num'>{mpt_s}</td>"
            f"<td class='num'>{ms1_s}</td>"
            f"<td class='num'>{msg_s}</td>"
            "</tr>"
        )
    meta_parts = [
        f"데이터셋: {html_mod.escape(str(summary.get('hr_bench', '')))}",
        f"split: {html_mod.escape(str(summary.get('split', '')))}",
        f"샘플 모드: {html_mod.escape(str(summary.get('sample_mode', '')))}",
        f"max_samples={summary.get('max_samples', '—')}",
        f"model(Qwen): {html_mod.escape(str(summary.get('model', '')))}",
    ]
    if summary.get("model_small"):
        meta_parts.append(
            f"model(Smol): {html_mod.escape(str(summary.get('model_small', '')))}"
        )
    body_extra = ""
    chart_block = (
        _hr_compare_chart_section(strategy_summaries) if include_charts else ""
    )
    config_block = _hr_config_table_html(
        summary.get("config") if isinstance(summary.get("config"), dict) else None
    )

    if include_sample_tables:
        blocks = []
        for strat, rows in per_strategy_rows.items():
            sub = []
            for r in rows[:500]:
                raw = html_mod.escape((r.get("raw_reply") or "")[:400])
                ok = r.get("correct")
                mark = "✓" if ok else "✗"
                sub.append(
                    "<tr>"
                    f"<td class='num'>{r.get('dataset_row_index', '—')}</td>"
                    f"<td class='num'>{r.get('index', '')}</td>"
                    f"<td>{html_mod.escape(str(r.get('ground_truth_letter', '')))}</td>"
                    f"<td>{html_mod.escape(str(r.get('pred_letter', '')))}</td>"
                    f"<td>{mark}</td>"
                    f"<td class='num'>{r.get('seconds', '')}</td>"
                    f"<td class='num'>{r.get('prompt_tokens', '')}</td>"
                    f"<td class='reply'>{raw}</td>"
                    "</tr>"
                )
            blocks.append(
                f"<h2>{html_mod.escape(strat)}</h2>\n"
                "<table><thead><tr>"
                "<th>ds#</th><th>idx</th><th>정답</th><th>예측</th><th>O/X</th>"
                "<th>초</th><th>prompt_tok</th><th>응답(앞)</th>"
                "</tr></thead><tbody>\n"
                + "\n".join(sub)
                + "</tbody></table>"
            )
        body_extra = "\n".join(blocks)

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>HR-Bench 전략 비교</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.2rem; background: #fafafa; color: #222; }}
    h1 {{ font-size: 1.2rem; }}
    h2 {{ font-size: 1rem; margin-top: 1.5rem; }}
    .meta {{ font-size: 0.9rem; color: #444; margin: 0.8rem 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin-bottom: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; vertical-align: top; }}
    th {{ background: #eee; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.reply {{ white-space: pre-wrap; max-width: 24rem; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; max-width: 56rem; }}
    .chart-cell {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.5rem; min-height: 14rem; }}
  </style>
</head>
<body>
  <h1>HR-Bench — 전략 비교</h1>
  <p class="meta">{" · ".join(meta_parts)}</p>
  {config_block}
  <h2>요약 표</h2>
  <table>
    <thead>
      <tr>
        <th>전략</th><th>평가수</th><th>정답수</th><th>정확도</th>
        <th>평균초</th><th>평균Qwen prompt tok</th><th>평균Smol prompt tok</th><th>최대 GPU MB</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(acc_rows)}
    </tbody>
  </table>
  {chart_block}
  {body_extra}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def write_hr_compare_charts_png(
    path: Path,
    strategy_summaries: list[dict[str, Any]],
) -> None:
    """matplotlib 으로 2×2 막대 그래프 PNG 를 저장한다 (--group dev)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "PNG 차트는 matplotlib 이 필요합니다. uv sync --group dev"
        ) from e

    names = [str(s.get("strategy", "?")) for s in strategy_summaries]

    def nzf(key: str) -> list[float]:
        return [_chart_float(s.get(key)) or 0.0 for s in strategy_summaries]

    acc = nzf("accuracy")
    secs = nzf("mean_seconds")
    toks = nzf("mean_prompt_tokens")
    gpus = nzf("max_gpu_memory_used_mb")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = range(len(names))
    axes[0, 0].bar(x, acc, color="steelblue")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(names, rotation=25, ha="right")

    axes[0, 1].bar(x, secs, color="darkorange")
    axes[0, 1].set_title("Mean seconds / sample")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(names, rotation=25, ha="right")

    axes[1, 0].bar(x, toks, color="seagreen")
    axes[1, 0].set_title("Mean Qwen prompt tokens")
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(names, rotation=25, ha="right")

    axes[1, 1].bar(x, gpus, color="mediumpurple")
    axes[1, 1].set_title("Max GPU memory used (MB)")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(names, rotation=25, ha="right")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
