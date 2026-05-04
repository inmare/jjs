"""HR-Bench 비교 HTML·PNG 차트."""
from __future__ import annotations

import base64
import copy
import html as html_mod
import json
import math
import re
import shutil
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


def _build_hr_problem_browser_payload(
    per_strategy_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """문제 선택 UI용 데이터: dataset 행별로 전략별 요약과 ``hr_debug``."""
    if not per_strategy_rows:
        return None
    by_ds: dict[int, dict[str, Any]] = {}
    for strat, rows in per_strategy_rows.items():
        for r in rows:
            ds_val = r.get("dataset_row_index")
            if ds_val is None:
                continue
            try:
                ds = int(ds_val)
            except (TypeError, ValueError):
                continue
            blk = by_ds.setdefault(
                ds,
                {
                    "dataset_row_index": ds,
                    "hr_index": r.get("index"),
                    "ground_truth_letter": r.get("ground_truth_letter"),
                    "strategies": {},
                },
            )
            blk["strategies"][str(strat)] = {
                "strategy": strat,
                "correct": r.get("correct"),
                "pred_letter": r.get("pred_letter"),
                "raw_reply": r.get("raw_reply"),
                "prompt_tokens": r.get("prompt_tokens"),
                "stage1_prompt_tokens": r.get("stage1_prompt_tokens"),
                "n_crops_sent": r.get("n_crops_sent"),
                "n_detections": r.get("n_detections"),
                "parsed_risk": r.get("parsed_risk"),
                "smol_ok": r.get("smol_ok"),
                "yolo_seconds": r.get("yolo_seconds"),
                "smol_seconds": r.get("smol_seconds"),
                "seconds": r.get("seconds"),
                "hr_debug": r.get("hr_debug"),
            }
    ids = sorted(by_ds.keys())
    if not ids:
        return None
    return {
        "ordered_ids": ids,
        "by_dataset_row": {str(k): by_ds[k] for k in ids},
        "strategy_order": sorted(per_strategy_rows.keys()),
    }


def _jpeg_bytes_from_data_url(data_url: str) -> bytes | None:
    if not isinstance(data_url, str) or not data_url.startswith("data:") or "base64," not in data_url:
        return None
    b64 = data_url.split("base64,", 1)[1].strip()
    try:
        return base64.standard_b64decode(b64)
    except Exception:
        return None


def _sanitize_problem_asset_segment(seg: str) -> str:
    s = re.sub(r"[^\w\-.()+]", "_", str(seg)).strip("._")
    return (s[:120] if s else "x")


def _materialized_asset_href_under_html(asset_file: Path, html_dir: Path) -> str:
    """HTML 파일과 같은 루트를 기준으로 한 상대 href(항상 `./` 로 시작해 file·http 로 열어도 해석 안정화)."""
    ap = asset_file.resolve()
    hd = html_dir.resolve()
    try:
        rel = ap.relative_to(hd).as_posix()
        if rel == ".":
            return "./"
        if rel.startswith((".", "/")):
            return rel
        return "./" + rel
    except ValueError:
        return ap.resolve().as_uri()


def inline_hr_compare_materialized_images_for_srcdoc(html: str, html_path: Path) -> str:
    """``iframe.srcdoc``·일부 WebView 에서 로컬 ``./stem_img/*.jpg``·``file://`` 로드가 막힐 때 data URL 로 치환.

    디스크에 저장된 ``html_path`` 파일은 건드리지 않고, 미리보기용 문자열에만 적용한다.
    """
    root = html_path.parent / f"{html_path.stem}_img"
    if not root.is_dir():
        return html
    out = html
    for fp in sorted(root.rglob("*.jpg"), key=lambda p: len(str(p)), reverse=True):
        try:
            blob = fp.read_bytes()
        except OSError:
            continue
        du = "data:image/jpeg;base64," + base64.standard_b64encode(blob).decode("ascii")
        rel = "./" + fp.relative_to(html_path.parent).as_posix()
        out = out.replace(json.dumps(rel), json.dumps(du))
        uri = fp.resolve().as_uri()
        out = out.replace(json.dumps(uri), json.dumps(du))
    return out


def _materialize_hr_problem_payload_assets(
    payload_obj: dict[str, Any],
    html_path: Path,
) -> None:
    """인라인 data URL 디버깅 이미지를 HTML 옆 ``{stem}_img/`` 디렉터리에 저장하고 경로 문자열로 치환."""
    html_dir = html_path.parent
    stem = html_path.stem or "hr_compare"
    root = html_dir / f"{stem}_img"
    root.mkdir(parents=True, exist_ok=True)
    by = payload_obj.get("by_dataset_row")
    if not isinstance(by, dict):
        return

    for ds_str, blk in by.items():
        if not isinstance(blk, dict):
            continue
        strategies = blk.get("strategies")
        if not isinstance(strategies, dict):
            continue
        for strat_name, srec in strategies.items():
            if not isinstance(srec, dict):
                continue
            dbg = srec.get("hr_debug")
            if not isinstance(dbg, dict):
                continue
            ss = _sanitize_problem_asset_segment(strat_name)
            sub = root / f"ds{ds_str}_{ss}"
            sub.mkdir(parents=True, exist_ok=True)

            for key, fname in (
                ("thumbnail_original_jpeg_url", "thumbnail_original.jpg"),
                ("overview_thumbnail_url", "overview_thumbnail.jpg"),
                ("qwen_sent_thumbnail_url", "qwen_sent_thumbnail.jpg"),
            ):
                val = dbg.get(key)
                if not isinstance(val, str) or not val.startswith("data:"):
                    continue
                blob = _jpeg_bytes_from_data_url(val)
                if not blob:
                    continue
                out = sub / fname
                out.write_bytes(blob)
                dbg[key] = _materialized_asset_href_under_html(out, html_dir)

            crops_val = dbg.get("crop_thumbnails_jpeg_urls")
            if isinstance(crops_val, list):
                replaced: list[str | Any] = []
                for i, c in enumerate(crops_val):
                    if isinstance(c, str) and c.startswith("data:"):
                        blob = _jpeg_bytes_from_data_url(c)
                        if blob:
                            out = sub / f"crop_{i + 1:02d}.jpg"
                            out.write_bytes(blob)
                            replaced.append(_materialized_asset_href_under_html(out, html_dir))
                        else:
                            replaced.append(c)
                    else:
                        replaced.append(c)
                dbg["crop_thumbnails_jpeg_urls"] = replaced


def _hr_problem_detail_browser_section_from_payload(
    payload_obj: dict[str, Any] | None,
) -> str:
    """문제 디버깅 패널(Chart 하단). ``payload_obj`` 는 ``_build_hr_problem_browser_payload`` 결과."""
    if not payload_obj:
        return ""
    raw = json.dumps(payload_obj, ensure_ascii=False, allow_nan=False)
    safe = raw.replace("</", "<\\/")
    _js = r"""
(function() {
  const el = document.getElementById('hr-problem-payload');
  const picks = document.getElementById('hr-problem-picks');
  const wrap = document.getElementById('hr-problem-panel-wrap');
  const panel = document.getElementById('hr-problem-panel');
  if (!el || !picks || !panel || !wrap) return;
  let PAYLOAD;
  try { PAYLOAD = JSON.parse(el.textContent); } catch (e) { return; }
  const by = PAYLOAD.by_dataset_row || {};
  const ids = PAYLOAD.ordered_ids || [];
  const stratOrder = PAYLOAD.strategy_order || [];
  function esc(s) {
    if (s === null || s === undefined) return '';
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML.replace(/\n/g, '<br>');
  }
  function escAttr(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;');
  }
  function normalizeImgSrc(href) {
    var s = String(href || '').trim();
    if (!s) return '';
    if (/^(data:|https?:|file:)/i.test(s)) return s;
    try {
      var b = document.baseURI || window.location.href;
      return new URL(s, b).href;
    } catch (err) {
      return s;
    }
  }
  function imgSrcEscaped(href) {
    return escAttr(normalizeImgSrc(href));
  }
  function renderImgs(debug) {
    if (!debug) return '';
    let html = '';
    if (debug.thumbnail_original_jpeg_url) {
      html += '<div class="hr-dbg-block"><strong>원본(썸네일)</strong><br/><img src="'
        + imgSrcEscaped(debug.thumbnail_original_jpeg_url) + '" class="hr-dbg-thumb"/></div>';
    }
    if (debug.overview_thumbnail_url) {
      var srcOv = imgSrcEscaped(debug.overview_thumbnail_url);
      var hasBox = Array.isArray(debug.yolo_bbox_xyxy_overview)
        && debug.yolo_bbox_xyxy_overview.length > 0
        && Array.isArray(debug.overview_sent_size)
        && debug.overview_sent_size.length >= 2;
      if (hasBox) {
        try {
          var ove = encodeURIComponent(JSON.stringify(debug.overview_sent_size));
          var bxe = encodeURIComponent(JSON.stringify(debug.yolo_bbox_xyxy_overview));
          html += '<div class="hr-dbg-block"><strong>VLM 전망 + 탐지 박스</strong>'
            + '<span class="hr-bbox-hint"> (전망 픽셀 좌표, 썸네일 축소에 맞춤)</span><br/>';
          html += '<div class="hr-bbox-wrap" data-ovs-e="' + escAttr(ove) + '" data-boxes-e="'
            + escAttr(bxe) + '">';
          html += '<img alt="overview" src="' + srcOv + '" class="hr-dbg-thumb hr-ov-underlay"/>';
          html += '</div></div>';
        } catch (err) {
          html += '<div class="hr-dbg-block"><strong>VLM 전망 썸네일</strong><br/><img src="'
            + srcOv + '" class="hr-dbg-thumb"/></div>';
        }
      } else {
        html += '<div class="hr-dbg-block"><strong>VLM 전망 썸네일</strong><br/><img src="'
          + srcOv + '" class="hr-dbg-thumb"/></div>';
      }
    }
    var crops = debug.crop_thumbnails_jpeg_urls || [];
    if (crops.length) {
      html += '<div class="hr-dbg-block"><strong>크롭 ' + crops.length + '개</strong><div class="hr-crops">';
      for (var i = 0; i < crops.length; i++) {
        html += '<figure><figcaption>크롭 ' + (i + 1) + '</figcaption><img src="' + imgSrcEscaped(crops[i])
          + '" class="hr-dbg-thumb"/></figure>';
      }
      html += '</div></div>';
    }
    if (debug.qwen_sent_thumbnail_url) {
      html += '<div class="hr-dbg-block"><strong>Qwen에 전송한 이미지 썸네일</strong><br/><img src="'
        + imgSrcEscaped(debug.qwen_sent_thumbnail_url) + '" class="hr-dbg-thumb"/></div>';
    }
    return html;
  }
  function initOverviewBBoxes(panel) {
    if (!panel || !panel.querySelectorAll) return;
    panel.querySelectorAll('.hr-bbox-wrap').forEach(function(wrap) {
      var oRaw = wrap.getAttribute('data-ovs-e');
      var bRaw = wrap.getAttribute('data-boxes-e');
      if (!oRaw || !bRaw) return;
      var ovs, boxes;
      try {
        ovs = JSON.parse(decodeURIComponent(oRaw));
        boxes = JSON.parse(decodeURIComponent(bRaw));
      } catch (e) { return; }
      if (!boxes || boxes.length === 0 || !ovs || ovs.length < 2) return;
      var ovW = ovs[0], ovH = ovs[1];
      var img = wrap.querySelector('img');
      if (!img) return;
      function draw() {
        var nw = img.naturalWidth;
        var nh = img.naturalHeight;
        if (!nw || !nh || !ovW || !ovH) return;
        var sx = nw / ovW;
        var sy = nh / ovH;
        var svgNS = 'http://www.w3.org/2000/svg';
        var prev = wrap.querySelector('svg.hr-bbox-layer');
        if (prev) prev.remove();
        var svg = document.createElementNS(svgNS, 'svg');
        svg.setAttribute('class', 'hr-bbox-layer');
        svg.setAttribute('width', String(nw));
        svg.setAttribute('height', String(nh));
        svg.style.position = 'absolute';
        svg.style.left = '0';
        svg.style.top = '0';
        svg.style.pointerEvents = 'none';
        boxes.forEach(function(b) {
          if (!b || b.length < 4) return;
          var r = document.createElementNS(svgNS, 'rect');
          var x1 = Number(b[0]) * sx, y1 = Number(b[1]) * sy;
          var x2 = Number(b[2]) * sx, y2 = Number(b[3]) * sy;
          var rx1 = Math.min(x1, x2), ry1 = Math.min(y1, y2);
          var rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1);
          r.setAttribute('x', String(rx1));
          r.setAttribute('y', String(ry1));
          r.setAttribute('width', String(rw));
          r.setAttribute('height', String(rh));
          r.setAttribute('fill', 'rgba(233,30,99,0.06)');
          r.setAttribute('stroke', '#e91e63');
          r.setAttribute('stroke-width', '2');
          svg.appendChild(r);
        });
        wrap.style.position = 'relative';
        wrap.style.display = 'inline-block';
        wrap.style.maxWidth = '100%';
        img.style.display = 'block';
        img.style.maxWidth = '100%';
        img.style.height = 'auto';
        wrap.appendChild(svg);
      }
      if (img.complete && img.naturalWidth) draw();
      else img.addEventListener('load', draw);
    });
  }
  function renderDebugText(debug) {
    if (!debug) return '';
    var t = '';
    if (debug.smol_prompt)
      t += '<h4>Smol 프롬프트(텍스트)</h4><pre class="hr-pre">' + esc(debug.smol_prompt) + '</pre>';
    var smolRaw = debug.smol_reply_text != null && String(debug.smol_reply_text).length
      ? String(debug.smol_reply_text)
      : (debug.smol_reply_preview || '');
    if (smolRaw)
      t += '<h4>Smol 원문 답변(전체)</h4><pre class="hr-pre">' + esc(smolRaw) + '</pre>';
    if (debug.compact_vision_sentence_en)
      t += '<h4>Qwen으로 보낸 Smol 영어 요약</h4><pre class="hr-pre">' + esc(debug.compact_vision_sentence_en) + '</pre>';
    else if (debug.stage1_merged_block)
      t += '<h4>레거시 1단계 블록</h4><pre class="hr-pre">' + esc(debug.stage1_merged_block) + '</pre>';
    if (debug.yolo_context_sentence_en)
      t += '<h4>YOLO 영어 요약 문장</h4><pre class="hr-pre">' + esc(debug.yolo_context_sentence_en) + '</pre>';
    else if (debug.coordinate_preamble)
      t += '<h4>YOLO 좌표 서문(레거시)</h4><pre class="hr-pre">' + esc(debug.coordinate_preamble) + '</pre>';
    if (debug.qwen_prompt)
      t += '<h4>Qwen 사용자 프롬프트 전체</h4><pre class="hr-pre">' + esc(debug.qwen_prompt) + '</pre>';
    if (debug.yolo_bbox_xyxy_on_canvas)
      t += '<h4>YOLO bbox (탐지 캔버스 xyxy)</h4><pre class="hr-pre">'
        + esc(JSON.stringify(debug.yolo_bbox_xyxy_on_canvas, null, 2)) + '</pre>';
    if (debug.yolo_canvas_size_px)
      t += '<p><strong>YOLO 캔버스:</strong> ' + esc(JSON.stringify(debug.yolo_canvas_size_px)) + '</p>';
    if (debug.parsed_risk)
      t += '<p><strong>parsed risk:</strong> ' + esc(debug.parsed_risk) + '</p>';
    if (debug.qwen_sent_image_sizes)
      t += '<p><strong>Qwen 입력 크기(px):</strong> ' + esc(JSON.stringify(debug.qwen_sent_image_sizes)) + '</p>';
    return t;
  }
  function renderOne(dsKey) {
    const P = by[String(dsKey)];
    if (!P) return;
    var html = '<div class="hr-problem-detail">';
    html += '<h3>행 ' + esc(P.dataset_row_index) + ' · HR 인덱스 ' + esc(P.hr_index)
      + ' · 정답 <strong>' + esc(P.ground_truth_letter) + '</strong></h3>';
    stratOrder.forEach(function(stratName) {
      const S = (P.strategies || {})[stratName];
      if (!S) return;
      var ok = S.correct ? ' ✓' : ' ✗';
      html += '<section class="hr-strategy-block"><h4>' + esc(stratName) + ok + '</h4>';
      html += '<p>예측: <code>' + esc(S.pred_letter) + '</code> · tok ' + esc(S.prompt_tokens) + '</p>';
      html += '<p class="mono-tight">모델 답(앞 300자): <code>'
        + esc((S.raw_reply || '').substring(0, 300)) + '</code></p>';
      html += renderImgs(S.hr_debug);
      html += renderDebugText(S.hr_debug);
      html += '</section>';
    });
    html += '</div>';
    panel.innerHTML = html;
    wrap.hidden = false;
    initOverviewBBoxes(panel);
  }
  ids.forEach(function(did) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'hr-btn-pick';
    const P = by[String(did)];
    btn.textContent = '#' + did + ' (HR ' + (P && P.hr_index !== undefined ? P.hr_index : '?') + ')';
    btn.addEventListener('click', function() {
      picks.querySelectorAll('button').forEach(function(b) { b.setAttribute('aria-pressed', 'false'); });
      btn.setAttribute('aria-pressed', 'true');
      renderOne(did);
    });
    picks.appendChild(btn);
  });
})();
"""
    return f"""
<h2 id="hr-problems">문제별 정보 더 보기</h2>
<p class="meta">데이터셋 행 번호를 클릭하면 해당 샘플만 아래 패널에 표시합니다. 디버깅 이미지는 HTML과 같은 폴더의
<code>&lt;stem&gt;_img/</code> 에 저장되고, 패널에는 상대 경로(<code>./stem_img/...</code>)가 들어갑니다.
페이지 헤더의 <code>&lt;base href&gt;</code> 와 브라우저가 URL 을 재해석해 <code>file://</code> 직접 열기·로컬 HTTP 서버 둘 다에서
로드하기 쉽게 했습니다. 원본 미리보기·YOLO 크롭·Smol 문자열 등은 결과 JSON 과 동일하게 참고용입니다.</p>
<div class="problem-picks" id="hr-problem-picks"></div>
<div class="problem-panel-wrap" id="hr-problem-panel-wrap" hidden>
  <div id="hr-problem-panel"></div>
</div>
<script type="application/json" id="hr-problem-payload">{safe}</script>
<script>{_js}</script>
"""


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
    materialize_problem_debug_images: bool = True,
) -> None:
    """
    HR-Bench 전략 비교 HTML 을 ``path`` 에 기록한다.

    요약 표 아래에 Chart.js(CDN) 막대 그래프 4종(정확도·시간·토큰·GPU)을 넣고,
    ``materialize_problem_debug_images`` 가 참이면(기본값) 문제 패널용 base64 JPEG 를
    ``path`` 와 같은 디렉터리의 ``{stem}_img/`` 에 파일로 저장하고, 임베디드 JSON 은 상대 경로만 담습니다.
    """
    stem_img = path.parent / f"{path.stem}_img"
    if stem_img.is_dir():
        shutil.rmtree(stem_img)

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

    bp = _build_hr_problem_browser_payload(per_strategy_rows)
    problem_payload_obj: dict[str, Any] | None
    if bp is None:
        problem_payload_obj = None
    elif materialize_problem_debug_images:
        problem_payload_obj = copy.deepcopy(bp)
        _materialize_hr_problem_payload_assets(problem_payload_obj, path)
    else:
        problem_payload_obj = bp
    problem_browser_block = _hr_problem_detail_browser_section_from_payload(problem_payload_obj)

    html_assets_base_uri = path.parent.resolve().as_uri()
    if not html_assets_base_uri.endswith("/"):
        html_assets_base_uri += "/"

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <base href="{html_mod.escape(html_assets_base_uri)}"/>
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
    .problem-picks {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.6rem 0; }}
    button.hr-btn-pick {{
      cursor: pointer; font: inherit; padding: 0.35rem 0.65rem;
      border: 1px solid #bbb; border-radius: 6px; background: #fff;
    }}
    button.hr-btn-pick[aria-pressed="true"] {{ background: #e3f2fd; border-color: #1976d2; }}
    .problem-panel-wrap {{ margin-top: 0.75rem; background: #fff; border: 1px solid #ccc;
      padding: 0.85rem; border-radius: 8px; max-width: 72rem; }}
    .hr-problem-detail h4 {{ margin: 0.6rem 0 0.2rem 0; font-size: 0.95rem; }}
    section.hr-strategy-block {{
      margin: 1rem 0; padding-bottom: 0.75rem; border-bottom: 1px dashed #ddd; }}
    .hr-strategy-block:last-child {{ border-bottom: none; }}
    .hr-bbox-wrap {{ margin: 0.25rem 0; }}
    .hr-bbox-hint {{ font-size: 0.72rem; color: #666; }}
    .hr-dbg-block {{ margin: 0.5rem 0; }}
    .hr-crops {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: flex-start; }}
    .hr-crops figure {{ margin: 0; max-width: 14rem; font-size: 0.72rem; color: #555; }}
    .hr-pre {{
      white-space: pre-wrap; word-break: break-word; font-size: 0.72rem;
      background: #f6f8fa; border: 1px solid #e1e4e8; padding: 0.5rem; border-radius: 6px;
      max-height: 26rem; overflow: auto;
    }}
    .mono-tight code {{ font-size: 0.78rem; }}
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
  {problem_browser_block}
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
