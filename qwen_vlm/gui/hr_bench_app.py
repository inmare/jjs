"""
HR-Bench 평가 UI: **pywebview** 단일 창 + 로컬 HTML 대시보드.

오른쪽 미리보기는 ``docs/hr_bench_last.html`` 내용을 Python 이 읽어 iframe ``srcdoc`` 으로 넣습니다
(인라인 HTML 부모에서 ``file://`` iframe 이 막히는 환경 대응). 요약 표·**Chart.js** 차트 동일.

실행:

  uv run python -m qwen_vlm.gui.hr_bench_app
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser

import webview
from pathlib import Path
from typing import Any

from qwen_vlm.hr_bench.io import HR_BENCH_YOLO_WEIGHTS
from qwen_vlm.hr_bench.report import inline_hr_compare_materialized_images_for_srcdoc
from qwen_vlm.hr_bench.strategies import STRATEGIES_ALL
from qwen_vlm.main import ROOT
from qwen_vlm.pipeline.experiment import DEFAULT_PROMPT_KO, list_frames
from qwen_vlm.pipeline.week import SMOL_GGUF, SMOL_MMPROJ
from qwen_vlm.utils.stdio_utf8 import configure_stdio_utf8
from qwen_vlm.utils.subprocess_util import (
    child_env_for_utf8_stdio,
    run_subprocess_stream_text_lines,
)


def _gui_subprocess_cli_prefix(module: str) -> list[str]:
    """
    ``HR-Bench 실행`` / 연속 프레임 비교가 띄우는 하위 프로세스 접두.

    ``uv run python -m …`` 대신 ``sys.executable -m …`` 를 쓰면, 이 GUI를 연
    인터프리터(예: ``uv run python scripts/experiment_gui.py``)와 동일 환경으로
    패키지 코드(Smol ``--no-jinja``·API 순서 등)가 바로 반영된다.
    """
    return [sys.executable, "-m", module]


_DOCS = ROOT / "docs"
HR_BENCH_JSON = _DOCS / "hr_bench_last.json"
HR_BENCH_HTML = _DOCS / "hr_bench_last.html"
HR_BENCH_PNG = _DOCS / "hr_bench_last_charts.png"
SEQUENCE_COMPARE_JSON = _DOCS / "sequence_compare_last.json"
SEQUENCE_COMPARE_HTML = _DOCS / "sequence_compare_last.html"
_DASHBOARD_TEMPLATE = Path(__file__).resolve().parent / "hr_bench_dashboard.html"


def _rel_posix_from_root(path: Path | str) -> str:
    """``ROOT`` 기준 상대 경로(posix). 저장소 밖이면 절대 posix 로 남긴다."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix().replace("\\", "/")


def _enrich_bootstrap_file_uris(boot: dict[str, Any]) -> None:
    """``file://`` 미리보기용 URI 를 상대 ``paths`` 로부터 채운다."""
    paths = boot.get("paths")
    if not isinstance(paths, dict):
        return
    for rel_key, uri_key in (
        ("reportHtml", "reportHtmlUri"),
        ("sequenceHtml", "sequenceHtmlUri"),
    ):
        rel = paths.get(rel_key)
        if not rel:
            continue
        paths[uri_key] = (ROOT / str(rel)).resolve().as_uri()


def default_gui_bootstrap() -> dict[str, Any]:
    """대시보드 초기 폼 값·경로·전략 목록을 담은 dict (JSON 직렬화 가능)."""
    return {
        "strategies": list(STRATEGIES_ALL),
        "splits": ["hrbench_4k", "hrbench_8k"],
        "paths": {
            "reportHtml": _rel_posix_from_root(HR_BENCH_HTML),
            "reportJson": _rel_posix_from_root(HR_BENCH_JSON),
            "pngOut": _rel_posix_from_root(HR_BENCH_PNG),
            "docsDir": _rel_posix_from_root(_DOCS),
            "sequenceHtml": _rel_posix_from_root(SEQUENCE_COMPARE_HTML),
            "sequenceJson": _rel_posix_from_root(SEQUENCE_COMPARE_JSON),
        },
        "defaults": {
            "strategies_checked": list(STRATEGIES_ALL),
            "split": "hrbench_4k",
            "max_samples": "10",
            "sample_mode": "sequential",
            "start": "0",
            "seed": "",
            "context_max_side": "960",
            "context_resize_scale": "0.5",
            "context_by_long_edge": False,
            "crop_max_side": "0",
            "max_crops": "0",
            "min_crop_short_side": "0",
            "min_crop_area": "0",
            "yolo_weights": HR_BENCH_YOLO_WEIGHTS,
            "yolo_surveillance_classes_only": False,
            "yolo_max_bbox_area_num": "1",
            "yolo_max_bbox_area_den": "4",
            "yolo_overview_max_side": "960",
            "qwen_only_use_original": True,
            "qwen_image_max_long_side": "0",
            "max_tokens": "8",
            "smol_max_tokens": "256",
            "save_png": True,
            "no_sample_tables": False,
            "omit_smol_if_unavailable": False,
            "disable_yolo_vlm_budget": False,
            "smol_gguf": _rel_posix_from_root(SMOL_GGUF),
            "smol_mmproj": _rel_posix_from_root(SMOL_MMPROJ),
        },
        "sequence": {
            "frames_dir": (
                "data/datasets/shanghaitech/shanghaitech/testing/frames"
            ),
            "frame_mask_dir": (
                "data/datasets/shanghaitech/shanghaitech/testing/"
                "test_frame_mask"
            ),
            "pixel_mask_dir": (
                "data/datasets/shanghaitech/shanghaitech/testing/"
                "test_pixel_mask"
            ),
            "max_frames": "6",
            "base_url": "http://127.0.0.1:8765/v1",
            "api_key": "sk-local",
            "model": "qwen3-vl-4b-q8",
            "prompt": DEFAULT_PROMPT_KO,
            "max_tokens": "256",
            "context_max_side": "960",
            "context_resize_scale": "0.5",
            "context_by_long_edge": False,
            "crop_max_side": "0",
            "max_crops": "0",
            "min_crop_short_side": "0",
            "min_crop_area": "0",
            "yolo_model": "yolo26n.pt",
            "yolo_surveillance_classes_only": False,
            "yolo_max_bbox_area_num": "1",
            "yolo_max_bbox_area_den": "4",
            "yolo_device": "cpu",
            "frame_gate": "yolo",
            "mse_threshold": "2.5",
            "use_crop_layout_gate": True,
            "crop_gate_max_shift": "0.02",
            "gate_min_crops_for_vlm": "0",
            "gate_max_crops_for_vlm": "0",
            "disable_yolo_vlm_budget": False,
        },
    }


def validate_gui_config(cfg: dict[str, Any]) -> str | None:
    """
    ``run_bench`` 직전 검증.

    Returns:
        오류 메시지(한 줄) 또는 ``None`` 이면 통과.
    """
    strategies = cfg.get("strategies") or []
    if not strategies:
        return "전략을 하나 이상 선택하세요."
    try:
        int(str(cfg.get("max_samples") or "0").strip())
        int(str(cfg.get("context_max_side") or "0").strip())
        int(str(cfg.get("crop_max_side") or "0").strip())
        int(str(cfg.get("max_crops") or "0").strip())
        int(str(cfg.get("yolo_max_bbox_area_num") or "1").strip())
        int(str(cfg.get("yolo_max_bbox_area_den") or "4").strip())
        int(str(cfg.get("yolo_overview_max_side") or "960").strip())
    except ValueError:
        return "숫자 필드(max-samples, context, crop, max-crops, yolo 면적 N/M, yolo-overview-max-side)를 확인하세요."
    if not cfg.get("context_by_long_edge"):
        sc = str(cfg.get("context_resize_scale") or "").strip()
        if sc:
            try:
                if float(sc) <= 0:
                    return "context-resize-scale 는 양수이거나 비워 두세요."
            except ValueError:
                return "context-resize-scale 는 숫자이거나 비워 두세요."
    den = int(str(cfg.get("yolo_max_bbox_area_den") or "4").strip())
    if den <= 0:
        return "yolo-max-bbox-area-den 은 1 이상이어야 합니다."
    if str(cfg.get("sample_mode") or "") == "sequential":
        try:
            int(str(cfg.get("start") or "0").strip())
        except ValueError:
            return "start 는 정수여야 합니다."
    return None


def hr_bench_command_from_gui_config(cfg: dict[str, Any]) -> list[str]:
    """
    GUI에서 수집한 설정으로 ``uv run python -m qwen_vlm.cli.hr_bench`` 인자 리스트를 만든다.

    Args:
        cfg: ``collectConfig()`` 와 동일 키 (strategies, split, …).

    Returns:
        subprocess 에 그대로 넘길 argv (``sys.executable -m …`` 로 시작).
    """
    strategies: list[str] = list(cfg.get("strategies") or [])
    strat_arg = ",".join(strategies)
    cmd: list[str] = [
        *_gui_subprocess_cli_prefix("qwen_vlm.cli.hr_bench"),
        "--split",
        str(cfg.get("split") or "hrbench_4k").strip(),
        "--max-samples",
        str(cfg.get("max_samples") or "0").strip(),
        "--sample-mode",
        str(cfg.get("sample_mode") or "sequential").strip(),
        "--strategies",
        strat_arg,
        "--context-max-side",
        str(cfg.get("context_max_side") or "0").strip(),
        "--crop-max-side",
        str(cfg.get("crop_max_side") or "0").strip(),
        "--max-crops",
        str(cfg.get("max_crops") or "0").strip(),
        "--yolo-weights",
        str(cfg.get("yolo_weights") or HR_BENCH_YOLO_WEIGHTS).strip(),
        "--yolo-max-bbox-area-num",
        str(cfg.get("yolo_max_bbox_area_num") or "1").strip(),
        "--yolo-max-bbox-area-den",
        str(cfg.get("yolo_max_bbox_area_den") or "4").strip(),
        "--min-crop-short-side",
        str(cfg.get("min_crop_short_side") or "0").strip() or "0",
        "--min-crop-area",
        str(cfg.get("min_crop_area") or "0").strip() or "0",
        "--yolo-overview-max-side",
        str(cfg.get("yolo_overview_max_side") or "960").strip() or "960",
        "--qwen-image-max-long-side",
        str(cfg.get("qwen_image_max_long_side") or "0").strip() or "0",
        "--max-tokens",
        str(cfg.get("max_tokens") or "8").strip() or "8",
        "--smol-max-tokens",
        str(cfg.get("smol_max_tokens") or "256").strip() or "256",
        "--json-out",
        _rel_posix_from_root(HR_BENCH_JSON),
        "--html-out",
        _rel_posix_from_root(HR_BENCH_HTML),
    ]
    if str(cfg.get("sample_mode") or "") == "sequential":
        cmd.extend(["--start", str(int(str(cfg.get("start") or "0").strip()))])
    else:
        seed_s = str(cfg.get("seed") or "").strip()
        if seed_s:
            cmd.extend(["--seed", seed_s])
    cmd.extend(
        [
            "--smol-gguf",
            _rel_posix_from_root(str(cfg.get("smol_gguf") or "").strip() or SMOL_GGUF),
            "--smol-mmproj",
            _rel_posix_from_root(str(cfg.get("smol_mmproj") or "").strip() or SMOL_MMPROJ),
        ]
    )
    if cfg.get("omit_smol_if_unavailable"):
        cmd.append("--omit-smol-if-unavailable")
    if cfg.get("context_by_long_edge"):
        cmd.append("--context-by-long-edge")
    else:
        sc = str(cfg.get("context_resize_scale") or "").strip()
        if sc:
            cmd.extend(["--context-resize-scale", sc])
    if cfg.get("yolo_surveillance_classes_only"):
        cmd.append("--yolo-surveillance-classes-only")
    if not cfg.get("qwen_only_use_original", True):
        cmd.append("--no-qwen-only-use-original")
    if cfg.get("disable_yolo_vlm_budget"):
        cmd.append("--disable-yolo-vlm-budget")
    if cfg.get("save_png"):
        cmd.extend(["--png-out", _rel_posix_from_root(HR_BENCH_PNG)])
    if cfg.get("no_sample_tables"):
        cmd.append("--no-sample-tables")
    return cmd


def validate_sequence_config(cfg: dict[str, Any]) -> str | None:
    """``run_sequence_compare`` 직전 검증."""
    d = str(cfg.get("frames_dir") or "").strip()
    if not d:
        return "프레임 디렉터리 경로를 입력하세요."
    p = Path(d).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    rp = p.resolve()
    if not rp.is_dir():
        return f"프레임 디렉터리가 없습니다: {rp}"
    if not list_frames(rp, 1):
        return (
            f"프레임(jpg/jpeg/png)을 찾지 못했습니다: {rp} — "
            "평탄한 폴더 또는 testing/frames/<시퀀스>/ 구조인지 확인하세요."
        )
    try:
        int(str(cfg.get("max_frames") or "0").strip())
        int(str(cfg.get("context_max_side") or "0").strip())
        int(str(cfg.get("crop_max_side") or "0").strip())
        int(str(cfg.get("max_crops") or "0").strip())
        int(str(cfg.get("max_tokens") or "0").strip())
        int(str(cfg.get("yolo_max_bbox_area_num") or "1").strip())
        int(str(cfg.get("yolo_max_bbox_area_den") or "4").strip())
        int(str(cfg.get("gate_min_crops_for_vlm") or "0").strip())
        int(str(cfg.get("gate_max_crops_for_vlm") or "0").strip())
        float(str(cfg.get("mse_threshold") or "0").strip())
        float(str(cfg.get("crop_gate_max_shift") or "0.02").strip())
    except ValueError:
        return "연속 프레임 탭의 숫자 필드를 확인하세요."
    if not cfg.get("context_by_long_edge"):
        sc = str(cfg.get("context_resize_scale") or "").strip()
        if sc:
            try:
                if float(sc) <= 0:
                    return "context-resize-scale 는 양수이거나 비워 두세요."
            except ValueError:
                return "context-resize-scale 는 숫자이거나 비워 두세요."
    den = int(str(cfg.get("yolo_max_bbox_area_den") or "4").strip())
    if den <= 0:
        return "yolo-max-bbox-area-den 은 1 이상이어야 합니다."
    fg = str(cfg.get("frame_gate") or "").strip().lower()
    if fg not in ("mse", "yolo", "mse_then_yolo", "off", "none", ""):
        return "frame_gate 는 mse | yolo | mse_then_yolo | off 중 하나여야 합니다."
    return None


def sequence_compare_command_from_gui_config(cfg: dict[str, Any]) -> list[str]:
    """GUI 설정으로 ``sequence-compare`` CLI argv 생성."""
    cmd: list[str] = [
        *_gui_subprocess_cli_prefix("qwen_vlm.pipeline.experiment"),
        "sequence-compare",
        "--frames-dir",
        _rel_posix_from_root(str(cfg.get("frames_dir") or "").strip()),
        "--max-frames",
        str(cfg.get("max_frames") or "6").strip(),
        "--base-url",
        str(cfg.get("base_url") or "http://127.0.0.1:8765/v1").strip(),
        "--api-key",
        str(cfg.get("api_key") or "sk-local").strip(),
        "--model",
        str(cfg.get("model") or "qwen3-vl-4b-q8").strip(),
        "--prompt",
        str(cfg.get("prompt") or DEFAULT_PROMPT_KO).strip(),
        "--max-tokens",
        str(cfg.get("max_tokens") or "256").strip(),
        "--context-max-side",
        str(cfg.get("context_max_side") or "960").strip(),
        "--crop-max-side",
        str(cfg.get("crop_max_side") or "0").strip(),
        "--max-crops",
        str(cfg.get("max_crops") or "0").strip(),
        "--min-crop-short-side",
        str(cfg.get("min_crop_short_side") or "0").strip() or "0",
        "--min-crop-area",
        str(cfg.get("min_crop_area") or "0").strip() or "0",
        "--yolo-model",
        str(cfg.get("yolo_model") or "yolo26n.pt").strip(),
        "--yolo-max-bbox-area-num",
        str(cfg.get("yolo_max_bbox_area_num") or "1").strip(),
        "--yolo-max-bbox-area-den",
        str(cfg.get("yolo_max_bbox_area_den") or "4").strip(),
        "--yolo-device",
        str(cfg.get("yolo_device") or "cpu").strip(),
        "--frame-gate",
        str(cfg.get("frame_gate") or "yolo").strip(),
        "--mse-threshold",
        str(cfg.get("mse_threshold") or "2.5").strip(),
        "--crop-gate-max-shift",
        str(cfg.get("crop_gate_max_shift") or "0.02").strip(),
        "--gate-min-crops-for-vlm",
        str(cfg.get("gate_min_crops_for_vlm") or "0").strip(),
        "--gate-max-crops-for-vlm",
        str(cfg.get("gate_max_crops_for_vlm") or "0").strip(),
        "--json-out",
        _rel_posix_from_root(SEQUENCE_COMPARE_JSON),
        "--compare-html-out",
        _rel_posix_from_root(SEQUENCE_COMPARE_HTML),
    ]
    if cfg.get("context_by_long_edge"):
        cmd.append("--context-by-long-edge")
    else:
        sc = str(cfg.get("context_resize_scale") or "").strip()
        if sc:
            cmd.extend(["--context-resize-scale", sc])
    if cfg.get("use_crop_layout_gate"):
        cmd.append("--use-crop-layout-gate")
    if cfg.get("yolo_surveillance_classes_only"):
        cmd.append("--yolo-surveillance-classes-only")
    if cfg.get("disable_yolo_vlm_budget"):
        cmd.append("--disable-yolo-vlm-budget")
    return cmd


def _load_dashboard_html() -> str:
    """패키지 옆 ``hr_bench_dashboard.html`` 템플릿을 UTF-8 로 읽는다."""
    if not _DASHBOARD_TEMPLATE.is_file():
        raise FileNotFoundError(f"대시보드 템플릿 없음: {_DASHBOARD_TEMPLATE}")
    return _DASHBOARD_TEMPLATE.read_text(encoding="utf-8")


def _dashboard_document() -> str:
    """부트스트랩 JSON을 넣은 완성 HTML 문자열."""
    tpl = _load_dashboard_html()
    boot = default_gui_bootstrap()
    _enrich_bootstrap_file_uris(boot)
    blob = json.dumps(boot, ensure_ascii=False)
    return tpl.replace("__BOOTSTRAP_PLACEHOLDER__", blob)


def _current_window():
    """pywebview 메인 창 (없으면 ``None``)."""
    try:
        wins = webview.windows
        return wins[0] if wins else None
    except Exception:
        return None


def _js_append_log(w, line: str) -> None:
    """한 줄 로그를 대시보드 ``<pre#log>`` 에 붙인다."""
    if not w:
        return
    try:
        w.evaluate_js(f"window._appendLog({json.dumps(line)})")
    except Exception:
        pass


def _js_bench_done(w, code: int) -> None:
    """실행 버튼 복구, 미리보기 갱신(JS), 창 포커스."""
    if not w:
        return
    try:
        w.evaluate_js(f"window._onBenchFinished({int(code)})")
    except Exception:
        pass
    try:
        w.show()
    except Exception:
        pass


def _js_sequence_done(w, code: int) -> None:
    if not w:
        return
    try:
        w.evaluate_js(f"window._onSequenceFinished({int(code)})")
    except Exception:
        pass
    try:
        w.show()
    except Exception:
        pass


class HrBenchWebApi:
    """
    pywebview ``js_api`` 객체.

    JS 에서 ``pywebview.api.run_bench(...)`` 등으로 호출한다.
    """

    def run_bench(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """
        HR-Bench CLI 를 백그라운드에서 실행하고, 로그는 ``evaluate_js`` 로 스트리밍한다.

        Args:
            cfg: 폼에서 수집한 설정 dict.

        Returns:
            ``{\"error\": \"…\"}`` 검증 실패 시, 아니면 ``{\"started\": true}``.
        """
        err = validate_gui_config(cfg)
        if err:
            return {"error": err}

        def work() -> None:
            w = _current_window()
            code = -1
            try:
                cmd = hr_bench_command_from_gui_config(cfg)
                _js_append_log(w, "$ " + " ".join(cmd))
                code = run_subprocess_stream_text_lines(
                    cmd,
                    cwd=str(ROOT),
                    env=child_env_for_utf8_stdio(),
                    append_line=lambda line: _js_append_log(w, line),
                )
            except Exception as e:
                _js_append_log(w, f"실행 실패: {e}")
                code = -1
            finally:
                _js_bench_done(w, code)

        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    def run_sequence_compare(self, cfg: dict[str, Any]) -> dict[str, Any]:
        err = validate_sequence_config(cfg)
        if err:
            return {"error": err}

        def work() -> None:
            w = _current_window()
            code = -1
            try:
                cmd = sequence_compare_command_from_gui_config(cfg)
                _js_append_log(w, "$ " + " ".join(cmd))
                code = run_subprocess_stream_text_lines(
                    cmd,
                    cwd=str(ROOT),
                    env=child_env_for_utf8_stdio(),
                    append_line=lambda line: _js_append_log(w, line),
                )
            except Exception as e:
                _js_append_log(w, f"실행 실패: {e}")
                code = -1
            finally:
                _js_sequence_done(w, code)

        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    def open_report_browser(self) -> None:
        """기본 브라우저로 ``hr_bench_last.html`` 을 연다."""
        p = HR_BENCH_HTML.resolve()
        if p.is_file():
            webbrowser.open(p.as_uri())

    def embed_report_preview(self) -> dict[str, Any]:
        """
        GUI iframe 에 넣을 리포트 HTML.

        pywebview 는 부모가 인라인 HTML일 때 자식 iframe 의 ``file://`` 가 막히는 경우가 있어
        내용을 ``srcdoc`` 으로 넣는데, 같은 이유로 ``./stem_img/*.jpg`` 도 로드가 거절되는 경우가 있어
        ``hr_problem`` JSON 속 경로만 data URL 로 치환한 뒤 base64 로 전달한다.
        """
        p = HR_BENCH_HTML.resolve()
        if not p.is_file():
            return {"error": "missing", "html_b64": None}
        try:
            text = inline_hr_compare_materialized_images_for_srcdoc(
                p.read_text(encoding="utf-8"), p
            )
            raw_bytes = text.encode("utf-8")
        except OSError as e:
            return {"error": str(e), "html_b64": None}
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return {"html_b64": b64, "error": None}

    def open_sequence_browser(self) -> None:
        p = SEQUENCE_COMPARE_HTML.resolve()
        if p.is_file():
            webbrowser.open(p.as_uri())

    def embed_sequence_preview(self) -> dict[str, Any]:
        p = SEQUENCE_COMPARE_HTML.resolve()
        if not p.is_file():
            return {"error": "missing", "html_b64": None}
        try:
            raw = p.read_bytes()
        except OSError as e:
            return {"error": str(e), "html_b64": None}
        b64 = base64.b64encode(raw).decode("ascii")
        return {"html_b64": b64, "error": None}

    def open_docs_folder(self) -> None:
        """``docs`` 디렉터리를 OS 탐색기로 연다."""
        d = _DOCS.resolve()
        d.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(d)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(d)], check=False)
        else:
            subprocess.run(["xdg-open", str(d)], check=False)

    def save_png_dialog(self) -> None:
        """차트 PNG 가 있으면 ``create_file_dialog`` 로 다른 이름 저장."""
        src = HR_BENCH_PNG.resolve()
        if not src.is_file():
            _js_append_log(_current_window(), f"[안내] PNG 없음: {src}")
            return
        w = _current_window()
        if not w:
            return
        try:
            paths = w.create_file_dialog(
                webview.FileDialog.SAVE,
                directory=str(_DOCS),
                save_filename=src.name,
            )
        except Exception as e:
            _js_append_log(w, f"다이얼로그 실패: {e}")
            return
        if not paths:
            return
        dest = Path(paths[0] if isinstance(paths, (list, tuple)) else paths)
        try:
            shutil.copy2(src, dest)
            _js_append_log(w, f"[저장] {dest}")
        except OSError as e:
            _js_append_log(w, f"[오류] {e}")


def main() -> None:
    """UTF-8 콘솔 설정 후 pywebview 창을 띄운다."""
    configure_stdio_utf8()
    html = _dashboard_document()
    webview.create_window(
        "qwen-vlm — HR-Bench · 연속 프레임",
        html=html,
        js_api=HrBenchWebApi(),
        width=1280,
        height=900,
    )
    webview.start()


if __name__ == "__main__":
    main()
