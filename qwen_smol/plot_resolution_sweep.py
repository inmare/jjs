#!/usr/bin/env python3
"""resolution_sweep_summary.csv → 축별 종합 대시보드 (정확도·토큰·VRAM·시간, 한국어 파일명)."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_W3 = _REPO / "11주차-3"
for p in (_REPO, _W3, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from week3.font_setup import configure_matplotlib_korean  # noqa: E402

QWEN_SMOL_DIR = Path(__file__).resolve().parent

# sweep_axis → 메타 (파일명 slug, 제목, x축, 설명)
SWEEP_AXIS_META: dict[str, dict[str, str]] = {
    "control": {
        "file_slug": "01_대조군_Qwen단독_입력해상도",
        "title": "01 대조군 — Qwen 단독 (원본 1장)",
        "xlabel": "Qwen에 보내는 이미지 긴 변 (px)",
        "varied_ko": (
            "변경: Qwen에 원본 이미지 1장만 리사이즈해서 전송 (Single_VLM). "
            "YOLO·Smol·crop·썸네일 미사용"
        ),
    },
    "single_vlm": {
        "file_slug": "01_대조군_Qwen단독_입력해상도",
        "title": "01 대조군 — Qwen 단독 (원본 1장)",
        "xlabel": "Qwen에 보내는 이미지 긴 변 (px)",
        "varied_ko": (
            "변경: Qwen에 원본 이미지 1장만 리사이즈해서 전송 (Single_VLM). "
            "YOLO·Smol·crop·썸네일 미사용"
        ),
    },
    "smol": {
        "file_slug": "02_Smol입력해상도",
        "title": "02 Ablation — SmolVLM 설명용 입력 해상도",
        "xlabel": "SmolVLM Phase1 입력 긴 변 (px)",
        "varied_ko": (
            "변경: SmolVLM이 장면 설명을 만들 때 보는 이미지 크기. "
            "Qwen crop·썸네일·YOLO 탐지 해상도는 baseline 고정"
        ),
    },
    "crop_qwen": {
        "file_slug": "03_YOLOcrop_Qwen축소배율",
        "title": "03 Ablation — YOLO crop → Qwen 전송 축소",
        "xlabel": "crop 균일 축소 배율 (1.0=YOLO가 자른 크기 그대로)",
        "varied_ko": (
            "변경: YOLO로 잘라낸 crop을 Qwen에 넣기 직전에 줄이는 비율. "
            "YOLO 탐지·Smol·썸네일은 baseline 고정"
        ),
    },
    "thumb": {
        "file_slug": "04_썸네일크기",
        "title": "04 Ablation — 전체 씬 썸네일(Thumb) 크기",
        "xlabel": "썸네일 긴 변 (px)",
        "varied_ko": (
            "변경: YOLO+Smol+썸네일 파이프라인에서 Qwen에 보내는 "
            "전체 장면 썸네일 1장의 해상도. Smol·crop·YOLO 탐지는 baseline 고정"
        ),
    },
    "yolo_ctx": {
        "file_slug": "05_YOLO탐지해상도_ctx축소",
        "title": "05 Ablation — YOLO 탐지 해상도 (ctx 축소)",
        "xlabel": "원본 대비 YOLO 탐지 이미지 축소 배율 (0.5=절반 해상도)",
        "varied_ko": (
            "변경: YOLO가 객체를 찾을 때 쓰는 이미지 = 원본×배율로 축소 후 탐지 → "
            "bbox·crop 개수·크기가 달라짐. Smol·crop축소·썸네일은 baseline 고정"
        ),
    },
}

# 파이프라인 3종 — 모든 ablation 그래프에 공통 표시 (데이터 있을 때만 선 그림)
PIPELINE_SERIES: list[tuple[str, str, str]] = [
    ("Qwen 단독", r"^Single_VLM_Qwen_4B", "#2563eb"),
    ("YOLO+Smol", r"^Smol_YOLO_Qwen_4B", "#dc2626"),
    ("YOLO+Smol+썸네일", r"^Thumb_Smol_YOLO_Qwen_4B", "#16a34a"),
]

SCALE_AXES = frozenset({"crop_qwen", "yolo_ctx"})
CONTROL_AXES = frozenset({"control", "single_vlm"})
CONTROL_BASELINE_PX = 1280
SINGLE_VLM_PATTERN = r"^Single_VLM_Qwen_4B"
CONTROL_OVERLAY_LABEL = "Qwen 단독 (대조군)"
CONTROL_HLINE_LABEL = f"Qwen 단독 ({CONTROL_BASELINE_PX}px 대조)"

RESOURCE_PANELS: list[tuple[str, str, str]] = [
    ("avg_smol_image_tokens", "Smol 이미지 토큰", "토큰"),
    ("avg_qwen_image_tokens", "Qwen 이미지 토큰", "토큰"),
    ("avg_smol_peak_mem_mb", "Smol VRAM 피크", "MB"),
    ("avg_qwen_peak_mem_mb", "Qwen VRAM 피크", "MB"),
    ("avg_total_time_sec", "총 처리 시간", "초"),
]

# ablation 축별 종합 대시보드 패널 (정확도 + 리소스)
DASHBOARD_PANELS: list[tuple[str, str, str, str]] = [
    ("accuracy_percent", "정확도", "%", "{:.1f}%"),
    ("avg_qwen_image_tokens", "Qwen 이미지 토큰", "토큰", "{:.0f}"),
    ("avg_smol_image_tokens", "Smol 이미지 토큰", "토큰", "{:.0f}"),
    ("avg_qwen_peak_mem_mb", "Qwen VRAM 피크", "MB", "{:.0f}"),
    ("avg_smol_peak_mem_mb", "Smol VRAM 피크", "MB", "{:.0f}"),
    ("avg_total_time_sec", "총 처리 시간", "초", "{:.1f}s"),
]


def side_sort_key(v: float) -> tuple[int, float]:
    if v <= 0:
        return (1, 10**9)
    return (0, v)


def value_label(axis: str, v: float) -> str:
    if axis in SCALE_AXES:
        if abs(v - round(v)) < 0.01:
            return f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{v:.2g}"
    return "Orig" if v <= 0 else str(int(v))


def safe_filename(text: str) -> str:
    """Windows 호환 파일명."""
    return re.sub(r'[<>:"/\\|?*]', "_", text)


def axis_meta(axis: str) -> dict[str, str]:
    return SWEEP_AXIS_META.get(
        axis,
        {
            "file_slug": axis,
            "title": axis,
            "xlabel": "sweep_value",
            "varied_ko": f"변경 요소: {axis}",
        },
    )


def resolve_sweep_csv(target: str) -> Path:
    p = Path(target).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        sweep = p / "resolution_sweep_summary.csv"
        if sweep.is_file():
            return sweep
        raise FileNotFoundError(f"resolution_sweep_summary.csv 없음: {p}")
    raise FileNotFoundError(f"경로 없음: {p}")


def find_latest_sweep_dir(root: Path | None = None) -> Path | None:
    base = root if root is not None else QWEN_SMOL_DIR
    runs = sorted(
        base.glob("qwen_smol_resolution_sweep_*/"),
        key=lambda x: x.stat().st_mtime,
    )
    return runs[-1] if runs else None


def filter_pipeline(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    mask = df["method"].astype(str).str.match(pattern, na=False)
    return df.loc[mask].copy()


def sort_by_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["sweep_value"] = pd.to_numeric(out["sweep_value"], errors="coerce").fillna(0)
    out["_sort"] = out["sweep_value"].map(side_sort_key)
    return out.sort_values("_sort").drop(columns="_sort")


def unified_sweep_values(frames: list[pd.DataFrame]) -> list[float]:
    vals: set[float] = set()
    for frame in frames:
        if frame.empty:
            continue
        for v in frame["sweep_value"]:
            vals.add(float(v))
    return sorted(vals, key=side_sort_key)


def get_control_single_vlm(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """대조군(control) 스윕의 Single_VLM 행."""
    cdf = df[
        (df["sweep_axis"].astype(str).isin(CONTROL_AXES))
        & (df["benchmark"].astype(str) == benchmark)
    ]
    return sort_by_sweep(filter_pipeline(cdf, SINGLE_VLM_PATTERN))


def plot_pipelines_on_axis(
    ax: Any,
    *,
    axis: str,
    bench_df: pd.DataFrame,
    y_col: str,
    panel_title: str,
    y_unit: str,
    control_df: pd.DataFrame | None = None,
    annotate: bool = False,
    value_fmt: str = "{:.0f}",
    show_legend: bool = True,
    show_xlabel: bool = True,
) -> bool:
    """파이프라인 3종 + (ablation 시) 대조군 Qwen 단독 곡선/기준선."""
    overlay_control = (
        axis not in CONTROL_AXES
        and control_df is not None
        and not y_col.startswith("avg_smol_")
    )
    control_frame = (
        get_control_single_vlm(control_df, str(bench_df["benchmark"].iloc[0]))
        if overlay_control and control_df is not None and not bench_df.empty
        else pd.DataFrame()
    )

    series_frames: list[tuple[str, pd.DataFrame, str, str]] = []
    for name, pat, color in PIPELINE_SERIES:
        if name == "Qwen 단독" and overlay_control:
            continue  # ablation 축: control 스윕에서 별도 오버레이
        sf = sort_by_sweep(filter_pipeline(bench_df, pat))
        if not sf.empty and y_col in sf.columns:
            series_frames.append((name, sf, color, "-"))

    if overlay_control and not control_frame.empty and y_col in control_frame.columns:
        series_frames.append((CONTROL_OVERLAY_LABEL, control_frame, "#2563eb", "--"))

    if not series_frames:
        ax.set_title(f"{panel_title}\n(데이터 없음)", fontsize=9)
        return False

    meta = axis_meta(axis)
    ablation_frames = [sf for name, sf, _, _ in series_frames if name != CONTROL_OVERLAY_LABEL]

    if axis in SCALE_AXES and overlay_control and not control_frame.empty:
        values = unified_sweep_values(ablation_frames if ablation_frames else [bench_df])
    else:
        all_frames = [sf for _, sf, _, _ in series_frames]
        values = unified_sweep_values(all_frames)

    xs = list(range(len(values)))
    tick_labels = [value_label(axis, v) for v in values]
    any_line = False

    for name, frame, color, linestyle in series_frames:
        lookup = {float(r["sweep_value"]): float(r[y_col]) for _, r in frame.iterrows()}
        ys = [lookup.get(v, math.nan) for v in values]
        if not any(not math.isnan(y) for y in ys):
            continue
        ax.plot(
            xs,
            ys,
            marker="o" if linestyle == "-" else "s",
            linewidth=2.0 if linestyle == "-" else 1.8,
            markersize=6 if linestyle == "-" else 5,
            color=color,
            linestyle=linestyle,
            label=name,
        )
        any_line = True
        if annotate and linestyle == "-":
            for x, y in zip(xs, ys):
                if math.isnan(y):
                    continue
                ax.annotate(
                    value_fmt.format(y),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=6,
                    color=color,
                )

    # 배율 축: 1280px 대조군 기준 수평선 (곡선 x축과 직접 대응 불가)
    if (
        axis in SCALE_AXES
        and overlay_control
        and not control_frame.empty
        and y_col in control_frame.columns
    ):
        baseline_rows = control_frame[
            control_frame["sweep_value"].astype(float) == float(CONTROL_BASELINE_PX)
        ]
        if baseline_rows.empty:
            baseline_rows = control_frame.iloc[[-1]]
        y_ref = float(baseline_rows[y_col].iloc[0])
        ax.axhline(
            y_ref,
            color="#2563eb",
            linestyle=":",
            linewidth=1.6,
            label=CONTROL_HLINE_LABEL,
        )
        any_line = True

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(y_unit, fontsize=9)
    ax.set_title(panel_title, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    if show_legend:
        ax.legend(loc="best", fontsize=7, title="파이프라인")
    if show_xlabel:
        ax.set_xlabel(meta["xlabel"], fontsize=8)
    return any_line


def skip_smol_panel(axis: str, y_col: str, bench_df: pd.DataFrame) -> bool:
    """대조군에서 Smol 미사용 → Smol 패널 생략."""
    if axis not in ("control", "single_vlm"):
        return False
    if not y_col.startswith("avg_smol_"):
        return False
    col_vals = pd.to_numeric(bench_df.get(y_col, 0), errors="coerce").fillna(0)
    return bool((col_vals <= 0).all())


def dashboard_panels_for_axis(axis: str, bench_df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """축·데이터에 맞는 대시보드 패널 목록."""
    out: list[tuple[str, str, str, str]] = []
    for col, title, unit, fmt in DASHBOARD_PANELS:
        if col not in bench_df.columns:
            continue
        if skip_smol_panel(axis, col, bench_df):
            continue
        out.append((col, title, unit, fmt))
    return out


def subplot_grid(n_panels: int) -> tuple[int, int]:
    """패널 수에 맞는 (rows, cols)."""
    if n_panels <= 4:
        return 2, 2
    if n_panels <= 6:
        return 2, 3
    cols = 3
    rows = math.ceil(n_panels / cols)
    return rows, cols


def plot_axis_dashboard(
    *,
    axis: str,
    bench: str,
    bench_df: pd.DataFrame,
    full_df: pd.DataFrame,
    out_dir: Path,
    file_slug: str,
    bench_slug: str,
) -> Path | None:
    """ablation 축 1개 · 벤치 1개 — 정확도·토큰·VRAM·시간 종합 대시보드."""
    import matplotlib.pyplot as plt

    panels = dashboard_panels_for_axis(axis, bench_df)
    if not panels:
        return None

    meta = axis_meta(axis)
    nrows, ncols = subplot_grid(len(panels))
    fig_w = 5.2 * ncols
    fig_h = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    suptitle = f"{bench} — {meta['title']}\n{meta['varied_ko']}"
    if axis not in CONTROL_AXES and "control" in set(full_df["sweep_axis"].astype(str)):
        suptitle += (
            "\n파란 점선 = 대조군 Qwen 단독 · 각 패널 = 정확도 / 토큰 / VRAM / 처리시간"
        )
    else:
        suptitle += "\n각 패널 = 정확도 / 토큰 / VRAM / 처리시간"

    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for idx, (col, title, unit, fmt) in enumerate(panels):
        ax = flat_axes[idx]
        is_acc = col == "accuracy_percent"
        ok = plot_pipelines_on_axis(
            ax,
            axis=axis,
            bench_df=bench_df,
            y_col=col,
            panel_title=title,
            y_unit=unit,
            control_df=full_df,
            annotate=is_acc,
            value_fmt=fmt,
            show_legend=idx == 0,
            show_xlabel=idx >= (len(panels) - ncols),
        )
        if is_acc and ok:
            ax.set_ylim(-2, 102)
        if idx == 0 and ok:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    for ax in flat_axes[len(panels):]:
        ax.set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(4, len(legend_labels)),
            fontsize=8,
            title="파이프라인",
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08 if legend_handles else 0.06)

    dash_path = out_dir / f"{bench_slug}_{file_slug}_지표종합.png"
    fig.savefig(dash_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dash_path


def generate_resolution_sweep_plots(
    csv_path: Path,
    *,
    plots_dir: Path | None = None,
) -> list[Path]:
    configure_matplotlib_korean()

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"비어 있는 CSV: {csv_path}")

    if "sweep_value" not in df.columns:
        df["sweep_value"] = df.get("max_side", 0)

    out_dir = plots_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 축 순서: 대조군 → 4 ablation
    axis_order = ["control", "single_vlm", "smol", "crop_qwen", "thumb", "yolo_ctx"]
    present = [a for a in axis_order if a in set(df["sweep_axis"].astype(str))]
    present += sorted(
        a for a in df["sweep_axis"].astype(str).unique() if a not in axis_order
    )

    benchmarks = sorted(df["benchmark"].astype(str).unique())

    for axis in present:
        meta = axis_meta(axis)
        axis_df = df[df["sweep_axis"].astype(str) == axis]
        file_slug = safe_filename(meta["file_slug"])

        for bench in benchmarks:
            bench_df = axis_df[axis_df["benchmark"].astype(str) == bench]
            if bench_df.empty:
                continue

            bench_slug = safe_filename(bench)
            dash_path = plot_axis_dashboard(
                axis=axis,
                bench=bench,
                bench_df=bench_df,
                full_df=df,
                out_dir=out_dir,
                file_slug=file_slug,
                bench_slug=bench_slug,
            )
            if dash_path is not None:
                saved.append(dash_path)

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="resolution sweep 그래프 (한국어 파일명 · 파이프라인 3종 통합)",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="run 폴더 또는 resolution_sweep_summary.csv",
    )
    parser.add_argument("--out-dir", default="", help="PNG 저장 폴더")
    args = parser.parse_args()

    if args.target.strip():
        csv_path = resolve_sweep_csv(args.target)
    else:
        latest = find_latest_sweep_dir()
        if latest is None:
            print("[오류] qwen_smol_resolution_sweep_* 폴더가 없습니다.", file=sys.stderr)
            return 1
        csv_path = resolve_sweep_csv(str(latest))
        print(f"최신 스윕 폴더: {csv_path.parent}", flush=True)

    plots_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else None
    paths = generate_resolution_sweep_plots(csv_path, plots_dir=plots_dir)
    print(f"그래프 {len(paths)}개 저장:", flush=True)
    for p in paths:
        print(f"  {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
