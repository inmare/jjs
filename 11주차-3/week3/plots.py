"""벤치마크 summary CSV → 11주차 스타일 PNG 그래프."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _short_method(name: str, max_len: int = 28) -> str:
    s = str(name)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _f(val: Any) -> float:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _require_matplotlib():
    try:
        from week3.font_setup import configure_matplotlib_korean

        configure_matplotlib_korean()
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise RuntimeError(
            "그래프 생성에는 matplotlib 이 필요합니다: uv sync --group dev"
        ) from e


def write_four_metric_bars(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    metrics: list[tuple[str, str, str]],
) -> None:
    """metrics: (column, ylabel, subplot_title) × 4 → 2×2 PNG."""
    plt = _require_matplotlib()
    if df.empty:
        return

    methods = [_short_method(m) for m in df["method"].tolist()]
    x = list(range(len(methods)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=13, y=1.02)

    colors = plt.cm.tab10(range(len(methods)))  # type: ignore[attr-defined]

    for ax, (col, ylabel, sub_title) in zip(axes.flat, metrics):
        vals = [_f(v) for v in df[col].tolist()]
        bars = ax.bar(x, vals, color=colors)
        ax.set_title(sub_title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        if col == "accuracy_percent" and max(vals, default=0) <= 0:
            ax.text(
                0.5,
                0.5,
                "detect-only\n(정확도 없음)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
        for bar, v in zip(bars, vals):
            if v > 0 and col in ("accuracy_percent", "avg_num_objects"):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_single_metric_comparison(
    df: pd.DataFrame,
    *,
    out_path: Path,
    column: str,
    ylabel: str,
    title: str,
) -> None:
    """벤치마크별 그룹 막대 (11주차 crop_policy_full_* 스타일)."""
    plt = _require_matplotlib()
    if df.empty or "benchmark" not in df.columns:
        return

    benchmarks = sorted(df["benchmark"].unique().tolist())
    methods = sorted(df["method"].unique().tolist())
    if not methods:
        return

    n_bench = len(benchmarks)
    n_methods = len(methods)
    width = 0.8 / max(n_methods, 1)
    x_base = list(range(n_bench))

    fig, ax = plt.subplots(figsize=(max(10, n_bench * 2.2), 6))
    for j, method in enumerate(methods):
        offsets = [xb + (j - (n_methods - 1) / 2) * width for xb in x_base]
        vals = []
        for bench in benchmarks:
            row = df[(df["benchmark"] == bench) & (df["method"] == method)]
            vals.append(_f(row[column].iloc[0]) if len(row) else 0.0)
        ax.bar(
            offsets,
            vals,
            width=width * 0.95,
            label=_short_method(method, 36),
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_base)
    ax.set_xticklabels(benchmarks, rotation=15, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


DEFAULT_FOUR_METRICS: list[tuple[str, str, str]] = [
    ("accuracy_percent", "%", "정확도"),
    ("avg_image_tokens", "tokens", "이미지 토큰"),
    ("avg_total_time_sec", "sec", "총 시간"),
    ("avg_overall_peak_allocated_mb", "MB", "피크 메모리"),
]

DETECT_FOUR_METRICS: list[tuple[str, str, str]] = [
    ("avg_preprocess_time_sec", "sec", "전처리(탐지) 시간"),
    ("avg_num_objects", "count", "crop 개수"),
    ("avg_total_time_sec", "sec", "총 시간"),
    ("avg_image_tokens", "tokens", "이미지 토큰"),
]

# CSV / 마크다운 요약 표용 한글 컬럼명
SUMMARY_TABLE_COLUMNS: list[tuple[str, str]] = [
    ("benchmark", "벤치마크"),
    ("method", "방법"),
    ("rows", "문항 수"),
    ("accuracy_percent", "정확도(%)"),
    ("avg_preprocess_time_sec", "전처리 시간(s)"),
    ("avg_total_time_sec", "총 시간(s)"),
    ("avg_image_tokens", "이미지 토큰"),
    ("avg_num_objects", "crop 수"),
    ("avg_overall_peak_allocated_mb", "피크 메모리(MB)"),
]


def generate_plots_from_summary_csv(
    csv_path: Path,
    *,
    plots_dir: Path | None = None,
) -> list[Path]:
    """
    summary CSV 를 읽어 PNG 생성.

    - ``{benchmark}_summary_panel.png`` — 벤치별 2×2
    - ``detector_sweep_{metric}.png`` — 전 벤치 통합 비교 (11주차 crop_policy 스타일)
    """
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV 가 비어 있습니다: {csv_path}")

    if plots_dir is not None:
        out_dir = plots_dir
    elif csv_path.name == "summary.csv" and csv_path.parent.name.startswith("summary_"):
        out_dir = csv_path.parent
    else:
        out_dir = csv_path.parent / f"{csv_path.stem}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    has_accuracy = df["accuracy_percent"].fillna(0).max() > 0
    four = DEFAULT_FOUR_METRICS if has_accuracy else DETECT_FOUR_METRICS

    written: list[Path] = []

    for bench in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == bench].copy()
        panel = out_dir / f"{bench}_summary_panel.png"
        write_four_metric_bars(
            sub,
            out_path=panel,
            title=f"{bench} — 탐지 CNN 비교",
            metrics=four,
        )
        written.append(panel)

    for col, ylabel, nice in four:
        p = out_dir / f"detector_sweep_{col}.png"
        write_single_metric_comparison(
            df,
            out_path=p,
            column=col,
            ylabel=ylabel,
            title=f"전체 벤치마크 — {nice}",
        )
        written.append(p)

    md_path = write_summary_markdown_table(df, out_dir / "summary_table.md")
    written.append(md_path)

    return written


def write_summary_markdown_table(df: pd.DataFrame, out_path: Path) -> Path:
    """엑셀·노션용 한글 헤더 마크다운 표."""
    cols = [(c, label) for c, label in SUMMARY_TABLE_COLUMNS if c in df.columns]
    if not cols:
        out_path.write_text("(데이터 없음)\n", encoding="utf-8")
        return out_path

    header = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells: list[str] = []
        for col, _ in cols:
            v = row[col]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if col != "accuracy_percent" else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
