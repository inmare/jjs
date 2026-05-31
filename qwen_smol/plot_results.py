#!/usr/bin/env python3
"""qwen_smol summary.csv → PNG 그래프 (11주차-3 plot_results 와 동일 스타일)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_W3 = _REPO / "11주차-3"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from week3.plots import generate_plots_from_summary_csv  # noqa: E402

from summary import load_results_from_run_dir, write_summary_csvs  # noqa: E402

QWEN_SMOL_DIR = Path(__file__).resolve().parent


def find_latest_run_dir(root: Path | None = None) -> Path | None:
    base = root if root is not None else QWEN_SMOL_DIR
    runs = sorted(
        base.glob("qwen_smol_*/"),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else None


def resolve_summary_csv(target: str) -> Path:
    p = Path(target).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        summary = p / "summary.csv"
        if summary.is_file():
            return summary
        # summary.csv 없으면 method CSV 로부터 생성
        results = load_results_from_run_dir(p)
        if not results:
            raise FileNotFoundError(f"method CSV 없음: {p}")
        summary_path, _ = write_summary_csvs(p, results)
        print(f"summary.csv 생성: {summary_path}", flush=True)
        return summary_path
    raise FileNotFoundError(f"경로 없음: {p}")


def main() -> int:
    parser = argparse.ArgumentParser(description="qwen_smol 벤치 결과 그래프 생성")
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="run 폴더 또는 summary.csv (비우면 최신 qwen_smol_*/)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="PNG 저장 폴더 (기본: summary.csv 와 같은 run 폴더)",
    )
    args = parser.parse_args()

    if args.target.strip():
        csv_path = resolve_summary_csv(args.target)
    else:
        latest = find_latest_run_dir()
        if latest is None:
            print(f"[오류] {QWEN_SMOL_DIR} 에 qwen_smol_*/ 가 없습니다.", file=sys.stderr)
            return 1
        csv_path = resolve_summary_csv(str(latest))
        print(f"최신 실행 폴더: {csv_path.parent}", flush=True)

    plots_dir = Path(args.out_dir) if args.out_dir.strip() else None
    if plots_dir is None and csv_path.name == "summary.csv":
        # qwen_smol_YYYYMMDD_HHMMSS/ 등 run 폴더에 바로 저장
        plots_dir = csv_path.parent
    paths = generate_plots_from_summary_csv(csv_path, plots_dir=plots_dir)
    print(f"그래프 {len(paths)}개 저장:", flush=True)
    for path in paths:
        print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
