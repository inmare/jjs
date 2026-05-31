#!/usr/bin/env python3
"""summary CSV → PNG 그래프 (11주차 스타일)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_W3 = Path(__file__).resolve().parent
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))

from week3.config import results_dir  # noqa: E402
from week3.plots import generate_plots_from_summary_csv  # noqa: E402


def find_latest_summary_csv(root: Path) -> Path | None:
    nested = sorted(
        root.glob("summary_*/summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if nested:
        return nested[-1]
    flat = sorted(
        root.glob("summary_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return flat[-1] if flat else None


def main() -> int:
    p = argparse.ArgumentParser(description="11주차-3 벤치 결과 그래프 생성")
    p.add_argument(
        "csv",
        nargs="?",
        default="",
        help="summary.csv 또는 summary_*/summary.csv (비우면 최신 실행 폴더)",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="PNG 저장 폴더 (기본: CSV 가 있는 summary_YYYYMMDD_HHMMSS/)",
    )
    args = p.parse_args()

    if args.csv.strip():
        csv_path = Path(args.csv)
    else:
        results = results_dir()
        latest = find_latest_summary_csv(results)
        if latest is None:
            print(f"[오류] {results} 에 summary_*/summary.csv 가 없습니다.", file=sys.stderr)
            return 1
        csv_path = latest
        print(f"최신 실행 폴더: {csv_path.parent}", flush=True)

    if not csv_path.is_file():
        print(f"[오류] 파일 없음: {csv_path}", file=sys.stderr)
        return 1

    plots_dir = Path(args.out_dir) if args.out_dir.strip() else None
    paths = generate_plots_from_summary_csv(csv_path, plots_dir=plots_dir)
    print(f"그래프 {len(paths)}개 저장:", flush=True)
    for path in paths:
        print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
