#!/usr/bin/env python3
"""Ultralytics YOLO 가중치를 WEEK3_YOLO_WEIGHTS(기본 data/yolo_weights)에 받습니다."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_W3 = Path(__file__).resolve().parent
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))

from week3.config import DEFAULT_YOLO_MODELS, yolo_weights_dir  # noqa: E402

# 기본은 yolov8n.pt 1종만


def main() -> int:
    dest = yolo_weights_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {dest}", flush=True)

    from ultralytics import YOLO

    for name in DEFAULT_YOLO_MODELS:
        target = dest / name
        if target.is_file():
            print(f"  skip (exists): {target}", flush=True)
            continue
        print(f"  downloading {name}…", flush=True)
        model = YOLO(name)
        src = Path(getattr(model, "ckpt_path", None) or name)
        if not src.is_file():
            cwd_candidate = Path.cwd() / name
            if cwd_candidate.is_file():
                src = cwd_candidate
        if src.is_file() and src.resolve() != target.resolve():
            shutil.copy2(src, target)
        elif not target.is_file():
            # ultralytics 가 cwd 에 받은 경우
            raise FileNotFoundError(f"Could not locate downloaded weights for {name}")
        print(f"  -> {target}", flush=True)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
