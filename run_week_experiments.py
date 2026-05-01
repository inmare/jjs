"""
저장소 루트에서의 호환 래퍼. 구현은 :mod:`qwen_vlm.run_week_experiments` 를 참고.

  uv run python run_week_experiments.py
  uv run python -m qwen_vlm.run_week_experiments
"""
from __future__ import annotations

from qwen_vlm.run_week_experiments import main

if __name__ == "__main__":
    main()
