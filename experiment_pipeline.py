"""
저장소 루트에서의 호환 래퍼. 구현은 :mod:`qwen_vlm.experiment_pipeline` 를 참고.

  uv run python experiment_pipeline.py bench --help
  uv run python -m qwen_vlm.experiment_pipeline --help
"""
from __future__ import annotations

from qwen_vlm.experiment_pipeline import main

if __name__ == "__main__":
    main()
