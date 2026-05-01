"""
주간 데모 실험 CLI·집계 심볼 호환 진입점.

구현은 :mod:`qwen_vlm.pipeline.week` 에 있습니다.

  uv run python -m qwen_vlm.run_week_experiments --help
"""
from __future__ import annotations

from qwen_vlm.pipeline.week import *  # noqa: F403

if __name__ == "__main__":
    main()
