"""
VLM 실험 파이프라인 CLI·심볼 호환 진입점.

구현은 :mod:`qwen_vlm.pipeline.experiment` 에 있습니다.

  uv run python -m qwen_vlm.experiment_pipeline bench --help
"""
from __future__ import annotations

from qwen_vlm.pipeline.experiment import *  # noqa: F403

if __name__ == "__main__":
    main()
