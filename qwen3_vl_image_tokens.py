"""
저장소 루트 호환 래퍼. 구현은 :mod:`qwen_vlm.vision.tokens`.

  uv run python qwen3_vl_image_tokens.py --help
  uv run python -m qwen_vlm.vision.tokens --help
"""
from __future__ import annotations

from qwen_vlm.vision.tokens import main

if __name__ == "__main__":
    main()
