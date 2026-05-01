"""
저장소 루트에서의 호환 래퍼. 구현은 :mod:`qwen_vlm.main` 를 참고.

  uv run python main.py --help
  uv run python -m qwen_vlm.main --help
"""
from __future__ import annotations

from qwen_vlm.main import main

if __name__ == "__main__":
    main()
