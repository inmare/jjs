"""
호환 래퍼. 구현: :mod:`qwen_vlm.gui.hr_bench_app`.

  uv run python -m qwen_vlm.gui.hr_bench_app
"""
from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("qwen_vlm.gui.hr_bench_app", run_name="__main__")
