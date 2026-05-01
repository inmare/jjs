"""
호환 래퍼. 구현: :mod:`qwen_vlm.cli.analyze_experiment_results`.

  uv run python -m qwen_vlm.cli.analyze_experiment_results
"""
from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("qwen_vlm.cli.analyze_experiment_results", run_name="__main__")
