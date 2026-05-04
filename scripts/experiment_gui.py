"""
호환 래퍼. 구현: :mod:`qwen_vlm.gui.hr_bench_app`.

  uv run python scripts/experiment_gui.py
  uv run python -m qwen_vlm.gui.hr_bench_app

``HR-Bench 실행`` 은 하위에서 ``sys.executable -m qwen_vlm.cli.hr_bench`` 를 쓰므로,
이 스크립트로 GUI를 연 Python 환경과 동일한 패키지 코드가 적용된다(Smol 스왑 등).
"""
from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("qwen_vlm.gui.hr_bench_app", run_name="__main__")
