"""
호환 래퍼. 구현: :mod:`qwen_vlm.cli.open_html_webview`.

  uv run python -m qwen_vlm.cli.open_html_webview path/to/file.html
"""
from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("qwen_vlm.cli.open_html_webview", run_name="__main__")
