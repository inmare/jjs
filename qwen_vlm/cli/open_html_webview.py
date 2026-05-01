"""
로컬 HTML 파일을 pywebview 창에서 연다.

  uv run python -m qwen_vlm.cli.open_html_webview path/to/file.html
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m qwen_vlm.cli.open_html_webview <file.html>", file=sys.stderr)
        return 1
    p = Path(sys.argv[1]).resolve()
    if not p.is_file():
        print(f"not a file: {p}", file=sys.stderr)
        return 1
    try:
        import webview
    except ImportError:
        print(
            "pywebview 가 필요합니다: uv sync",
            file=sys.stderr,
        )
        return 1
    webview.create_window(
        str(p.name),
        url=p.as_uri(),
        width=1280,
        height=900,
    )
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
