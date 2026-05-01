"""콘솔·자식 프로세스에서 한글 등 유니코드 출력 안정화."""
from __future__ import annotations

import sys


def configure_stdio_utf8() -> None:
    """stdout/stderr 를 UTF-8로 맞춘다 (Windows cp949 깨짐 완화)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except OSError:
                pass
