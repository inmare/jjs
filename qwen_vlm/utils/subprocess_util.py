"""GUI·CLI에서 서브프로세스 출력 인코딩 통일."""
from __future__ import annotations

import os
import sys


def child_env_for_utf8_stdio() -> dict[str, str]:
    """자식 Python이 stderr/stdout을 UTF-8로 쓰도록 환경 변수 설정."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    if sys.platform == "win32":
        env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
    return env


def decode_process_stdout_stderr(
    stdout: bytes | None,
    stderr: bytes | None,
) -> tuple[str, str]:
    """바이트 스트림을 텍스트로 복원. Windows에서 cp949 혼재 시 대비."""

    def dec(b: bytes) -> str:
        if not b:
            return ""
        for enc in ("utf-8", "utf-8-sig"):
            try:
                return b.decode(enc)
            except UnicodeDecodeError:
                continue
        if sys.platform == "win32":
            try:
                return b.decode("cp949")
            except UnicodeDecodeError:
                pass
        return b.decode("utf-8", errors="replace")

    return dec(stdout or b""), dec(stderr or b"")
