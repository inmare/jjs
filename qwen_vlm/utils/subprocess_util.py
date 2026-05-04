"""GUI·CLI에서 서브프로세스 출력 인코딩 통일."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable


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


def run_subprocess_stream_text_lines(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    append_line: Callable[[str], None],
) -> int:
    """
    자식 stdout/stderr 한 줄씩 ``append_line`` 으로 넘기며 대기.

    ``PYTHONUNBUFFERED=1`` 로 Python 자식의 ``print`` 도 가능한 한 즉시 흘린다.
    """
    merged = dict(env)
    merged.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def pump_out() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            append_line(line.rstrip("\r\n"))

    def pump_err() -> None:
        if proc.stderr is None:
            return
        append_line("[stderr]")
        for line in proc.stderr:
            append_line(line.rstrip("\r\n"))

    t_out = threading.Thread(target=pump_out, daemon=True)
    t_err = threading.Thread(target=pump_err, daemon=True)
    t_out.start()
    t_err.start()
    code = proc.wait()
    t_out.join(timeout=30.0)
    t_err.join(timeout=30.0)
    return int(code)
