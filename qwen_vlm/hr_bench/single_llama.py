"""단일 ``llama-server`` 포트에서 Smol ↔ Qwen GGUF 를 순차 스왑 (GPU 1대)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from openai import OpenAI

from qwen_vlm.main import (
    openai_base_url,
    start_llama_server,
    wait_for_server,
)


@dataclass(frozen=True)
class SmolSingleServerHooks:
    """``run_strategy_on_indices`` 가 Smol 샘플마다 호출하는 훅."""

    before_smol_sample: Callable[[], None]
    """샘플 시작: Smol GGUF 로 서버 기동(이전 Qwen 프로세스 정리)."""
    after_smol_get_qwen_client: Callable[[], OpenAI]
    """Smol 호출 직후: Qwen GGUF 로 바꿔 기동하고 Qwen용 ``OpenAI`` 클라이언트 반환."""
    after_qwen_sample: Callable[[], None]
    """샘플 종료(성공·실패 무관): Qwen 프로세스 정리."""


class SingleLlamaSwapController:
    """
    동일 host:port 에서 Smol / Qwen 을 번갈아 띄운다.

    HR-Bench ``yolo_smol_*`` 샘플당: Smol 기동 → 스테이지1 → 종료 → Qwen 기동 → 스테이지2 → 종료.
    """

    def __init__(
        self,
        *,
        llama_server: str | Path,
        host: str,
        port: int,
        qwen_gguf: str | Path,
        qwen_mmproj: str | Path,
        qwen_model: str,
        smol_gguf: str | Path,
        smol_mmproj: str | Path,
        smol_model: str,
        ngl: int,
        flash_attn: str,
        ctx_size: int,
        qwen_log: Path,
        smol_log: Path,
        api_key: str,
        server_timeout: float,
    ) -> None:
        self._llama = llama_server
        self._host = host
        self._port = port
        self._qwen_g = qwen_gguf
        self._qwen_m = qwen_mmproj
        self._qwen_alias = qwen_model
        self._smol_g = smol_gguf
        self._smol_m = smol_mmproj
        self._smol_alias = smol_model
        self._ngl = ngl
        self._fa = flash_attn
        self._ctx = ctx_size
        self._qlog = qwen_log
        self._slog = smol_log
        self._api_key = api_key
        self._to = server_timeout
        self._proc: subprocess.Popen | None = None
        self._log_f: TextIO | None = None
        self._kind: str | None = None

    @property
    def base(self) -> str:
        return openai_base_url(self._host, self._port)

    def stop(self) -> None:
        """실행 중인 llama-server 가 있으면 종료하고 로그를 닫는다."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=25.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._log_f is not None:
            try:
                self._log_f.close()
            except OSError:
                pass
            self._log_f = None
        self._kind = None

    def start_smol(self) -> None:
        """Smol VLM(GGUF+mmproj)으로 서버를 띄운 뒤 ``/models`` 가 뜰 때까지 대기."""
        self.stop()
        self._proc, _, self._log_f = start_llama_server(
            llama_server=self._llama,
            gguf=self._smol_g,
            mmproj=self._smol_m,
            host=self._host,
            port=self._port,
            ngl=self._ngl,
            flash_attn=self._fa,
            model=self._smol_alias,
            ctx_size=self._ctx,
            log_file=self._slog,
        )
        wait_for_server(self.base, self._to)
        self._kind = "smol"

    def start_qwen(self) -> None:
        """Qwen3-VL(GGUF+mmproj)으로 서버를 띄운 뒤 대기."""
        self.stop()
        self._proc, _, self._log_f = start_llama_server(
            llama_server=self._llama,
            gguf=self._qwen_g,
            mmproj=self._qwen_m,
            host=self._host,
            port=self._port,
            ngl=self._ngl,
            flash_attn=self._fa,
            model=self._qwen_alias,
            ctx_size=self._ctx,
            log_file=self._qlog,
        )
        wait_for_server(self.base, self._to)
        self._kind = "qwen"

    def ensure_qwen(self) -> None:
        """Qwen GGUF 가 이미 떠 있으면 재사용, 아니면(또는 Smol 중이면) Qwen 으로 맞춘다."""
        if (
            self._kind == "qwen"
            and self._proc is not None
            and self._proc.poll() is None
        ):
            return
        self.start_qwen()

    def hooks(self) -> SmolSingleServerHooks:
        """``run_strategy_on_indices`` 에 넘길 훅 묶음."""
        ctrl = self

        def before() -> None:
            ctrl.start_smol()

        def after_smol() -> OpenAI:
            ctrl.start_qwen()
            return OpenAI(base_url=ctrl.base, api_key=ctrl._api_key)

        def after_sample() -> None:
            ctrl.stop()

        return SmolSingleServerHooks(before, after_smol, after_sample)
