"""
VLM 추론: llama-server + OpenAI 호환 API만 사용 (GGUF + mmproj).

기본값은 프로젝트 vendor 경로의 Qwen3-VL-4B Q8_0.

예:
  uv run python main.py --image demo.jpg --prompt "설명해줘"
  uv run python main.py --no-spawn --base-url http://127.0.0.1:8080/v1 --model my-vlm \\
    --image https://example.com/x.jpg
"""
from __future__ import annotations

import argparse
import base64
import io
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_LLAMA_SERVER = ROOT / "vendor" / "llama-cpp-win-cuda" / "llama-server.exe"
DEFAULT_GGUF = ROOT / "vendor" / "qwen3-vl-4b-q8-gguf" / "Qwen3VL-4B-Instruct-Q8_0.gguf"
DEFAULT_MMPROJ = ROOT / "vendor" / "qwen3-vl-4b-q8-gguf" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
DEFAULT_IMAGE = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
)


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def image_to_data_url(image: str | Path) -> tuple[str, str]:
    """로컬 경로 또는 http(s) URL → data URL. (data_url, 원본 설명 문자열)."""
    s = str(image).strip()
    if s.startswith("http://") or s.startswith("https://"):
        req = urllib.request.Request(
            s,
            headers={"User-Agent": "qwen-vlm-main/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        ctype = r.headers.get_content_type() if hasattr(r, "headers") else None
        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        jpeg = buf.getvalue()
        mime = ctype if ctype and ctype.startswith("image/") else "image/jpeg"
        b64 = base64.standard_b64encode(jpeg).decode("ascii")
        return f"data:{mime};base64,{b64}", s

    p = Path(s)
    if not p.is_file():
        raise FileNotFoundError(f"이미지 파일 없음: {p}")
    mime = _guess_mime(p)
    raw = p.read_bytes()
    if mime in ("image/jpeg", "image/jpg") or p.suffix.lower() in (".jpg", ".jpeg"):
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}", str(p)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", str(p)


def wait_for_server(base: str, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    url = f"{base.rstrip('/')}/models"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"llama-server 응답 없음: {url}")


def pil_to_data_url(img: Image.Image, *, quality: int = 92) -> str:
    """RGB 이미지 → JPEG data URL."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + b64


def chat_vlm(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    image_data_url: str,
    max_tokens: int,
) -> tuple[str, object | None]:
    return chat_vlm_multi(
        client=client,
        model=model,
        prompt=prompt,
        image_data_urls=[image_data_url],
        max_tokens=max_tokens,
    )


def chat_vlm_multi(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    image_data_urls: list[str],
    max_tokens: int,
) -> tuple[str, object | None]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    text = (resp.choices[0].message.content or "").strip()
    return text, resp.usage


def run_spawned_server(args: argparse.Namespace) -> int:
    llama = Path(args.llama_server)
    gguf = Path(args.gguf)
    mmproj = Path(args.mmproj)
    for label, p in ("llama-server", llama), ("GGUF", gguf), ("mmproj", mmproj):
        if not p.is_file():
            print(f"[오류] {label} 파일이 없습니다: {p}", file=sys.stderr)
            return 1

    cmd = [
        str(llama),
        "-m",
        str(gguf),
        "--mmproj",
        str(mmproj),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "-ngl",
        str(args.ngl),
        "-fa",
        args.flash_attn,
        "-a",
        args.model,
        "--ctx-size",
        str(args.ctx_size),
    ]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    api_base = f"http://{args.host}:{args.port}/v1"

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            cwd=str(llama.parent),
            stdout=subprocess.DEVNULL,
            stderr=log_f,
            creationflags=creationflags,
        )
        try:
            wait_for_server(api_base, args.server_timeout)
            load_s = time.perf_counter() - t0
            client = OpenAI(base_url=api_base, api_key=args.api_key)
            t1 = time.perf_counter()
            text, usage = chat_vlm(
                client=client,
                model=args.model,
                prompt=args.prompt,
                image_data_url=args._data_url,
                max_tokens=args.max_tokens,
            )
            gen_s = time.perf_counter() - t1
            _print_result(
                text=text,
                usage=usage,
                load_s=load_s,
                gen_s=gen_s,
                api_base=api_base,
            )
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


def run_external_server(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    client = OpenAI(base_url=base, api_key=args.api_key)
    t1 = time.perf_counter()
    text, usage = chat_vlm(
        client=client,
        model=args.model,
        prompt=args.prompt,
        image_data_url=args._data_url,
        max_tokens=args.max_tokens,
    )
    gen_s = time.perf_counter() - t1
    _print_result(
        text=text,
        usage=usage,
        load_s=0.0,
        gen_s=gen_s,
        api_base=base,
        external=True,
    )
    return 0


def _print_result(
    *,
    text: str,
    usage: object | None,
    load_s: float,
    gen_s: float,
    api_base: str,
    external: bool = False,
) -> None:
    n_in = n_out = 0
    if usage is not None:
        n_in = getattr(usage, "prompt_tokens", 0) or 0
        n_out = getattr(usage, "completion_tokens", 0) or 0
    print("=" * 60)
    print("[생성 텍스트]")
    print(text)
    print("=" * 60)
    print(f"API base     : {api_base}")
    if not external:
        print(f"서버 기동·로드: {load_s:.1f}s")
    print(f"생성 시간    : {gen_s:.1f}s")
    print(f"입력 토큰    : {n_in}")
    print(f"출력 토큰    : {n_out}")
    if gen_s > 0 and n_out > 0:
        print(f"속도         : {n_out / gen_s:.1f} tok/s")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VLM 추론 (llama-server + OpenAI API)")
    p.add_argument("--image", default=DEFAULT_IMAGE, help="이미지 경로 또는 URL")
    p.add_argument(
        "--prompt",
        default="이 사진에 대해 자세히 설명해줘. 한국어로 대답해줘.",
        help="질문 텍스트",
    )
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--model",
        default="qwen3-vl-4b-q8",
        help="API 모델 이름 (llama-server -a 와 동일)",
    )
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-local"))
    p.add_argument(
        "--no-spawn",
        action="store_true",
        help="이미 떠 있는 llama-server에만 연결",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLAMA_OPENAI_BASE", "http://127.0.0.1:8765/v1"),
        help="--no-spawn 시 베이스 URL (예: http://127.0.0.1:8080/v1)",
    )
    p.add_argument("--llama-server", default=str(DEFAULT_LLAMA_SERVER))
    p.add_argument("--gguf", default=str(DEFAULT_GGUF))
    p.add_argument("--mmproj", default=str(DEFAULT_MMPROJ))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("LLAMA_SERVER_PORT", "8765")))
    p.add_argument("--ngl", type=int, default=99, help="GPU 레이어 오프로드 개수")
    p.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    p.add_argument("--ctx-size", type=int, default=8192)
    p.add_argument(
        "--server-timeout",
        type=float,
        default=180.0,
        help="서버 준비 대기 최대 시간(초)",
    )
    p.add_argument(
        "--log-file",
        default=str(ROOT / "vendor" / "llama-server-vlm.log"),
        help="스폰 모드에서 llama-server stderr 로그",
    )
    return p


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        args._data_url, src = image_to_data_url(args.image)
    except (OSError, urllib.error.URLError) as e:
        print(f"[오류] 이미지 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"이미지 소스: {src}\n")

    if args.no_spawn:
        raise SystemExit(run_external_server(args))
    raise SystemExit(run_spawned_server(args))


if __name__ == "__main__":
    main()
