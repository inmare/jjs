"""ggml-org SmolVLM-256M-Instruct Q8 GGUF + mmproj 를 vendor/smolvlm-256m-q8/ 에 받습니다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "vendor" / "smolvlm-256m-q8"
REPO_ID = "ggml-org/SmolVLM-256M-Instruct-GGUF"
FILES = (
    "SmolVLM-256M-Instruct-Q8_0.gguf",
    "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
)


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[오류] huggingface_hub 필요: uv sync --group dev", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUT_DIR / name
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            print(f"건너뜀(이미 있음): {name}")
            continue
        print(f"다운로드: {REPO_ID} / {name}")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            local_dir=str(OUT_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"완료: {path}")
    print("SmolVLM-256M Q8 준비 완료:", OUT_DIR)


if __name__ == "__main__":
    main()
