"""OpenAI 호환 llama-server 베이스 URL 정규화."""
from __future__ import annotations


def normalize_openai_base_url(url: str) -> str:
    """`/v1` 접미사가 없으면 붙인다."""
    base = (url or "").rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base
