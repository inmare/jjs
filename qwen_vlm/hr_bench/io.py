"""HR-Bench(HuggingFace) 로드, 이미지 디코딩, MCQ 프롬프트, 인덱스 선택."""
from __future__ import annotations

import base64
import io
import os
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image

from qwen_vlm.utils.image_resize import resize_max_long

HR_BENCH_ID = "DreamMr/HR-Bench"
HR_CONFIG = "hrbench_version_split"
SPLITS = ("hrbench_4k", "hrbench_8k")

# llama-server 등록 별칭 · OpenAI 호환 ``model`` (HR-Bench CLI 고정)
HR_BENCH_MODEL_QWEN = "qwen"
HR_BENCH_MODEL_SMOL = "smolvlm"

# HR-Bench GUI/CLI 고정: 포트·YOLO Ultralytics 가중치 (환경설정 없음)
HR_BENCH_LLAMA_OPENAI_BASE = "http://127.0.0.1:8765/v1"
HR_BENCH_YOLO_WEIGHTS = "yolo26n.pt"

MCQ_PROMPT = """Answer the following multiple choice question for the given image.
Reply with a single line containing ONLY the letter: A, B, C, or D (no other text).

{question}
A) {A}
B) {B}
C) {C}
D) {D}"""


def to_pil(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        # datasets 가 행 간 동일 디코드 버퍼를 가리키는 PIL 을 줄 때 `.copy()` 만으로는
        # 픽셀이 이전 행과 공유되는 사례가 있어 RGB 바이트를 새 버퍼로 복사한다.
        rgb = image_field.convert("RGB")
        return Image.frombytes("RGB", rgb.size, rgb.tobytes())
    if isinstance(image_field, dict) and "bytes" in image_field:
        raw = image_field["bytes"]
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        elif isinstance(raw, bytearray):
            raw = bytes(raw)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(image_field, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_field)).convert("RGB")
    if isinstance(image_field, str):
        s = image_field.strip()
        if not s:
            raise ValueError("빈 image 문자열")
        if s.startswith("data:") and "base64," in s:
            b64 = s.split("base64,", 1)[1]
            raw = base64.standard_b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if s.startswith(("http://", "https://")):
            req = urllib.request.Request(
                s, headers={"User-Agent": "qwen-vlm-hr-bench/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        p = Path(os.path.expanduser(s))
        if p.is_file():
            return Image.open(p).convert("RGB")
        try:
            raw = base64.b64decode(s, validate=False)
        except Exception:
            raw = b""
        if len(raw) >= 8 and (
            raw[:3] == b"\xff\xd8\xff"
            or raw[:8] == b"\x89PNG\r\n\x1a\n"
            or raw[:6] in (b"GIF87a", b"GIF89a")
            or raw[:4] == b"RIFF"
        ):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if len(s) < 1024:
            try:
                from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

                local = hf_hub_download(
                    repo_id=HR_BENCH_ID,
                    filename=s.replace("\\", "/").lstrip("/"),
                    repo_type="dataset",
                )
                return Image.open(local).convert("RGB")
            except Exception:
                pass
        raise TypeError(
            f"image 문자열을 해석할 수 없습니다(로컬 경로·http(s)·base64). "
            f"앞부분: {s[:80]!r}…"
        )
    raise TypeError(f"Unsupported image type: {type(image_field)}")


def load_hr_split(split: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as e:
        print("[오류] `pip install datasets` 필요: uv sync --group dev", file=sys.stderr)
        raise SystemExit(1) from e
    if split not in SPLITS:
        print(f"[오류] split 은 {SPLITS} 중 하나: {split!r}", file=sys.stderr)
        raise SystemExit(1)
    return load_dataset(HR_BENCH_ID, HR_CONFIG, split=split)


def norm_gt(answer: str | None) -> str | None:
    if not answer:
        return None
    s = str(answer).strip().upper()
    m = re.match(r"^([ABCD])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", s)
    return m.group(1) if m else None


def parse_pred(text: str) -> str | None:
    t = (text or "").strip().upper()
    if not t:
        return None
    m = re.search(r"^\s*([ABCD])\s*$", t, re.MULTILINE)
    if m:
        return m.group(1)
    m = re.search(r"^\s*([ABCD])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\bANSWER\s*[:=]\s*([ABCD])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", t)
    return m.group(1) if m else None


def select_indices(
    *,
    n: int,
    max_samples: int,
    sample_mode: str,
    start: int,
    seed: int | None,
) -> list[int]:
    if sample_mode == "random":
        k = n if max_samples == 0 else min(n, max(0, max_samples))
        if k <= 0:
            return []
        rng = random.Random(seed)
        if k == n:
            indices = list(range(n))
            rng.shuffle(indices)
        else:
            indices = rng.sample(range(n), k)
        return indices
    end = n if max_samples == 0 else min(n, start + max(0, max_samples))
    if end <= start:
        return []
    return list(range(start, end))


def snapshot_hr_dataset_row(ds: Any, i: int) -> dict[str, Any]:
    """``ds[i]`` 한 건을 컬럼 단위로 새 dict 로 복사한다.

    HuggingFace ``datasets`` 가 반환하는 행 객체·내부 버퍼 재사용 때문에
    다음 인덱스를 읽으면 이전 ``row['question']`` 등이 바뀌는 사례가 있어,
    MCQ·정답 문자열이 샘플 간 뒤섞이지 않게 스냅샷을 만든다.
    ``image`` 컬럼은 즉시 :func:`to_pil` 로 고정해 행 간 이미지 버퍼가 섞이지 않게 한다.
    """
    row = ds[i]
    names = getattr(ds, "column_names", None)
    if names is None:
        if isinstance(row, dict):
            out = dict(row)
        else:
            out = {str(k): row[k] for k in row}  # type: ignore[misc]
    else:
        out = {c: row[c] for c in names}
    if out.get("image") is not None:
        out["image"] = to_pil(out["image"])
    return out


def format_mcq_prompt(row: Mapping[str, Any]) -> str:
    def _tx(k: str) -> str:
        v = row.get(k)
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)

    return MCQ_PROMPT.format(
        question=_tx("question"),
        A=_tx("A"),
        B=_tx("B"),
        C=_tx("C"),
        D=_tx("D"),
    )


LogFn = Callable[[str], None]


def noop_log(_: str) -> None:
    pass
