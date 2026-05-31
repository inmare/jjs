"""11주차-2 와 동일 벤치마크 로더 (ref/9주차 BENCHMARK_CONFIGS 기준)."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "hrbench_4k": {
        "direct_parquet_url": (
            "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.parquet"
        ),
        "type": "mcq",
    },
    "vstar": {
        "dataset_candidates": [
            ("craigwu/vstar_bench", None, "test"),
            ("xjtupanda/VStar_Bench", None, "train"),
        ],
        "type": "mcq",
    },
    "mme_rw_lite": {
        "dataset_candidates": [
            ("yifanzhang114/MME-RealWorld-lite-lmms-eval", None, "test"),
            ("yifanzhang114/MME-RealWorld-lite-lmms-eval", None, "train"),
            ("yifanzhang114/MME-RealWorld-Lite", None, "train"),
            ("yifanzhang114/MME-RealWorld-Lite", None, "test"),
        ],
        "type": "mixed",
    },
    "treebench": {
        "dataset_candidates": [
            ("HaochenWang/TreeBench", None, "train"),
            ("HaochenWang/TreeBench", None, "test"),
        ],
        "type": "mcq",
    },
    "visualprobe": {
        "dataset_candidates": [
            ("xjtupanda/VisualProbe_Hard", None, "test"),
            ("Mini-o3/VisualProbe_Hard", None, "test"),
        ],
        "type": "freeform",
    },
}

# load_benchmark_dataset 성공 시 HF repo id (상대 경로 이미지용)
_CURRENT_DATASET_ID: str | None = None


def load_benchmark_dataset(benchmark_name: str) -> Any:
    global _CURRENT_DATASET_ID
    from datasets import load_dataset

    if benchmark_name not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    cfg = BENCHMARK_CONFIGS[benchmark_name]

    if benchmark_name == "hrbench_4k":
        print("Loading HR-Bench 4K from parquet…", flush=True)
        return load_dataset(
            "parquet",
            data_files={"data": cfg["direct_parquet_url"]},
            split="data",
        )

    errors: list[str] = []
    for dataset_id, config_name, split in cfg["dataset_candidates"]:
        try:
            print(
                f"Trying dataset: {dataset_id}, config={config_name}, split={split}",
                flush=True,
            )
            if config_name is None:
                ds = load_dataset(dataset_id, split=split)
            else:
                ds = load_dataset(dataset_id, config_name, split=split)
            print(f"Loaded: {dataset_id} / {split}, rows={len(ds)}", flush=True)
            _CURRENT_DATASET_ID = dataset_id
            return ds
        except Exception as exc:
            errors.append(f"{dataset_id}/{split}: {exc!r}")

    raise RuntimeError("All dataset candidates failed:\n" + "\n".join(errors))


def load_image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            raw = value["bytes"]
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if value.get("path") not in (None, "", "dummy/path"):
            return load_image_from_value(value["path"])
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, str):
        text_value = value.strip()
        local_path = Path(text_value).expanduser()
        if local_path.is_file():
            return Image.open(local_path).convert("RGB")
        if _CURRENT_DATASET_ID is not None and text_value and not text_value.startswith(
            ("http://", "https://", "data:")
        ):
            from huggingface_hub import hf_hub_download, list_repo_files

            cleaned = text_value.replace("\\", "/").lstrip("./")
            basename = Path(cleaned).name
            stem = Path(cleaned).stem
            for filename in (
                cleaned,
                basename,
                f"images/{cleaned}",
                f"image/{cleaned}",
                f"imgs/{cleaned}",
                f"data/{cleaned}",
                f"test/{cleaned}",
                f"VStarBench/{cleaned}",
            ):
                try:
                    local_file = hf_hub_download(
                        repo_id=_CURRENT_DATASET_ID,
                        filename=filename,
                        repo_type="dataset",
                    )
                    return Image.open(local_file).convert("RGB")
                except Exception:
                    continue
            try:
                repo_files = list_repo_files(
                    repo_id=_CURRENT_DATASET_ID, repo_type="dataset"
                )
                image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                matched = [
                    f
                    for f in repo_files
                    if f.lower().endswith(image_exts)
                    and (
                        f.replace("\\", "/") == cleaned
                        or f.replace("\\", "/").endswith("/" + cleaned)
                        or Path(f).name == basename
                        or Path(f).stem == stem
                    )
                ]
                if matched:
                    local_file = hf_hub_download(
                        repo_id=_CURRENT_DATASET_ID,
                        filename=matched[0],
                        repo_type="dataset",
                    )
                    return Image.open(local_file).convert("RGB")
            except Exception:
                pass
    raise TypeError(f"Unsupported image type: {type(value)}")


def extract_image(example: dict[str, Any]) -> Image.Image | None:
    from qwen_vlm.hr_bench.io import to_pil

    for key in ("image", "images", "img", "picture"):
        if key in example and example[key] is not None:
            try:
                return to_pil(example[key])
            except Exception:
                try:
                    return load_image_from_value(example[key])
                except Exception:
                    pass
    if "bytes" in example:
        try:
            return to_pil(example["bytes"])
        except Exception:
            try:
                return load_image_from_value(example["bytes"])
            except Exception:
                pass
    return None


def extract_options(example: dict[str, Any]) -> list[str]:
    options: list[str] = []
    for letter in "ABCDEFGHIJK":
        for key in (letter, letter.lower(), f"option_{letter}", f"option_{letter.lower()}"):
            if key in example and example[key] is not None:
                val = str(example[key]).strip()
                if val and val.lower() != "nan":
                    options.append(f"{letter}. {val}")
                break
    return options


def extract_question(example: dict[str, Any]) -> str:
    q = str(
        example.get("question")
        or example.get("text")
        or example.get("query")
        or example.get("prompt")
        or ""
    ).strip()
    opts = extract_options(example)
    if opts and "Options:" not in q and not re.search(r"\bA[\.\)]", q):
        q = q + "\n\nOptions:\n" + "\n".join(opts)
    return q


def extract_gt_answer(example: dict[str, Any]) -> str:
    for key in ("answer", "gt_answer", "label", "correct_answer"):
        if key in example and example[key] is not None:
            return str(example[key]).strip()
    return ""


def is_multiple_choice(question: str) -> bool:
    if not question:
        return False
    return bool(re.search(r"\bA[\.\)]", question)) or "Options:" in question


def normalize_answer_letter(text: Any) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    upper = raw.upper()
    m = re.search(r"\b([A-K])\b", upper)
    if m:
        return m.group(1)
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < 11:
            return chr(ord("A") + idx)
    return upper[:1]


def parse_predicted_answer(text: str) -> str:
    if not text:
        return ""
    upper = str(text).upper()
    for pattern in (
        r"answer\s*[:：]\s*([A-K])",
        r"정답\s*[:：]\s*([A-K])",
        r"\(([A-K])\)",
        r"\b([A-K])\b",
    ):
        m = re.search(pattern, upper)
        if m:
            return m.group(1)
    return ""


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9가-힣\s\.\-_/]", "", t)
    return t.strip()


def compute_correctness(
    response: str, gt_answer: str, question: str
) -> tuple[str, int | None]:
    if not gt_answer:
        return "", None
    if is_multiple_choice(question):
        pred = parse_predicted_answer(response)
        gt = normalize_answer_letter(gt_answer)
        return pred, int(pred == gt) if pred else 0
    pred_norm = normalize_text(response)
    raw = gt_answer.strip()
    try:
        parsed = json.loads(raw)
        gt_candidates = [str(x) for x in parsed] if isinstance(parsed, list) else [raw]
    except Exception:
        if "|" in raw:
            gt_candidates = [x.strip() for x in raw.split("|")]
        elif ";" in raw:
            gt_candidates = [x.strip() for x in raw.split(";")]
        else:
            gt_candidates = [raw]
    gt_norms = [normalize_text(x) for x in gt_candidates if normalize_text(x)]
    if not pred_norm or not gt_norms:
        return response.strip()[:80], None
    ok = int(any(pred_norm == g or g in pred_norm for g in gt_norms))
    return response.strip()[:80], ok
