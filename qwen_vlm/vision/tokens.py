"""
Qwen3-VL 이미지 → smart_resize 이후 픽셀 수·(근사) 비전 토큰 수 계산.

공식 구현: QwenLM/Qwen3-VL 의 qwen_vl_utils/vision_process.py 의 smart_resize
(https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)

의존성 없음: ``uv run python -m qwen_vlm.vision.tokens --help`` (루트 ``qwen3_vl_image_tokens.py`` 래퍼 동일)

주의:
- 여기서의「토큰」은 (H×W) // (patch_size×merge_size)² 로 두는 **공간 merge 이후
  LLM 쪽에 대응하는 그리드 수**에 가깝습니다. (HF·공식 문서에서 쓰는 픽셀 예산과 동일한 factor)
- Transformers 파이프라인은 <|image_pad|> 개수 등으로 약간 다를 수 있으나,
  픽셀 예산·리사이즈 경계를 잡는 데는 이 값이면 충분한 경우가 많습니다.
- llama.cpp 는 GGUF(mmproj) 메타데이터의 image_min_pixels / image_max_pixels 를 쓰며,
  HF preprocessor 와 숫자가 다를 수 있습니다. 기본 프리셋 `vendor` 는 이 프로젝트의
  `vendor/.../mmproj-Qwen3VL-4B-Instruct-*.gguf` 조합에서 흔한 값(8192 ~ 4194304 px)입니다.
"""
from __future__ import annotations

import argparse
import math

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
# qwen-vl-utils 기본 (image_patch_size=16 일 때 factor=32)
UTILS_MIN_TOKEN = 4
UTILS_MAX_TOKEN = 16384

# Hugging Face Qwen/Qwen3-VL-4B-Instruct preprocessor_config.json
HF4B_MIN_PIXELS = 65536
HF4B_MAX_PIXELS = 16777216

# 이 저장소 vendor 의 Qwen3VL-4B-Instruct GGUF(mmproj) + llama.cpp 가 읽는 전형적 메타
# (서버 기동 시 로그: load_hparams: image_min_pixels / image_max_pixels)
VENDOR_QWEN3VL_4B_MIN_PIXELS = 8192
VENDOR_QWEN3VL_4B_MAX_PIXELS = 4194304


def round_by_factor(number: float, factor: int) -> int:
    return int(round(number / factor) * factor)


def ceil_by_factor(number: float, factor: int) -> int:
    return int(math.ceil(number / factor) * factor)


def floor_by_factor(number: float, factor: int) -> int:
    return int(math.floor(number / factor) * factor)


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be >= min_pixels")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"aspect ratio must be < {MAX_RATIO} (got {max(height, width) / min(height, width):.2f})"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def vision_grid_tokens(height: int, width: int, patch_size: int, merge_size: int) -> int:
    factor = patch_size * merge_size
    return (height * width) // (factor * factor)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-VL smart_resize + 비전 토큰(근사) 미리보기")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--merge-size", type=int, default=2)
    p.add_argument(
        "--preset",
        choices=("vendor", "utils", "hf-4b", "custom"),
        default="vendor",
        help="vendor=vendor/Qwen3VL-4B mmproj 메타(8192~4194304 px), hf-4b=HF preprocessor, utils=qwen-vl-utils 기본",
    )
    p.add_argument("--min-pixels", type=int, default=0, help="--preset custom 일 때 필수")
    p.add_argument("--max-pixels", type=int, default=0, help="--preset custom 일 때 필수")
    args = p.parse_args()

    patch = args.patch_size
    merge = args.merge_size
    factor = patch * merge

    if args.preset == "utils":
        min_px = UTILS_MIN_TOKEN * factor * factor
        max_px = UTILS_MAX_TOKEN * factor * factor
    elif args.preset == "hf-4b":
        min_px, max_px = HF4B_MIN_PIXELS, HF4B_MAX_PIXELS
    elif args.preset == "vendor":
        min_px, max_px = VENDOR_QWEN3VL_4B_MIN_PIXELS, VENDOR_QWEN3VL_4B_MAX_PIXELS
    else:
        if args.min_pixels <= 0 or args.max_pixels <= 0:
            p.error("--preset custom 은 --min-pixels, --max-pixels 가 필요합니다")
        min_px, max_px = args.min_pixels, args.max_pixels

    rh, rw = smart_resize(
        args.height,
        args.width,
        factor=factor,
        min_pixels=min_px,
        max_pixels=max_px,
    )
    tok = vision_grid_tokens(rh, rw, patch, merge)
    in_min = args.width * args.height >= min_px
    in_max = args.width * args.height <= max_px

    print(f"preset        : {args.preset}")
    print(f"factor        : {factor} (= patch {patch} × merge {merge})")
    print(f"min/max px    : {min_px:,} / {max_px:,}")
    print(f"입력 (W×H)   : {args.width} × {args.height} = {args.width * args.height:,} px")
    print(f"원본이 [min,max] 안? : {'예' if in_min and in_max else '아니오 (리사이즈 발생 가능)'}")
    print(f"리사이즈 후   : {rw} × {rh} = {rw * rh:,} px")
    print(f"비전 토큰(근사): {tok}  (= {rw * rh} // {factor*factor})")
    print()
    print("힌트: 이미지가 여러 장이면 보통 **각각** smart_resize 되어 토큰이 **합산**됩니다.")


if __name__ == "__main__":
    main()
