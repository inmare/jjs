"""11주차-3 실험 기본 설정 (11주차-2 벤치·11주차 crop 정책 반영)."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# 리포 루트 (11주차-3/ 의 부모)
REPO_ROOT = Path(__file__).resolve().parents[2]

# 11주차-2 와 동일한 4+1 벤치 (visualprobe 는 선택)
DEFAULT_BENCHMARKS: tuple[str, ...] = (
    "hrbench_4k",
    "mme_rw_lite",
    "treebench",
    "vstar",
)

OPTIONAL_BENCHMARKS: tuple[str, ...] = ("visualprobe",)

# 11주차 결론: THUMB 1280 + CROP8 (박스 면적 상한 ≈ 원본의 8%, 11주차 팀 권장)
THUMB_MAX_SIDE = 1280
CROP_AREA_NUMERATOR = 8
CROP_AREA_DENOMINATOR = 100

# YOLO 탐지 캔버스 (CNN_VLM V2 와 유사: 원본 대비 축소 후 탐지)
YOLO_CONTEXT_SCALE = 0.5

# YOLO: 기본 1종만 (바리에이션 축소)
DEFAULT_YOLO_MODELS: tuple[str, ...] = ("yolov8n.pt",)

# 탐지 입력 긴 변 상한 (YOLO·Torchvision 공통)
DEFAULT_YOLO_IMGSZ_MAX_LONG: tuple[int, ...] = (1536, 640)

# Torchvision COCO detectors (다양성 테스트)
DEFAULT_TORCHVISION_MODELS: tuple[str, ...] = (
    "fasterrcnn_mobilenet_v3_large_fpn",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
)

# VLM
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 8
# Qwen 추론 시 crop 상한 (0=무제한, RetinaNet 등 dense 탐지 OOM 방지)
DEFAULT_MAX_CROPS_VLM = 12


def resolve_yolo_weights_path(model_name: str) -> str:
    p = yolo_weights_dir() / model_name
    if p.is_file():
        return str(p)
    return model_name


def yolo_weights_dir() -> Path:
    raw = os.environ.get("WEEK3_YOLO_WEIGHTS", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT / "data" / "yolo_weights"


def results_dir() -> Path:
    raw = os.environ.get("WEEK3_RESULTS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "results"


# 결과 폴더: results/summary_YYYYMMDD_HHMMSS/{summary.csv, rows/, *.png}
SUMMARY_RUN_PREFIX = "summary"
SUMMARY_STAMP_FMT = "%Y%m%d_%H%M%S"
SUMMARY_CSV_FILENAME = "summary.csv"
SUMMARY_ROWS_SUBDIR = "rows"
SUMMARY_SAMPLES_SUBDIR = "samples"


def format_summary_stamp(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime(SUMMARY_STAMP_FMT)


def make_summary_run_dir(
    base: Path | None = None,
    stamp: str | None = None,
) -> Path:
    root = base if base is not None else results_dir()
    run_dir = root / f"{SUMMARY_RUN_PREFIX}_{stamp or format_summary_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_summary_run_paths(output: str = "") -> tuple[Path, Path, Path, Path]:
    """
    (run_dir, summary.csv 경로, rows/ 경로, samples/ 경로).

    ``output`` 비우면 ``results/summary_YYYYMMDD_HHMMSS/`` 를 새로 만든다.
    디렉터리를 주면 그 안에 저장. ``.csv`` 파일만 주면 타임스탬프 폴더를 새로 만든다.
    """
    if not output.strip():
        run_dir = make_summary_run_dir()
    else:
        p = Path(output).expanduser()
        if p.suffix.lower() == ".csv":
            if (
                p.name == SUMMARY_CSV_FILENAME
                and p.parent.name.startswith(f"{SUMMARY_RUN_PREFIX}_")
            ):
                run_dir = p.parent.resolve()
            else:
                run_dir = make_summary_run_dir()
        else:
            run_dir = p.resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

    rows_dir = run_dir / SUMMARY_ROWS_SUBDIR
    rows_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / SUMMARY_SAMPLES_SUBDIR
    samples_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / SUMMARY_CSV_FILENAME
    return run_dir, csv_path, rows_dir, samples_dir


def build_run_config(
    *,
    benchmarks: list[str],
    detector_methods: list[str],
    detect_only: bool,
    max_crops: int,
    max_samples: int,
    start: int,
    device: str,
) -> dict[str, object]:
    """실행 시점 설정 스냅샷 (run_config.json)."""
    return {
        "thumb_max_side": THUMB_MAX_SIDE,
        "crop_area_max_ratio": f"{CROP_AREA_NUMERATOR}/{CROP_AREA_DENOMINATOR}",
        "yolo_context_scale": YOLO_CONTEXT_SCALE,
        "default_max_crops_vlm": DEFAULT_MAX_CROPS_VLM,
        "max_crops_applied": max_crops,
        "detect_only": detect_only,
        "benchmarks": benchmarks,
        "detectors": detector_methods,
        "max_samples": max_samples,
        "start_index": start,
        "device": device,
        "qwen_model_id": DEFAULT_QWEN_MODEL_ID,
    }


def method_name_for_spec(spec: object) -> str:
    """DetectorSpec.method_name 과 동일 규칙."""
    from week3.detectors.base import DetectorSpec

    if not isinstance(spec, DetectorSpec):
        raise TypeError("expected DetectorSpec")
    isz = "native" if spec.imgsz_max_long <= 0 else str(spec.imgsz_max_long)
    if spec.backend == "yolo":
        stem = Path(spec.model_id).stem.replace(".", "_")
        tag = f"YOLO_{stem}"
    else:
        tag = f"TV_{spec.model_id}"
    return f"CNN_VLM_{tag}_ISZ{isz}_THUMB{THUMB_MAX_SIDE}_CROP{CROP_AREA_NUMERATOR}"


def method_name(*, yolo_model: str, imgsz_max_long: int) -> str:
    """레거시 호환."""
    from week3.detectors.base import DetectorSpec

    return method_name_for_spec(
        DetectorSpec("yolo", yolo_model, imgsz_max_long)
    )
