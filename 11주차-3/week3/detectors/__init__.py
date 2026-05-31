"""Detection CNN backends (YOLO, Torchvision)."""
from week3.detectors.base import (
    DEFAULT_DETECTOR_SPECS,
    DetectorSpec,
    expand_legacy_yolo_args,
    load_detector_backend,
    parse_detector_spec,
    specs_from_strings,
)

__all__ = [
    "DEFAULT_DETECTOR_SPECS",
    "DetectorSpec",
    "expand_legacy_yolo_args",
    "load_detector_backend",
    "parse_detector_spec",
    "specs_from_strings",
]
