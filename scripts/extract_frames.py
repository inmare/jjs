"""비디오에서 프레임을 이미지로 추출합니다. ffmpeg 가 PATH 에 있으면 우선 사용합니다."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def extract_ffmpeg(video: Path, out_dir: Path, fps: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)


def extract_opencv(video: Path, out_dir: Path, fps: float) -> None:
    if cv2 is None:
        raise RuntimeError(
            "opencv-python-headless 가 없습니다. uv sync --group dev 후 다시 시도하거나 ffmpeg 를 설치하세요."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(src_fps / fps))) if fps > 0 else 1
    idx = 0
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % interval == 0:
            path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            idx += 1
        frame_i += 1
    cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description="비디오 → JPG 프레임")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="출력 디렉터리")
    p.add_argument("--fps", type=float, default=1.0, help="초당 저장 프레임 수 (ffmpeg fps 필터)")
    args = p.parse_args()
    if not args.video.is_file():
        print(f"파일 없음: {args.video}", file=sys.stderr)
        sys.exit(1)
    if shutil.which("ffmpeg"):
        extract_ffmpeg(args.video, args.out, args.fps)
    else:
        extract_opencv(args.video, args.out, args.fps)
    n = len(list(args.out.glob("frame_*.jpg")))
    print(f"저장 완료: {n}장 -> {args.out}")


if __name__ == "__main__":
    main()
