"""데모용 짧은 MP4를 받아 연속 프레임 파이프라인을 바로 시험할 수 있게 합니다.

실제 ShanghaiTech / UCF-Crime 은 용량·링크 이슈로 여기서 자동 받지 않습니다.
공식 페이지에서 받아 data/datasets/ 아래에 두면 extract_frames.py 로 동일하게 처리됩니다.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "datasets" / "demo" / "sample-5s.mp4"
# 짧은 공개 샘플 MP4 (연속 프레임·추출 테스트용)
URL = "https://download.samplelib.com/mp4/sample-5s.mp4"


def main() -> None:
    out = DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        print(f"이미 있음: {out}")
        return
    print(f"다운로드: {URL}\n -> {out}")
    req = urllib.request.Request(URL, headers={"User-Agent": "qwen-vlm-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    out.write_bytes(data)
    print(f"완료 ({len(data) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
