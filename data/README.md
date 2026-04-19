# 데이터 디렉터리

대용량 영상·추출 프레임은 Git에 포함하지 않습니다. 실제 파일은 `data/datasets/` 아래에 두며, 이 경로는 `.gitignore`로 제외됩니다.

## 권장 구조

| 경로 | 용도 |
|------|------|
| `data/datasets/demo/` | 스크립트로 받은 짧은 데모 클립(연속 프레임 실험용) |
| `data/datasets/shanghaitech/` | [ShanghaiTech Campus](https://svip-lab.github.io/dataset/campus_dataset.html) 압축 해제본 |
| `data/datasets/ucf-crime/` | [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) 영상(예: Dropbox 분할 다운로드) |

## 빠른 시작

```bash
# 데모용 짧은 MP4 다운로드 + 프레임 추출 (1 fps)
uv run python scripts/fetch_demo_video.py
uv run python scripts/extract_frames.py --video data/datasets/demo/sample-5s.mp4 --out data/datasets/demo/frames --fps 1
```

실제 감시 데이터셋은 공식 페이지에서 받은 뒤 `--video`에 해당 `.mp4` / `.avi` 경로를 지정하면 됩니다.
