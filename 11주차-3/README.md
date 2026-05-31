# 11주차-3 — Detection CNN(YOLO) 비교 (todo 1번)

**11주차-2**와 같은 벤치(`hrbench_4k`, `mme_rw_lite`, `treebench`, `vstar`)에서,  
**11주차**에서 정한 VLM 입력(`THUMB1280` + `CROP8`)은 고정한 채 **Detection CNN(YOLO·Torchvision)·탐지 해상도(imgsz)** 를 바꿔 정확도·시간·토큰·메모리를 측정합니다.

## 빠른 시작

```powershell
cd D:\Programming\jjs
. .\11주차-3\setup_storage.ps1   # C: 캐시 대신 프로젝트 디스크 사용
uv sync --group dev
uv run python 11주차-3/download_yolo_models.py
```

### 1) YOLO만 스윕 (VLM 없음, 빠름·디스크 절약)

```powershell
uv run python 11주차-3/run_benchmark.py `
  --detect-only `
  --benchmarks hrbench_4k `
  --max-samples 50
```

### 2) 전체 CNN_VLM (Qwen3-VL-4B, GPU 필요)

```powershell
uv run python 11주차-3/run_benchmark.py `
  --benchmarks hrbench_4k `
  --max-samples 20
```

결과 (실행마다 한 폴더):

- `11주차-3/results/summary_YYYYMMDD_HHMMSS/summary.csv`
- `run_config.json` — CROP8·crop 상한·벤치 목록 등 실행 설정
- `rows/` — 벤치×탐지기 요약 JSON
- `samples/` — 샘플별 `*.jsonl` (정답·모델 응답·crop 수·탐지 요약)
- 같은 폴더: `summary_table.md`, `*_summary_panel.png`, `detector_sweep_*.png`

벤치만 돌린 뒤 그래프만 다시 그리기:

```powershell
uv run python 11주차-3/plot_results.py
# 또는
uv run python 11주차-3/plot_results.py 11주차-3/results/summary_20260528_073435/summary.csv
```

## 탐지 CNN (기본 5조합)

| backend | model | imgsz |
|---------|-------|-------|
| yolo | `yolov8n.pt` | 1536, 640 |
| torchvision | `fasterrcnn_mobilenet_v3_large_fpn` | 1536 |
| torchvision | `retinanet_resnet50_fpn` | 1536 |
| torchvision | `ssd300_vgg16` | 1536 |

`--detectors` 로 개별 지정 가능. `--detect-only` 없으면 **Qwen**까지 실행.

## C: 드라이브 공간 부족

프로젝트는 이미 `D:\Programming\jjs`에 있어도 **uv / Hugging Face / torch** 캐시는 기본적으로 **사용자 프로필(C:)** 에 쌓입니다.

| 조치 | 설정 |
|------|------|
| **매 세션** | `.\11주차-3\setup_storage.ps1` 실행 후 `uv`·벤치 실행 |
| **영구** | Windows 환경 변수에 `.env.example` 값 등록 |
| **정리** | `uv cache clean` · `D:\Programming\jjs\.cache` 삭제(재다운로드 필요) |
| **가중치** | `WEEK3_YOLO_WEIGHTS` → `data/yolo_weights` (git 제외 `*.pt`) |

추가로 Windows **디스크 정리** → “이전 Windows 설치”, `pip`/`uv` 캐시, 사용하지 않는 conda 환경을 확인하세요.

## 11주차 / 11주차-2 와 관계

- **11주차-2**: `CNN_VLM_V2` + 썸네일 픽셀 스윕 (VLM 쪽)
- **11주차**: crop threshold / GRID (2번) — 본 스크립트는 **CROP8 고정** (`config.py`에서 변경 가능)
- **11주차-3 (본 폴더)**: **Detection CNN** 스윕 (1번)
