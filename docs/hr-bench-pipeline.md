# HR-Bench — 네 가지 입력 전략 비교

[Hugging Face `DreamMr/HR-Bench`](https://huggingface.co/datasets/DreamMr/HR-Bench) 객관식을 동일한 문제 집합(인덱스)에 대해 여러 번 돌려, **정확도·평균 응답 시간·평균 Qwen 프롬프트 토큰·GPU 메모리 스냅샷**을 한 표로 비교합니다.

## 전략

| 이름 | 설명 |
|------|------|
| `qwen_only` | 저해상 1장(기본: `context_max_side`, 또는 `qwen_image_max_long_side` 지정)만 Qwen |
| `yolo_lowres_crops` | 저해상 전망 + YOLO 크롭(좌표만 서술, 클래스 라벨 없음) |
| `yolo_smol_parallel` | YOLO와 Smol **병렬**(스레드) 후 Qwen — 단일 GPU에서는 **한 포트**에서 Smol↔Qwen GGUF 순차 스왑 |
| `yolo_smol_sequential` | YOLO → Smol → Qwen **순차** (최종 Qwen 입력 구성은 병렬 전략과 동일) |

구현: `qwen_vlm/hr_bench/io.py`, `strategies.py`, `metrics.py`, `report.py` · CLI `qwen_vlm/cli/hr_bench.py`.

## YOLO → VLM 픽셀 예산 (기본 켜짐)

YOLO 탐지를 Qwen에 넣기 전 `qwen_vlm/vision/yolo_vlm_budget.py`에서 다음을 적용합니다(외부 프로파일링 스크립트와 동일한 의도).

- 원본 대비 **과대 박스** 제거(기본: 박스 면적이 원본의 1/4 이상이면 제외)
- **동일 좌표** 중복 제거, **다른 박스에 완전 포함**되는 박스 제거
- 저해상 전망 좌표계로 투영했을 때 **같아 보이는** 박스 중복 제거
- **(전망 w×h) + Σ(원본 박스 면적)** 이 원본 픽셀 수를 넘지 않도록, **작은 박스부터** 제거

끄려면 CLI에 `--disable-yolo-vlm-budget`, GUI에서 「YOLO→VLM 픽셀 예산·필터 끄기」를 선택합니다.

자세한 설명: [pipeline-yolo-crops-vlm.md](pipeline-yolo-crops-vlm.md).

## CLI

```bash
uv sync --group dev

# 기본: 127.0.0.1:8765 에서 llama-server 자동 기동(또는 재사용), Smol 전략은 같은 포트 GGUF 스왑
uv run python -m qwen_vlm.cli.hr_bench --max-samples 10 --strategies all

# Qwen + YOLO만
uv run python scripts/run_hr_bench.py --max-samples 10 \
  --strategies qwen_only,yolo_lowres_crops
```

주요 옵션:

- `--split` `hrbench_4k` | `hrbench_8k`
- `--max-samples`, `--sample-mode` sequential | random, `--start`, `--seed`
- `--context-max-side` 저해상 전망 긴 변
- `--crop-max-side` 크롭을 Qwen에 넣기 전 긴 변 상한
- `--max-crops`, `--min-crop-short-side`, `--min-crop-area`
- `--disable-yolo-vlm-budget` 위 픽셀 예산·필터 끄기
- `--qwen-image-max-long-side` (`qwen_only` 전용; `0`이면 `context_max_side`와 동일)
- `--omit-smol-if-unavailable` Smol GGUF/mmproj 없으면 Smol 전략만 생략
- `--json-out`, `--html-out`, `--png-out` (matplotlib 비교 차트)

실행 로그에는 **전략 번호**, **샘플 k/n 및 퍼센트**, **데이터셋 행·HR 인덱스**, **정답/오답(gt vs pred)** 가 순서대로 출력됩니다.

## GUI

```bash
uv sync
uv run python -m qwen_vlm.gui.hr_bench_app
```

체크박스로 전략 선택 후 실행합니다. 결과는 기본적으로 `docs/hr_bench_last.json`, `docs/hr_bench_last.html`, 선택 시 `docs/hr_bench_last_charts.png`.

- **HTML (내장 창)**: `pywebview` 로 별도 창에서 `file://` HTML 을 띄웁니다. Windows 에는 [WebView2 런타임](https://developer.microsoft.com/microsoft-edge/webview2/) 이 필요할 수 있습니다.
- **PNG 다른 이름 저장**: 실행 후 생성된 차트를 원하는 경로로 복사합니다.

## 서버·VRAM

- `yolo_smol_*` 는 Qwen + Smol + YOLO가 부하를 줄 수 있습니다. VRAM이 빠듯하면 전략을 나누어 실행하거나 `max_crops` / 해상도 캡을 낮추세요.
- Smol↔Qwen 스왑을 쓰려면 **8765 포트에 다른 llama-server가 떠 있으면 안 됩니다**(비어 있는 포트 필요).

## 연속 프레임·게이트 실험 (별도)

로컬 `frame_*.jpg` 주간 파이프라인은 `qwen_vlm/run_week_experiments.py`, `qwen_vlm/pipeline/experiment.py` 를 사용합니다. 요약은 [week-demo-pipeline.md](week-demo-pipeline.md), YOLO·크롭·예산 필터는 [pipeline-yolo-crops-vlm.md](pipeline-yolo-crops-vlm.md)를 참고하세요.
