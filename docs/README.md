# 문서 색인 (`docs/`)

저장소 루트: [README.md](../README.md). **설치·명령어 입문**(pip, PyTorch GPU, CLI 예시): [사용법.md](../사용법.md) · Colab: [사용법-Colab.md](../사용법-Colab.md).


## 실험 GUI (벤치 중심)

대부분의 측정·비교는 **`uv run python scripts/experiment_gui.py`**(또는 `uv run python -m qwen_vlm.gui.hr_bench_app`)로 합니다. 한 창에서 다음을 수행합니다.

- **HR-Bench**: `qwen_vlm.cli.hr_bench`를 백그라운드 실행 → 아래 `hr_bench_last.*`
- **연속 프레임 비교**: `qwen_vlm.pipeline.experiment sequence-compare` → 아래 `sequence_compare_last.*`

로그는 GUI 내 콘솔에 스트리밍되고, 리포트는 기본적으로 이 폴더에 덮어씁니다.

| 문서 | 내용 |
|------|------|
| **[hr-bench-pipeline.md](hr-bench-pipeline.md)** | HR-Bench 4전략, YOLO→VLM 픽셀 예산, **GUI/CLI** 절차 |
| [week-demo-pipeline.md](week-demo-pipeline.md) | 연속 프레임 데이터·YOLO 프리셋·**GUI 탭**·`run_week_experiments`·`experiment_pipeline` |
| [pipeline-yolo-crops-vlm.md](pipeline-yolo-crops-vlm.md) | YOLO 크롭·좌표 프롬프트·게이트 |
| [qwen3-vl-image-tokens.md](qwen3-vl-image-tokens.md) | Qwen3-VL 비전 토큰·그리드 근사 |
| [realtime-vlm-input-strategies.md](realtime-vlm-input-strategies.md) | 실시간 VLM 입력 전략 |

### 결과 파일 (GUI 기본 출력)

| 산출물 | 설명 |
|--------|------|
| `hr_bench_last.json` / `.html` / `_charts.png`(선택) | HR-Bench 비교 표·Chart.js HTML·matplotlib PNG |
| `sequence_compare_last.json` / `.html` | 연속 프레임 `sequence-compare` 요약 |
| `experiment_results_last.html` (+ 동명 `.json`) | 주간/배치 `run_week_experiments` 등이 채우는 경우(자동 기록 블록은 [week-demo-pipeline.md](week-demo-pipeline.md) 참고) |
