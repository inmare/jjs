# 목표

- Qwen 계열 VLM으로 이미지 이해·설명 실험
- (향후) 앞단에 가벼운 detection 모델을 붙이거나, 고화질/크롭 이미지 등 조건별 벤치마크
- (향후) **MMBench** 등 공개 멀티모달 벤치마크(정답/선다형)로 VLM 품질을 측정(감시 데모 프레임 실험과는 목적이 다름)
- (향후) 다른 VLM/GGUF와 **동일 API 패턴**(OpenAI 호환)으로 속도·품질 비교

**처음 설치·테스트하는 분**은 아래를 권장합니다.

| 환경 | 문서 |
|------|------|
| 로컬 PC (pip·가상환경·CUDA PyTorch 안내가 필요하면) | [사용법.md](사용법.md) |
| Google Colab (브라우저만, 데스크톱 GUI 없음) | [사용법-Colab.md](사용법-Colab.md) · [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb) |

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inmare/jjs/blob/main/notebooks/colab_quickstart.ipynb)

자세한 옵션·파이프라인은 계속해서 [docs/README.md](docs/README.md)와 아래 각 절을 보시면 됩니다.

---

## 환경

- Python 3.12
- OS: Windows 11 (기본 경로는 Windows용 `llama-server.exe` 가정)
- GPU: NVIDIA + [llama.cpp CUDA 빌드](https://github.com/ggml-org/llama.cpp/releases) 권장 (CPU 빌드도 가능하나 매우 느림)

### 저장소 구조

| 경로 | 역할 |
|------|------|
| `qwen_vlm/` | VLM·벤치·주간 실험·YOLO·프레임 게이트 **구현** (`pip`/editable 설치 대상) |
| `scripts/` | 데모 영상·프레임 추출·**실험 GUI 래퍼**(`experiment_gui.py`) 등 실행 스크립트 |
| `docs/` | 절차·파이프라인 문서와 **실험 결과** JSON/HTML(HR-Bench·연속 프레임 요약 등) |
| `notebooks/` | Colab 등용 예시 노트북 (`colab_quickstart.ipynb`) |
| `data/` | 프레임·데이터셋(대용량은 git 제외) |
| 루트 `main.py`, `experiment_pipeline.py`, `run_week_experiments.py` 등 | `qwen_vlm`을 호출하는 **호환 래퍼** (`uv run python …` 유지) |

---

## 빠른 시작: 실험 GUI (권장)

> 전역 설치·PyTorch CUDA·CLI 입문식 설명은 [사용법.md](사용법.md). Colab은 [사용법-Colab.md](사용법-Colab.md).

대부분의 벤치마크와 비교 실행은 **`scripts/experiment_gui.py`** 한 창에서 합니다(pywebview + 로컬 HTML 대시보드). 내부 구현은 `qwen_vlm.gui.hr_bench_app`이며, GUI가 띄우는 벤치는 **`sys.executable -m …`** 이라 현재 활성화된 Python·패키지와 동일하게 동작합니다(Smol 교체 규칙 등).

### 의존성

- 기본 번들(`uv sync`): `openai`, `pillow`, `pywebview`(GUI 표시)
- **HR-Bench**·YOLO 크롭·연속 프레임 벤치: `ultralytics` 등이 필요하므로 **`uv sync --group dev`** 권장

```bash
uv sync --group dev
uv run python scripts/experiment_gui.py
```

동일 진입점: `uv run python -m qwen_vlm.gui.hr_bench_app`

### GUI에서 할 수 있는 일

| 탭 | 하는 일 | 하위 CLI(백그라운드) | 기본 산출물 (`docs/`) |
|----|---------|----------------------|------------------------|
| **HR-Bench** | DreamMr/HR-Bench 객관식, 4전략 정확도·시간·토큰·VRAM 비교 | `qwen_vlm.cli.hr_bench` | `hr_bench_last.json`, `hr_bench_last.html`(·선택 PNG) |
| **연속 프레임 비교** | 로컬 `frame_*.jpg` 디렉터리에 대한 시퀀스 비교(게이트·YOLO·Qwen 설정) | `qwen_vlm.pipeline.experiment sequence-compare` | `sequence_compare_last.json`, `sequence_compare_last.html` |

미리보기 우측 패널은 생성된 HTML을 `iframe srcdoc`으로 넣습니다(인라인 HTML에서 `file://` 차단되는 환경 대응). Windows에서는 [WebView2 런타임](https://developer.microsoft.com/microsoft-edge/webview2/)이 필요할 수 있습니다.

자세한 절차·옵션: [docs/README.md](docs/README.md) 및 [docs/hr-bench-pipeline.md](docs/hr-bench-pipeline.md), 연속 프레임 파이프라인은 [docs/week-demo-pipeline.md](docs/week-demo-pipeline.md).

---

## 현재 구현: llama.cpp `llama-server` + OpenAI 호환 API

Python 쪽은 **PyTorch / transformers 없음**(단, `uv sync --group dev` 로 YOLO용 `torch` 등이 따라옵니다).  
`llama-server`를 띄우고(또는 이미 떠 있는 서버에 연결하고) **OpenAI 호환** `chat.completions`로 이미지+텍스트를 보냅니다.

### 모델(GGUF)

- 권장 소스: [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)
- VLM은 **두 파일**이 필요합니다.
  - **LLM**: 예) `Qwen3VL-4B-Instruct-Q8_0.gguf`
  - **비전 프로젝터**: 예) `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`

기본값은 저장소 기준 다음 경로를 가리킵니다(없으면 실행 전에 받아 두거나 `--gguf` / `--mmproj`로 지정).

| 역할 | 기본 경로 |
|------|-----------|
| `llama-server` | `vendor/llama-cpp-win-cuda/llama-server.exe` |
| LLM GGUF | `vendor/qwen3-vl-4b-q8-gguf/Qwen3VL-4B-Instruct-Q8_0.gguf` |
| mmproj | `vendor/qwen3-vl-4b-q8-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` |

`vendor/`는 용량 때문에 git에 포함하지 않습니다(`.gitignore`).  
llama.cpp는 릴리스에서 **Windows + CUDA**에 맞는 zip을 받아 압축 해제하면 됩니다(예: `llama-*-bin-win-cuda-13.1-x64.zip`).  
GGUF는 Hugging Face에서 위 파일명으로 내려받아 같은 폴더에 두면 됩니다.

### 단일 이미지 스모크 테스트: `main.py`

```bash
uv sync
uv run python main.py --help
```

```bash
# 기본: 데모 이미지 URL + 한국어 프롬프트, 서버를 자동 기동 후 종료
uv run python main.py

# 로컬 이미지, 생성 길이
uv run python main.py --image .\photo.jpg --max-tokens 1024

# 다른 GGUF / 별칭(llama-server -a 와 동일해야 API model 과 맞음)
uv run python main.py --gguf .\models\a.gguf --mmproj .\models\b.gguf --model my-alias

# 이미 llama-server 가 떠 있을 때
uv run python main.py --no-spawn --base-url http://127.0.0.1:8080/v1 --model <서버에 등록된 alias>
```

서버 로그(스폰 모드): 기본 `vendor/llama-server-vlm.log` (`--log-file`로 변경 가능).

---

## HR-Bench (CLI 대안)

GUI가 기본이면, 자동화·CI·스크립트에서는 CLI를 씁니다.

```bash
uv sync --group dev
uv run python -m qwen_vlm.cli.hr_bench --help
# 또는
uv run python scripts/run_hr_bench.py --help
```

YOLO 크롭은 기본 **픽셀 예산 필터**로 비전 토큰을 줄입니다(`--disable-yolo-vlm-budget` 로 끔).  
문서: [docs/hr-bench-pipeline.md](docs/hr-bench-pipeline.md)

---

## 주간 데모·배치 실험 (CLI)

GUI의 「연속 프레임」과 같은 코어는 `qwen_vlm.pipeline.experiment`이며, 주간 자동 요약·여러 모드 일괄 실행은 루트 `run_week_experiments.py` / `-m qwen_vlm.run_week_experiments` 를 씁니다.

- 절차·데이터: [docs/week-demo-pipeline.md](docs/week-demo-pipeline.md)
- YOLO·크롭·게이트: [docs/pipeline-yolo-crops-vlm.md](docs/pipeline-yolo-crops-vlm.md)
- 데모 영상: `uv run python scripts/fetch_demo_video.py` → `data/datasets/demo/`
- `uv sync --group dev` 후: `uv run python experiment_pipeline.py bench --help`, `uv run python run_week_experiments.py --help`

---

## 참고: 속도·토큰

- 이미지 한 장 + 짧은 질문이면 **입력 토큰 수천 단위**(예: ~2,700 근처)가 흔합니다. VLM 특성입니다.
- **출력이 중간에 끊기면** 대부분 `--max-tokens` 한도 또는 `ctx-size` 부족을 의심하면 됩니다.
- 하드웨어·양자화·빌드에 따라 tok/s는 크게 달라집니다. 벤치마크는 동일 이미지·동일 `max-tokens`로 맞추는 것이 좋습니다.
- 비전 토큰 감: [docs/qwen3-vl-image-tokens.md](docs/qwen3-vl-image-tokens.md)

---

## 과거 메모

이 저장소는 이전에 **transformers + bitsandbytes NF4** 경로도 다뤘으나, 지금은 제거되었고 **llama.cpp + GGUF만** 유지합니다. 구버전 실험 기록이 필요하면 git 히스토리를 참고하면 됩니다.
