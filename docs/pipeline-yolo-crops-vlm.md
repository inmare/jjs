# YOLO 크롭 · 좌표 메타 · VLM 입력 (정리)

> 동일한 YOLO·좌표(클래스 비노출) 설계는 **HR-Bench** 파이프라인(`qwen_vlm/hr_bench/strategies.py`)에서도 재사용됩니다. 진입점: [hr-bench-pipeline.md](./hr-bench-pipeline.md).

이 문서는 `qwen_vlm.vision.yolo.run_yolo_crops`, `qwen_vlm.experiment_pipeline` 의 벤치/병렬/two-stage 경로, 그리고 **클래스명 없이 좌표 + 이미지**만 Qwen(Smol)에 넘기는 현재 설계를 요약한다.

## 목표

- **고해상도 원본**이 있어도 VLM에는 **(1) 저해상도 전망 1장** + **(2) YOLO가 자른 ROI를 `crop_max_side` 이하로 리사이즈한 이미지**를 보내 **입력 토큰**을 막는다.
- YOLO의 클래스·점수는 **신뢰가 들쑥날쑥할 수 있으므로**, VLM **프롬프트 본문에는 넣지 않는다.**
- 대신 **원본 픽셀 크기**와 각 크롭의 **원본 좌표 `xyxy`** 를 텍스트로 붙여, 이미지 슬롯(`#0` 전망, `#1~` 잘라내기)과 맞춘다.
- **너무 작은 박스**는 크롭 수가 늘면서도 시야만 잡아먹지 않게 **면적/짧은 변 임계값**으로 제외한다.

## `run_yolo_crops` (`qwen_vlm/vision/yolo.py`)

**반환값 (순서):**

| 순서 | 의미 |
|------|------|
| `crops` | `PIL.Image` 리스트, 원본과 동일 해상도에서 잘라낸 이미지 |
| `bboxes` | `crops`와 **동일 인덱스**의 `(x1,y1,x2,y2)` (원본 이미지 픽셀) |
| `summary` | **로그/JSON용** 문자열(필터 **통과한** 박스만 `class:conf` 나열). VLM 프롬프트에는 쓰지 않음 |
| `n_det` | YOLO **raw** 탐지 수(필터 전) |
| `all_cls` | 탐지에 등장한 클래스 id 집합(필터 전) |

**박스 선택:** 면적 **큰 순**으로 훑으면서, `max_crops`개가 찰 때까지 반복. 중간에 크기 필터에 걸리면 **건너뛰고** 다음 박스를 쓴다(상위 N개가 아니라, “필터 통과하는 것 중 앞에서부터 최대 K개”).

**인자:**

- `min_crop_short_side` (기본 0=끔): `min(가로,세로) < 임계` 이면 제외.
- `min_crop_area` (기본 0=끔): `가로*세로 < 임계` 이면 제외.
- `vlm_budget` (기본 `True`): 아래 **VLM 픽셀 예산 필터**를 탐지 박스에 적용한 뒤 `max_crops` 로 자름.
- `context_max_side`: `vlm_budget=True` 일 때 필수에 가깝습니다(미지정 시 `run_yolo_crops` 내부에서 960).
- `max_bbox_area_numerator` / `max_bbox_area_denominator` (기본 1/4): 원본 면적 대비 **너무 큰** 박스 제외.

**특수 `summary`:** 탐지는 있는데 **전부 필터에서 탈락**이면 `all_crops_filtered_by_size`.

## VLM 픽셀 예산 (`qwen_vlm/vision/yolo_vlm_budget.py`)

`vlm_budget=True`이면, 면적 큰 순으로 모은 후보 박스에 대해:

1. **과대 박스:** `area * denominator < W*H * numerator` 를 만족하는 것만 유지(기본: 원본의 1/4 미만).
2. **동일 `xyxy` 중복** 제거.
3. **다른 박스에 완전 포함**되는 박스 제거.
4. 저해상 전망(`context_max_side`로 `resize_max_side` 한 크기)으로 투영한 `xyxy` 기준 **중복** 제거.
5. `(전망_w×전망_h) + Σ(원본 박스 면적)` 이 `W×H` 를 넘으면 **면적이 작은 박스부터** 제거.

이후 `run_yolo_crops`는 남은 박스를 **원래 순서(면적 내림차순)**로 최대 `max_crops`개까지 크롭합니다.

실험 CLI `qwen_vlm.pipeline.experiment` 에는 `--disable-yolo-vlm-budget` 으로 이 경로를 끌 수 있습니다.

## VLM 쪽 문구: `vlm_yolo_crops_coordinate_preamble_ko` (`qwen_vlm/experiment_pipeline.py`)

- 원본 `W×H`, 이미지 `#0` = 전망(긴 변 `context_max_side` 이하), `#1~` = 잘라낸 영역(원본 좌표) + VLM에 넣을 때 `crop_max_side` 로 맞춘다는 설명.
- **YOLO `[YOLO 프리셋 탐지: …]` 한 줄**은 **제거**되었다(벤치 YOLO+Qwen 경로·게이트 full 추론 동일).

## 경로별 동작

### 벤치 `bench` / `bench_with_frame_gating`

- **Qwen-only:** `context_max_side` 로 맞춘 전망 1장.
- **YOLO+Qwen:** `urls_y = [전망] + [크롭 각각을 crop_max_side 로 리사이즈한 data URL]`, 프롬프트 = **좌표 프리엄블** + 사용자 프롬프트.
- JSON: `images.qwen_context` 에 `original_width` / `original_height`, `yolo_qwen.crop_bboxes_xyxy_orig` 등.

#### `--frame-gate` (MSE / YOLO / 조합)

| 값 | MSE(휘도 썸) | YOLO·크롭 | Qwen(VLM) 생략 조건(2프레임째~) |
|----|--------------|------------|----------------------------------|
| `off` | — | 매 프레임 | 생략 없음(풀 벤치) |
| `mse` | O (`--mse-threshold`) | MSE “전부 스킵”이면 **실행 안 함** | MSE “같다”이면 YOLO·Qwen **둘 다** 생략. MSE “다르다”이면 **품 추론**(YOLO로만 Qwen 스킵하는 경로 **없음**) |
| `yolo` | **안 씀** | **매 프레임** | 이전과 `n_det`+클래스 집합(또는 크롭 위치 게이트) 같으면 **Qwen만** 생략 |
| `mse_then_yolo` | O | MSE “전부 스킵”이면 안 함, 아니면 매 프레임 | ① MSE로 전부 생략 가능 ② 그 외 YOLO 후 서명/크롭으로 Qwen만 생략 |

**YOLO만** 쓰고 싶으면 `--frame-gate yolo` (또는 GUI에서 `yolo` 선택). 이때 `--mse-threshold` 는 읽지 않는다.

### 연속 프레임 — **크롭 위치 게이트** (`--use-crop-layout-gate`)

기본 `yolo` 게이트는 `n_det|클래스` 문자열만 비교한다. **픽셀(MSE)** 이 약간만 달라도 “다른 장면”이면 전부 풀 추론이 나갈 수 있다.

**대안:** `bench_with_frame_gating` + `frame_gate` 가 `yolo` 또는 `mse_then_yolo` 일 때

- `--use-crop-layout-gate` 를 켜면 **필터 통과한 크롭의 개수**가 같고,
- 각 박스 중심을 **프레임 크기로 0~1 정규화**한 뒤 (y,x)로 정렬해 짝을 맞추어,
- **정규화 거리**가 `--crop-gate-max-shift` (기본 `0.02`) **이하**이면 “장면 동일”로 보고 **Qwen(VLM)만 생략**한다.

YOLO·크롭 추출은 **매 프레임** 수행한다. JSON `frame_gating.path` 가 `skip_vlm_crop_layout` 이면 이 규칙으로 생략한 것이다. (기존 클래스 서명만 쓸 때는 `skip_vlm_yolo_fp`.)

### `run_yolo_smol_parallel_qwen` (병렬)

- YOLO와 Smol을 **스레드 풀**로 동시에 실행.
- Qwen: **`[low_url] + [크롭…]`** (병렬도 이제 벤치와 같이 **다중 이미지**).
- `_merge_parallel_stage1`: YOLO 클래스 문구 없이 **원본·좌표** + **Smol 텍스트**만.

### `run_two_stage`

- 1단이 Smol 실패/미설정일 때 YOLO **폴백 JSON**을 만들 때 `run_yolo_crops` 를 부른다(5-tuple, 앞의 두 항목은 사용 안 함).
- `min_crop_short_side` / `min_crop_area` 는 **폴백에서도** 동일 인자로 전달.

## CLI (루트 `experiment_pipeline.py` 또는 `qwen_vlm.experiment_pipeline`)

| 인자 | 기본 | 설명 |
|------|------|------|
| `--context-max-side` | 960 | 전망(짧/긴 변 중 긴 쪽) 상한 |
| `--crop-max-side` | 640 | 크롭을 VLM에 보내기 직전 리사이즈 상한 |
| `--max-crops` | 3 | YOLO에서 최대 몇 ROI까지 |
| `--min-crop-short-side` | 0 | (크기) 탐지 사각형의 **짧은 쪽 길이**(원본, 픽셀)가 이보다 작으면 VLM에 안 보냄. 0=끔 |
| `--min-crop-area` | 0 | (크기) **면적** 가×세(원본, 픽셀²)가 이보다 작으면 제외. 0=끔 |
| `--use-crop-layout-gate` | 끔 | 클래스 대신 **크롭 개수·박스 가운데 위치**로 이전 프레임과 비교해 VLM만 생략 |
| `--crop-gate-max-shift` | 0.02 | 위 옵션 켰을 때: 박스 가운데가 **화면마다 0~1**로 잡을 때, 이전과 **최대 몇 %**까지 옮겨가도 '같다'로 볼지(작을수록 VLM을 더 자주) |
| `--disable-yolo-vlm-budget` | 끔 | 켜면 YOLO→VLM **픽셀 예산·필터**를 적용하지 않음(기본은 적용) |

`run_week_experiments` (`qwen_vlm/run_week_experiments.py`)는 Phase1 `bench_with_frame_gating` 등에 동일한 벤치 인자를 넘깁니다. YOLO 예산은 **기본 켜짐**입니다.

**GUI (HR-Bench):** `uv run python -m qwen_vlm.gui.hr_bench_app` — pywebview 창에서 전략·옵션 입력, 오른쪽 iframe에 리포트(표+HTML 차트).

## HR(고해상도) 대조 실험 시

- **고정:** `--context-max-side`, `--crop-max-side`, `--min-crop-*`, `--max-crops`, 프롬프트.
- **바꿈:** `--frames-dir` (HR 프레임 폴더)만.
- 4K 등에서는 **VLM/서버 쪽** 해상도·패치 제한이 있을 수 있으니, Qwen/llama-server에 **실제로 보내는** 픽셀은 위 리사이즈 캡으로 맞춰진다는 점에 유의.

---

## 기본 테스트 (명령어)

프로젝트 루트 `D:\Private\Fiddles\qwen-vlm` 기준, PowerShell에서 경로는 환경에 맞게 조정.

### 1) 문법·임포트만 (서버 없음)

```powershell
Set-Location D:\Private\Fiddles\qwen-vlm
uv run python -m py_compile qwen_vlm/experiment_pipeline.py qwen_vlm/vision/yolo.py
uv run python -c "import qwen_vlm.experiment_pipeline; import qwen_vlm.vision.yolo; print('import ok')"
```

### 2) YOLO 크롭만 (VLM/llama-server 불필요, 첫 실행 시 `yolov8n.pt` 다운로드 가능)

```powershell
Set-Location D:\Private\Fiddles\qwen-vlm
uv run python -c "from pathlib import Path; from PIL import Image; from qwen_vlm.vision.yolo import load_yolo, run_yolo_crops; p=Path('data/datasets/demo/frames/frame_000001.jpg'); img=Image.open(p).convert('RGB'); m=load_yolo('yolov8n.pt'); c,b,s,n,a=run_yolo_crops(img,model=m,max_crops=3,min_crop_short_side=0,min_crop_area=0); print('crops',len(c),'bboxes',b,'summary',s,'n_det',n)"
```

필터 동작만 보려면 예:

```powershell
uv run python -c "from pathlib import Path; from PIL import Image; from qwen_vlm.vision.yolo import load_yolo, run_yolo_crops; img=Image.open(Path('data/datasets/demo/frames/frame_000001.jpg')).convert('RGB'); m=load_yolo('yolov8n.pt'); c,b,s,n,a=run_yolo_crops(img,model=m,max_crops=3,min_crop_short_side=48,min_crop_area=4096); print(len(c),b,s)"
```

### 3) `bench` 1프레임 (Qwen `llama-server` 가 **이미** `http://127.0.0.1:8765/v1` 에 떠 있어야 함)

`experiment_pipeline` 의 `--frame-gate` 기본값은 **`off`** (또는 환경변수 `BENCH_FRAME_GATE`). 별도 지정 없이 **프레임 게이트 없이** 벤치만 돌린다.

```powershell
Set-Location D:\Private\Fiddles\qwen-vlm
uv run python experiment_pipeline.py bench --max-frames 1
```

다른 llama-server 주소를 쓰면:

```powershell
uv run python experiment_pipeline.py bench --max-frames 1 --base-url "http://127.0.0.1:8765/v1"
```

크롭 필터를 켠 스모크:

```powershell
uv run python experiment_pipeline.py bench --max-frames 1 --min-crop-short-side 32 --min-crop-area 0
```

### 4) 전체 주간 스크립트 스모크 (서버 띄우고, 프레임 1장, 시간 오래 걸릴 수 있음)

```powershell
Set-Location D:\Private\Fiddles\qwen-vlm
uv run python run_week_experiments.py --smoke
```

`--smoke`는 `max-frames=1`으로 덮어쓴다.

### 5) 병렬 모드 (Qwen + **Smol** 두 서버 필요, `--small-vlm-url` 필수)

```powershell
# 예: Qwen 8765, Smol 8766
$env:SMALL_VLM_OPENAI_BASE = "http://127.0.0.1:8766/v1"
uv run python experiment_pipeline.py parallel-yolo-smol --max-frames 1 --small-vlm-url "http://127.0.0.1:8766/v1" --small-vlm-model smolvlm-256m-q8
```

(모델 별칭은 실제 `llama-server` 의 `-a`와 일치시킬 것.)

---

## 관련 소스

- `qwen_vlm/vision/yolo.py` — `run_yolo_crops`, 클래스 프리셋
- `qwen_vlm/vision/yolo_vlm_budget.py` — 픽셀 예산·박스 필터
- `qwen_vlm/pipeline/experiment.py` (`qwen_vlm/experiment_pipeline.py` shim) — `vlm_yolo_crops_coordinate_preamble_ko`, `bench`, `run_yolo_smol_parallel_qwen`, `run_two_stage`
- `qwen_vlm/run_week_experiments.py` — Phase별 서버 기동 + 위 파이프라인 호출
- 루트 `experiment_pipeline.py` 등은 동일 API를 쓰는 **얇은 래퍼** (이전 `python experiment_pipeline.py` 경로 유지)
