# 구글 Colab에서 돌려보기

Colab은 **브라우저 노트붱**이라 **이 저장소의 데스크톱 실험 GUI(`pywebview`)는 사용할 수 없습니다.** 대신 **같은 CLI**로 HR-Bench, 연속 프레임 등을 실행합니다.

- **실행 순서만 복사·붙여넣기** 하려면: 루트의 **`notebooks/colab_quickstart.ipynb`** 파일을 Colab에 업로드해서 열면 됩니다.
- Colab 무료 GPU는 세션 시간·VRAM에 제한이 있으므로 **`--max-samples` 작게**, 전략은 **`qwen_only`** 로 먼저 연결부터 확인하는 것을 권장합니다.

아래는 `.ipynb` 없이 마크다운만으로 따라 할 때용 **요약**입니다.

---

## 1. 런타임

메뉴 **런타임 → 런타임 유형 변경**에서 **GPU**(예: T4)가 있으면 선택합니다.

**Python 3.12**: 이 패키지는 `requires-python >= 3.12` 입니다. Colab 이미지에 따라 버전이 다를 수 있으므로 첫 셀에서 확인하세요.

```python
import sys
print(sys.version)
```

12 미만이면, 해당 Colab 이미지로는 호환되지 않을 수 있어 **런타임 교체**(또는 Python 12 지원 노트 북 제공 환경)를 검토해야 합니다.

---

## 2. PyTorch (CUDA)

Colab에는 보통 NVIDIA 드라이버가 있습니다. **현재 Colab의 CUDA에 맞는 wheel** 은 [PyTorch Get Started](https://pytorch.org/get-started/locally/)에서 **Linux + Pip + CUDA** 조합으로 고른 뒤, 나온 `pip install ...` 한 줄을 그대로 쓰는 것이 가장 안전합니다.

---

## 3. 저장소 가져오기

- **GitHub에 올려둔 경우**:

  ```bash
  !git clone https://github.com/<사용자>/<저장소>.git
  %cd <저장소>
  ```

- **아직 GitHub에 없고 zip만 있는 경우**: Colab 왼쪽 폴더 아이콘에서 zip 업로드 후 압축 해제하고 `%cd` 로 그 폴더로 이동합니다.

---

## 4. 패키지 설치

```bash
!pip install -U pip
!pip install -e .
!pip install ultralytics opencv-python-headless huggingface-hub psutil datasets matplotlib
```

(이미 torch를 위에서 설치했다면 이어서 진행.)

---

## 5. llama-server (Linux 바이너리)

[llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 에서 **Linux + CUDA + x64**용 아카이브를 고릅니다. 예시(버전·파일명은 릴리스마다 다름):

```bash
# 예: 압축을 풀면 bin/llama-server 가 생기는 구조라고 가정
!mkdir -p vendor/llama-cpp-linux-cuda
# 아래 URL은 릴리스 페이지에서 복사한 직접 링크로 바꿉니다
# !wget -q -O /tmp/llama.tgz "https://github.com/ggml-org/llama.cpp/releases/download/.../....tar.gz"
# !tar -xzf /tmp/llama.tgz -C vendor/llama-cpp-linux-cuda
!chmod +x vendor/llama-cpp-linux-cuda/**/llama-server
```

실제 경로에 맞춰 **`LLAMA`** 변수를 잡아 두면 이후 명령이 짧아집니다:

```python
import os
os.environ["LLAMA"] = "/content/.../llama-server"  # 실제 경로로 수정
```

---

## 6. Qwen GGUF 내려받기

```bash
!pip install -q huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import os
os.makedirs("vendor/qwen3-vl-4b-q8-gguf", exist_ok=True)
hf_hub_download("Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q8_0.gguf", local_dir="vendor/qwen3-vl-4b-q8-gguf")
hf_hub_download("Qwen/Qwen3-VL-4B-Instruct-GGUF", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf", local_dir="vendor/qwen3-vl-4b-q8-gguf")
```

---

## 7. 최소 HR-Bench (Qwen만, 샘플 2개)

Windows 기본값(`llama-server.exe`)은 Linux에서 동작하지 않으므로 **`--llama-server`** 를 Linux 바이너리로 **반드시** 지정합니다.

```bash
!python -m qwen_vlm.cli.hr_bench \
  --llama-server /content/.../llama-server \
  --gguf vendor/qwen3-vl-4b-q8-gguf/Qwen3VL-4B-Instruct-Q8_0.gguf \
  --mmproj vendor/qwen3-vl-4b-q8-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
  --max-samples 2 \
  --strategies qwen_only \
  --json-out docs/hr_bench_last.json \
  --html-out docs/hr_bench_last.html
```

결과 HTML은 Colab에서 `files.download("docs/hr_bench_last.html")` 등으로 내려받아 브라우저에서 열 수 있습니다.

---

## 8. 연속 프레임(`sequence-compare`)

로컬과 같이 `--frames-dir` 에 이미지가 있어야 합니다. Colab에서는 샘플을 업로드하거나 코드로 몇 장 생성해 넣습니다. 먼저 **동일하게 `llama-server`를 띄워 둘지**, 아니면 **CLI가 알아서 띄우는지**(해당 명령의 `--help`)를 확인합니다. 로컬 [사용법.md](사용법.md)의 “연속 프레임 비교” 절과 옵션 설명은 그대로 적용됩니다.

---

## 9. 알아두면 좋은 점

- **VRAM**: 무료 GPU는 종류·용량별로 크고 무거운 전략(`all`, Smol 병렬 등)이 **OOM** 날 수 있습니다. 샘플 수·전략·해상도 캽을 줄이세요.
- **세션 종료 시 파일 소멸**: Colab 디스크는 세션이 끝나면 날아갑니다. 중요한 `docs/*.json`, HTML은 빨리 다운로드하거나 Drive에 마운트해 저장하세요.
- 더 자세한 개념·옵션: [README.md](README.md), [docs/hr-bench-pipeline.md](docs/hr-bench-pipeline.md).
