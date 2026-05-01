"""
Qwen3-VL 실험 파이프라인 패키지.

- ``qwen_vlm.main`` — llama-server 스폰·단일 이미지 VLM
- ``qwen_vlm.pipeline.experiment`` / ``qwen_vlm.experiment_pipeline`` — bench, two-stage, 병렬 YOLO+Smol
- ``qwen_vlm.pipeline.week`` / ``qwen_vlm.run_week_experiments`` — 주간 Phase 실험
- ``qwen_vlm.hr_bench`` — HR-Bench (io, strategies, metrics, report)
- ``qwen_vlm.cli`` — HR-Bench CLI, HTML webview, 실험 JSON 분석
- ``qwen_vlm.gui`` — HR-Bench GUI
- ``qwen_vlm.utils`` — 리사이즈, OpenAI URL, UTF-8, 서브프로세스
"""

__version__ = "0.1.0"
