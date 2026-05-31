# C: 드라이브 uv/HF/torch 캐시 부담을 줄이기 위해 D: (또는 프로젝트 디스크)로 옮깁니다.
# PowerShell:  . .\11주차-3\setup_storage.ps1

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = "D:\Programming\jjs" }

$CacheRoot = Join-Path $Root ".cache"
$UvCache   = Join-Path $CacheRoot "uv"
$HfHome    = Join-Path $CacheRoot "huggingface"
$TorchHome = Join-Path $CacheRoot "torch"
$YoloW     = Join-Path $Root "data\yolo_weights"
$Week3Res  = Join-Path $PSScriptRoot "results"

New-Item -ItemType Directory -Force -Path $UvCache, $HfHome, $TorchHome, $YoloW, $Week3Res | Out-Null

$env:UV_CACHE_DIR          = $UvCache
$env:HF_HOME               = $HfHome
$env:HUGGINGFACE_HUB_CACHE = Join-Path $HfHome "hub"
$env:TRANSFORMERS_CACHE    = Join-Path $HfHome "transformers"
$env:TORCH_HOME            = $TorchHome
$env:WEEK3_YOLO_WEIGHTS    = $YoloW
$env:WEEK3_RESULTS_DIR     = $Week3Res
# Ultralytics 설정·가중치도 프로젝트 쪽으로
$UltralyticsCfg = Join-Path $CacheRoot "ultralytics"
New-Item -ItemType Directory -Force -Path $UltralyticsCfg | Out-Null
$env:YOLO_CONFIG_DIR       = $UltralyticsCfg

Write-Host "UV_CACHE_DIR          = $env:UV_CACHE_DIR"
Write-Host "HF_HOME                 = $env:HF_HOME"
Write-Host "TORCH_HOME              = $env:TORCH_HOME"
Write-Host "WEEK3_YOLO_WEIGHTS      = $env:WEEK3_YOLO_WEIGHTS"
Write-Host "WEEK3_RESULTS_DIR       = $env:WEEK3_RESULTS_DIR"
Write-Host ""
Write-Host "이 세션에서 uv / 벤치마크를 실행하세요. 영구 적용은 시스템 환경 변수에 동일 값을 등록하세요."
