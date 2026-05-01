"""HR-Bench 다전략 집계."""
from __future__ import annotations

import math
from typing import Any


def _finite_float(v: Any) -> float | None:
    """집계·JSON 에 넣을 유한 부동소수만. NaN/Inf 는 None 과 동일."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _finite_int(v: Any) -> int | None:
    f = _finite_float(v)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError, OverflowError):
        return None


def aggregate_strategy_rows(
    name: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not rows:
        return {
            "strategy": name,
            "evaluated": 0,
            "correct": 0,
            "accuracy": None,
            "mean_seconds": None,
            "mean_prompt_tokens": None,
            "mean_stage1_prompt_tokens": None,
            "max_gpu_memory_used_mb": None,
        }
    correct = sum(1 for r in rows if r.get("correct"))
    pts: list[int] = []
    for r in rows:
        pi = _finite_int(r.get("prompt_tokens"))
        if pi is not None:
            pts.append(pi)
    st1: list[int] = []
    for r in rows:
        si = _finite_int(r.get("stage1_prompt_tokens"))
        if si is not None:
            st1.append(si)
    secs: list[float] = []
    for r in rows:
        sf = _finite_float(r.get("seconds"))
        if sf is not None:
            secs.append(sf)
    gpu_vals: list[float] = []
    for r in rows:
        ra = r.get("resources_after")
        if isinstance(ra, dict):
            g = _finite_float(ra.get("gpu_memory_used_mb"))
            if g is not None:
                gpu_vals.append(g)
    return {
        "strategy": name,
        "evaluated": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 5),
        "mean_seconds": round(sum(secs) / len(secs), 4) if secs else None,
        "mean_prompt_tokens": round(sum(pts) / len(pts), 2) if pts else None,
        "mean_stage1_prompt_tokens": round(sum(st1) / len(st1), 2) if st1 else None,
        "max_gpu_memory_used_mb": round(max(gpu_vals), 2) if gpu_vals else None,
    }
