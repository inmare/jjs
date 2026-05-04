"""HR-Bench MCQ용 Qwen 보조 프롬프트(영어) 및 Smol 1단계 자연문 요약."""

from __future__ import annotations

import json
import re
from typing import Any


HR_BENCH_SMOL_STAGE1_PROMPT_EN = (
    "Describe this image briefly in plain English so another vision model can use it as extra "
    "context for a separate multiple-choice task. Mention salient objects, people, approximate "
    "layout/spatial relationships, rough setting, and readable text whenever it visibly matters. "
    "Use a few short sentences; plain prose only (no bullets, markdown, JSON, or rigid schemas)."
)


HR_BENCH_MC_SUPPLEMENT_INTRO_EN = (
    "The following lines are EXTRA hints for the MULTIPLE CHOICE problem above—not part of the "
    "formal question wording. Prefer the attached images whenever these hints contradict what you "
    "see in pixels."
)


def compose_hr_bench_mc_with_supplementary_en(
    mcq_prompt: str,
    *,
    yolo_context_sentence_en: str,
    compact_vlm_sentence_en: str | None,
) -> str:
    blocks = [
        mcq_prompt.strip(),
        "",
        HR_BENCH_MC_SUPPLEMENT_INTRO_EN,
        "",
        f"YOLO-derived context (English summary): {yolo_context_sentence_en}",
    ]
    if compact_vlm_sentence_en:
        blocks.extend(
            [
                "",
                (
                    "Compact auxiliary vision cue (English summary): "
                    f"{compact_vlm_sentence_en}"
                ),
            ]
        )
    return "\n".join(blocks).strip()


def hr_bench_yolo_context_sentence_en(
    *,
    overview_w: int,
    overview_h: int,
    canvas_w: int,
    canvas_h: int,
    n_raw_detections: int,
    n_crops_sent: int,
    bboxes_ov: list[tuple[int, int, int, int]],
    crop_max_side: int,
    context_resize_scale: float | None,
    context_max_side: int | None,
) -> str:
    if context_resize_scale is not None:
        ctx_desc = (
            f"object detection ran on an RGB frame uniformly scaled ({context_resize_scale:g}×) "
            f"to roughly {canvas_w}×{canvas_h} px from the benchmark raster;"
        )
    else:
        cms = context_max_side if context_max_side is not None else 960
        ctx_desc = (
            f"object detection ran on an RGB frame whose longest side was capped near {cms} px "
            f"(current canvas {canvas_w}×{canvas_h});"
        )
    if bboxes_ov:
        rects = "; ".join(
            f"c{i}[{x1},{y1},{x2},{y2}]"
            for i, (x1, y1, x2, y2) in enumerate(bboxes_ov, start=1)
        )
        res = (
            f"each patch has longest edge≤{crop_max_side} px for the VLM."
            if crop_max_side > 0
            else "patches keep native crop resolution versus the detector canvas."
        )
        crops_bit = (
            f" supplementary thumbnails #{1}–#{len(bboxes_ov)} line up with these overview-space "
            f"RECTS ({rects}) and {res}"
        )
    else:
        crops_bit = " no supplementary thumbnails passed filtering."
    return (
        f"Image slot #0 is a low-resolution overview about {overview_w}×{overview_h} px; {ctx_desc} "
        f"YOLO reported {n_raw_detections} raw boxes and attached {n_crops_sent} extra tiles.{crops_bit}"
    )


_PLACEHOLDER_NOTE_RE = re.compile(
    r'^["\']?one\s+short\s+english\s+sentence\.?["\']?$',
    re.I | re.DOTALL,
)


def _strip_code_fence(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _try_parse_json_object(s: str) -> dict[str, Any] | None:
    t = _strip_code_fence(s)
    try:
        d = json.loads(t)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    i0 = t.find("{")
    if i0 < 0:
        return None
    depth = 0
    for j in range(i0, len(t)):
        c = t[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                frag = t[i0 : j + 1]
                try:
                    d2 = json.loads(frag)
                    return d2 if isinstance(d2, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _junk_object_fragment(seg: str) -> bool:
    s = seg.strip()
    if not s:
        return True
    if len(s) > 42:
        return True
    digit_ratio = sum(1 for c in s if c.isdigit()) / max(len(s), 1)
    if digit_ratio > 0.45 and len(s) >= 8:
        return True
    if len(s.split()) >= 10:
        return True
    if re.search(r"\d{2,}[/\-:]\d{2,}", s):
        return True
    if re.fullmatch(r"[\d\s\-:/.Ee+]+", s):
        return True
    return False


def _clean_objects(lst: Any) -> list[str]:
    if not isinstance(lst, list):
        return []
    out: list[str] = []
    for x in lst:
        if isinstance(x, (int, float)):
            sx = str(x).strip()
        elif isinstance(x, str):
            sx = x.strip()
        else:
            continue
        if _junk_object_fragment(sx):
            continue
        if sx in (".", "…"):
            continue
        out.append(sx[:72])
        if len(out) >= 6:
            break
    return out


def _sanitize_note(note: Any) -> str:
    if not isinstance(note, str):
        return ""
    n = " ".join(note.strip().split())
    if not n:
        return ""
    if len(n) > 260:
        n = n[:260].rsplit(" ", 1)[0] + "…"
    stripped = n.strip(" .\"'`")
    if _PLACEHOLDER_NOTE_RE.match(stripped):
        return ""
    if _junk_object_fragment(n):
        return ""
    return n


def _looks_mostly_digit_or_symbol(s: str) -> bool:
    s2 = "".join(ch for ch in s if not ch.isspace())
    if not s2:
        return False
    return sum(c.isdigit() or c in ":/.-+eE" for c in s2) / len(s2) > 0.62


_SUMMARY_MAX_CHARS = 800


def summarize_smol_for_hr_bench_mc_en(raw: str | None, *, ok: bool) -> str:
    """
    Smol 원문은 그대로 두고 · 짧은 휴리스틱으로 노이즈(숫자/기호 과다 OCR 잔류 등) 여부만 판단한 뒤,
    깨끗하면 같은 언어(영문 위주)·적당한 길이로 잘라 Qwen 보조 한 줄만 만든다. (구버전 일줄 JSON 은 선택 파싱)
    """
    if not ok:
        return (
            "The compact cue model raised an HTTP/runtime error—ignore missing text cues and rely "
            "on thumbnails only."
        )
    s = (raw or "").strip()
    if not s:
        return "The compact cue model returned an empty reply; trust the images."
    if len(s) < 4:
        return "The compact cue model returned negligible text; trust the images."

    blob = _try_parse_json_object(s)
    if blob:
        objs = _clean_objects(blob.get("objects"))
        risk_raw = str(blob.get("risk") or "").strip().strip('"').lower()
        risk = risk_raw if risk_raw in ("low", "med", "high") else "med"
        note = _sanitize_note(blob.get("note"))
        if objs:
            obj_clause = "Suggested salient cues: " + "; ".join(objs) + "."
        else:
            obj_clause = (
                "No clean salient cue strings survived filtering (possible OCR hallucinations)."
            )
        risk_clause = f"Rough occupancy level flagged as `{risk}` risk."
        if note:
            return f"{risk_clause} {obj_clause} Scene gist: {note}"
        return f"{risk_clause} {obj_clause}"

    body = _strip_code_fence(s)
    condensed = body.replace("{", "").replace("}", "").strip()
    if _looks_mostly_digit_or_symbol(condensed):
        return (
            "The compact cue looked like noisy OCR/digits-heavy output and was withheld—prefer the "
            "question and thumbnails."
        )

    text = " ".join(body.split())
    if len(text) > _SUMMARY_MAX_CHARS:
        cut = text[: _SUMMARY_MAX_CHARS].rsplit(" ", 1)[0]
        text = cut + "…"
    return text