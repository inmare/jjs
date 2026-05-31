"""matplotlib 한글 폰트 (Windows 맑은 고딕 등)."""
from __future__ import annotations

import os
import platform
from functools import lru_cache


@lru_cache(maxsize=1)
def configure_matplotlib_korean() -> str:
    """
    한글 제목·축 라벨이 □□□ 로 깨지지 않도록 폰트 설정.
    사용한 폰트 이름을 반환한다.
    """
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")

    candidates: list[tuple[str, str | None]] = []

    if platform.system() == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        for fname, name in (
            ("malgun.ttf", "Malgun Gothic"),
            ("malgunbd.ttf", "Malgun Gothic"),
            ("NanumGothic.ttf", "NanumGothic"),
        ):
            path = os.path.join(windir, "Fonts", fname)
            if os.path.isfile(path):
                try:
                    font_manager.fontManager.addfont(path)
                except Exception:
                    pass
                candidates.append((name, path))

    candidates.extend(
        [
            ("Malgun Gothic", None),
            ("NanumGothic", None),
            ("Nanum Gothic", None),
            ("AppleGothic", None),
            ("Noto Sans CJK KR", None),
            ("DejaVu Sans", None),
        ]
    )

    chosen = "DejaVu Sans"
    for name, _ in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:
            continue

    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen
