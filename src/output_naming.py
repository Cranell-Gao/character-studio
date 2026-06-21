"""Helpers for stable, readable output filenames."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


def slugify(value: str, fallback: str = "character") -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", value.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


def output_stem(model_name: str, style_name: str, character_name: str) -> str:
    return "_".join(
        [
            slugify(model_name, "model"),
            slugify(style_name, "style"),
            slugify(character_name, "character"),
        ]
    )


def next_available_path(output_dir: str | Path, stem: str, suffix: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    candidate = path / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = path / f"{stem}_{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def save_named_image(image: Image.Image, stem: str, output_dir: str | Path = "outputs") -> Path:
    out_path = next_available_path(output_dir, stem, ".png")
    image.save(out_path)
    return out_path


def save_named_markdown(markdown: str, stem: str, output_dir: str | Path = "outputs") -> Path:
    out_path = next_available_path(output_dir, stem, ".md")
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
