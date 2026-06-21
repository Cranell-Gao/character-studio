"""Prompt engineering and schema recovery for character generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .ollama_client import OllamaClient


STYLE_PRESETS: dict[str, str] = {
    "fantasy": "high fantasy RPG concept art, ornate costume, magical details",
    "sci-fi": "science fiction game concept art, advanced materials, cinematic lighting",
    "dark fantasy": "dark fantasy game concept art, dramatic mood, gothic detail",
    "anime game art": "anime game character key art, clean linework, expressive design",
    "realistic concept art": "realistic AAA game concept art, detailed materials, believable anatomy",
}

STYLE_LABELS: dict[str, str] = {
    "奇幻 RPG": "fantasy",
    "科幻機甲": "sci-fi",
    "黑暗奇幻": "dark fantasy",
    "動漫遊戲美術": "anime game art",
    "寫實概念設計": "realistic concept art",
}


def normalize_style(style: str) -> str:
    return STYLE_LABELS.get(style, style)


@dataclass
class CharacterSpec:
    name: str
    archetype: str
    background: str
    abilities: list[str]
    outfit: str
    color_palette: list[str]
    visual_prompt: str
    negative_prompt: str
    raw_response: str = field(repr=False, default="")

    def to_markdown(self) -> str:
        abilities = "\n".join(f"- {item}" for item in self.abilities)
        colors = ", ".join(self.color_palette)
        return (
            f"# {self.name}\n\n"
            f"**角色定位：** {self.archetype}\n\n"
            f"## 背景故事\n{self.background}\n\n"
            f"## 角色能力\n{abilities}\n\n"
            f"## 服裝設計\n{self.outfit}\n\n"
            f"## 色彩配置\n{colors}\n\n"
            f"## Diffusion Prompt\n{self.visual_prompt}\n\n"
            f"## Negative Prompt\n{self.negative_prompt}\n"
        )


SYSTEM_PROMPT = """You are a senior game character art director and prompt engineer.
Return only valid compact JSON. Do not use markdown fences.
The JSON schema is:
{
  "name": "short character name in Traditional Chinese",
  "archetype": "short role in Traditional Chinese",
  "background": "2-4 sentences in Traditional Chinese",
  "abilities": ["3-5 concise abilities in Traditional Chinese"],
  "outfit": "costume and silhouette description in Traditional Chinese",
  "color_palette": ["3-6 color names in Traditional Chinese"],
  "visual_prompt": "English SDXL prompt for one full-body game character concept art image",
  "negative_prompt": "English negative prompt"
}
Use Traditional Chinese for name, archetype, background, abilities, outfit, and color_palette.
Use English only for visual_prompt and negative_prompt because the image models respond better to English prompts.
Visual prompt rules: include full body, centered character, readable silhouette, detailed costume, neutral background, and the selected art style. Avoid text, watermark, logo, cropped body, extra limbs.
"""


def build_user_prompt(concept: str, style: str, extra_notes: str = "") -> str:
    style = normalize_style(style)
    style_text = STYLE_PRESETS.get(style, STYLE_PRESETS["fantasy"])
    notes = extra_notes.strip() or "No extra constraints."
    return (
        f"Character concept: {concept.strip()}\n"
        f"Target style: {style} ({style_text})\n"
        f"Extra constraints: {notes}\n"
        "Create a coherent game character design. Write the character card fields in Traditional Chinese, but keep the diffusion prompts in English."
    )


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _string_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or fallback
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()] or fallback
    return fallback


def spec_from_response(response: str, concept: str, style: str) -> CharacterSpec:
    style = normalize_style(style)
    data = _extract_json(response)
    style_text = STYLE_PRESETS.get(style, STYLE_PRESETS["fantasy"])
    visual_prompt = str(data.get("visual_prompt") or "").strip()
    if not visual_prompt:
        visual_prompt = f"full body {style_text} character concept art, {concept}, neutral background"
    negative_prompt = str(data.get("negative_prompt") or "").strip()
    if not negative_prompt:
        negative_prompt = "low quality, blurry, watermark, text, logo, cropped, extra limbs, bad anatomy"

    return CharacterSpec(
        name=str(data.get("name") or "Unnamed Character").strip(),
        archetype=str(data.get("archetype") or "Game character").strip(),
        background=str(data.get("background") or "A mysterious character ready for a new adventure.").strip(),
        abilities=_string_list(data.get("abilities"), ["adaptive combat", "quick thinking", "survival instinct"]),
        outfit=str(data.get("outfit") or "Layered game-ready costume with a clear silhouette.").strip(),
        color_palette=_string_list(data.get("color_palette"), ["black", "silver", "accent color"]),
        visual_prompt=visual_prompt,
        negative_prompt=negative_prompt,
        raw_response=response,
    )


def generate_character_spec(
    concept: str,
    style: str,
    extra_notes: str,
    client: OllamaClient | None = None,
    temperature: float = 0.75,
) -> CharacterSpec:
    if not concept.strip():
        raise ValueError("Please enter a character concept.")
    ollama = client or OllamaClient()
    response = ollama.chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(concept, style, extra_notes),
        temperature=temperature,
    )
    return spec_from_response(response, concept=concept, style=style)
