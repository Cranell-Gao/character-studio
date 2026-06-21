from PIL import Image

import app
from app import OUTPUT_DIR, PROJECT_DIR
from src.prompt_engine import CharacterSpec


def test_output_dir_is_fixed_to_project_outputs():
    assert OUTPUT_DIR.is_absolute()
    assert OUTPUT_DIR == PROJECT_DIR / "outputs"


def test_generate_character_returns_named_image_path_for_gradio(monkeypatch, tmp_path):
    spec = CharacterSpec(
        name="測試角色",
        archetype="測試定位",
        background="測試背景。",
        abilities=["能力一"],
        outfit="測試服裝。",
        color_palette=["藍色"],
        visual_prompt="full body test character",
        negative_prompt="low quality",
    )

    class FakeZImage:
        def generate(self, prompt, negative_prompt, config):
            return Image.new("RGB", (16, 16), "white")

    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "z_image", FakeZImage())
    monkeypatch.setattr(app, "generate_character_spec", lambda **kwargs: spec)

    _, image_preview, _, card_file, image_file, image_path = app.generate_character(
        image_model="Z-Image Turbo",
        concept="測試角色",
        style="科幻機甲",
        extra_notes="",
        reference_image=None,
        seed=1,
        steps=8,
        guidance_scale=0.0,
        control_strength=0.0,
        width=16,
        height=16,
        llm_temperature=0.5,
    )

    assert image_preview == image_file == image_path
    assert image_path.endswith("Z_Image_Turbo_科幻機甲_測試角色.png")
    assert card_file.endswith("Z_Image_Turbo_科幻機甲_測試角色.md")
