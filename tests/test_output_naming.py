from PIL import Image

from src.output_naming import next_available_path, output_stem, save_named_image, slugify


def test_slugify_keeps_chinese_and_normalizes_symbols():
    assert slugify("Z-Image Turbo / 科幻機甲") == "Z_Image_Turbo_科幻機甲"


def test_output_stem_uses_model_style_and_character():
    stem = output_stem("SDXL + ControlNet Depth", "黑暗奇幻", "幻影煉金師")
    assert stem == "SDXL_ControlNet_Depth_黑暗奇幻_幻影煉金師"


def test_next_available_path_adds_generation_suffix(tmp_path):
    first = next_available_path(tmp_path, "z_image_角色", ".md")
    first.write_text("one", encoding="utf-8")
    second = next_available_path(tmp_path, "z_image_角色", ".md")
    second.write_text("two", encoding="utf-8")
    third = next_available_path(tmp_path, "z_image_角色", ".md")

    assert second.name == "z_image_角色_01.md"
    assert third.name == "z_image_角色_02.md"


def test_save_named_image_uses_same_suffix_rule(tmp_path):
    image = Image.new("RGB", (16, 16), "white")
    first = save_named_image(image, "demo", tmp_path)
    second = save_named_image(image, "demo", tmp_path)

    assert first.name == "demo.png"
    assert second.name == "demo_01.png"

