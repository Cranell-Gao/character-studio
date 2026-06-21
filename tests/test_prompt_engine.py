from src.prompt_engine import spec_from_response


def test_spec_from_json_response():
    response = """
    {
      "name": "Neon Alchemist",
      "archetype": "Cyberpunk spell engineer",
      "background": "A city exile who turns neon waste into living sigils.",
      "abilities": ["rune forging", "mechanical arm combat", "field repair"],
      "outfit": "Long reflective coat, exposed mechanical arm, utility belts.",
      "color_palette": ["cyan", "magenta", "gunmetal"],
      "visual_prompt": "full body sci-fi game character concept art",
      "negative_prompt": "low quality, blurry"
    }
    """
    spec = spec_from_response(response, concept="cyberpunk alchemist", style="sci-fi")

    assert spec.name == "Neon Alchemist"
    assert "full body" in spec.visual_prompt
    assert spec.abilities[0] == "rune forging"


def test_spec_recovers_json_inside_markdown_fence():
    response = """```json
    {"name":"Astra","archetype":"Mage","background":"A wandering mage.",
    "abilities":["starlight"],"outfit":"cloak","color_palette":["blue"],
    "visual_prompt":"full body fantasy character","negative_prompt":"text"}
    ```"""
    spec = spec_from_response(response, concept="mage", style="fantasy")

    assert spec.name == "Astra"
    assert spec.color_palette == ["blue"]

