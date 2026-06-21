from __future__ import annotations

import argparse

from src.control_image import make_depth_control_image
from src.ollama_client import OllamaClient
from src.prompt_engine import generate_character_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the HW7 character studio.")
    parser.add_argument("--ollama", action="store_true", help="Call local Ollama Gemma model.")
    parser.add_argument("--diffusion-import", action="store_true", help="Import diffusion pipeline classes.")
    args = parser.parse_args()

    control = make_depth_control_image(None, width=256, height=256)
    print(f"control image: {control.size}, {control.mode}")

    if args.ollama:
        client = OllamaClient()
        ok, message = client.healthcheck()
        print(message)
        if not ok:
            raise SystemExit(1)
        spec = generate_character_spec(
            concept="賽博龐克煉金術師",
            style="sci-fi",
            extra_notes="Use a readable full-body silhouette.",
            client=client,
            temperature=0.2,
        )
        print(spec.to_markdown())

    if args.diffusion_import:
        from src.diffusion_pipeline import CharacterDiffusionPipeline

        pipe = CharacterDiffusionPipeline()
        print(f"diffusion device: {pipe.device}")


if __name__ == "__main__":
    main()

