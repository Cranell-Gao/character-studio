from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from src.control_image import make_depth_control_image
from src.diffusion_pipeline import CharacterDiffusionPipeline, GenerationConfig, save_image
from src.ollama_client import OllamaClient
from src.prompt_engine import STYLE_PRESETS, CharacterSpec, generate_character_spec


OUTPUT_DIR = Path("outputs")
CARD_PATH = OUTPUT_DIR / "latest_character_card.md"

diffusion = CharacterDiffusionPipeline()
ollama = OllamaClient()


def check_system() -> str:
    ok, message = ollama.healthcheck()
    status = "Ready" if ok else "Needs attention"
    return f"**{status}:** {message}"


def _write_card(spec: CharacterSpec, image_path: Path | None) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    body = spec.to_markdown()
    if image_path is not None:
        body += f"\n## Generated Image\n{image_path.as_posix()}\n"
    CARD_PATH.write_text(body, encoding="utf-8")
    return CARD_PATH.as_posix()


def generate_character(
    concept: str,
    style: str,
    extra_notes: str,
    reference_image: Any,
    seed: int,
    steps: int,
    guidance_scale: float,
    control_strength: float,
    width: int,
    height: int,
    llm_temperature: float,
) -> tuple[str, Any, Any, str, str]:
    try:
        spec = generate_character_spec(
            concept=concept,
            style=style,
            extra_notes=extra_notes,
            client=ollama,
            temperature=llm_temperature,
        )
        control_image = make_depth_control_image(reference_image, width=int(width), height=int(height))
        config = GenerationConfig(
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(control_strength),
            seed=int(seed),
        )
        image = diffusion.generate(
            prompt=spec.visual_prompt,
            negative_prompt=spec.negative_prompt,
            control_image=control_image,
            config=config,
        )
        image_path = save_image(image, OUTPUT_DIR)
        card_file = _write_card(spec, image_path)
        return spec.to_markdown(), image, control_image, card_file, image_path.as_posix()
    except Exception as exc:
        error_markdown = f"## Generation failed\n\n```text\n{exc}\n```"
        return error_markdown, None, None, "", ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI Character Design Studio") as demo:
        gr.Markdown(
            "# AI Character Design Studio\n"
            "Gemma4 12B plans the character. SDXL + ControlNet turns the design into game concept art."
        )

        status = gr.Markdown(value=check_system)
        refresh = gr.Button("Refresh system status", size="sm")
        refresh.click(fn=check_system, outputs=status)

        with gr.Row():
            with gr.Column(scale=5):
                concept = gr.Textbox(
                    label="Character concept",
                    value="賽博龐克煉金術師，使用霓虹符文與機械義肢戰鬥",
                    lines=3,
                )
                style = gr.Dropdown(
                    label="Art style",
                    choices=list(STYLE_PRESETS.keys()),
                    value="sci-fi",
                )
                extra_notes = gr.Textbox(
                    label="Extra constraints",
                    placeholder="Optional: weapon, mood, color, faction, era...",
                    lines=3,
                )
                reference_image = gr.Image(
                    label="Pose / composition reference",
                    type="pil",
                    sources=["upload", "clipboard"],
                )

                with gr.Accordion("Generation settings", open=True):
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        steps = gr.Slider(12, 40, value=25, step=1, label="Steps")
                    with gr.Row():
                        guidance = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Guidance scale")
                        control = gr.Slider(0.1, 1.5, value=0.55, step=0.05, label="Control strength")
                    with gr.Row():
                        width = gr.Dropdown([640, 768, 896, 1024], value=768, label="Width")
                        height = gr.Dropdown([640, 768, 896, 1024], value=768, label="Height")
                    llm_temperature = gr.Slider(0.1, 1.2, value=0.75, step=0.05, label="Gemma temperature")

                generate = gr.Button("Generate character", variant="primary")

            with gr.Column(scale=6):
                output_image = gr.Image(label="Generated character", type="pil", format="png")
                control_preview = gr.Image(label="Control image preview", type="pil", format="png")

        spec_markdown = gr.Markdown(label="Character card")
        with gr.Row():
            card_file = gr.File(label="Download character card")
            image_path = gr.Textbox(label="Saved image path", interactive=False)

        generate.click(
            fn=generate_character,
            inputs=[
                concept,
                style,
                extra_notes,
                reference_image,
                seed,
                steps,
                guidance,
                control,
                width,
                height,
                llm_temperature,
            ],
            outputs=[spec_markdown, output_image, control_preview, card_file, image_path],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    build_demo().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=port)
