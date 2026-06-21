from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from src.control_image import make_depth_control_image
from src.diffusion_pipeline import CharacterDiffusionPipeline, GenerationConfig, save_image
from src.ollama_client import OllamaClient
from src.prompt_engine import STYLE_PRESETS, CharacterSpec, generate_character_spec
from src.z_image_pipeline import CharacterZImagePipeline, ZImageGenerationConfig, save_z_image


OUTPUT_DIR = Path("outputs")
CARD_PATH = OUTPUT_DIR / "latest_character_card.md"

diffusion = CharacterDiffusionPipeline()
z_image = CharacterZImagePipeline()
ollama = OllamaClient()


def check_system() -> str:
    ok, message = ollama.healthcheck()
    status = "系統就緒" if ok else "需要檢查"
    return f"**{status}:** {message}"


def _write_card(spec: CharacterSpec, image_path: Path | None) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    body = spec.to_markdown()
    if image_path is not None:
        body += f"\n## 生成圖片\n{image_path.as_posix()}\n"
    CARD_PATH.write_text(body, encoding="utf-8")
    return CARD_PATH.as_posix()


def generate_character(
    image_model: str,
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

        if image_model == "Z-Image Turbo":
            control_image = make_depth_control_image(reference_image, width=int(width), height=int(height))
            config = ZImageGenerationConfig(
                width=int(width),
                height=int(height),
                steps=min(int(steps), 16),
                guidance_scale=0.0,
                seed=int(seed),
            )
            image = z_image.generate(
                prompt=spec.visual_prompt,
                negative_prompt=spec.negative_prompt,
                config=config,
            )
            image_path = save_z_image(image, OUTPUT_DIR)
        else:
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
        error_markdown = f"## 生成失敗\n\n```text\n{exc}\n```"
        return error_markdown, None, None, "", ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI Character Design Studio") as demo:
        gr.Markdown(
            "# AI Character Design Studio\n"
            "使用 Gemma4 12B 產生角色設定，再透過 SDXL + ControlNet 生成遊戲角色概念圖。"
        )

        status = gr.Markdown(value=check_system)
        refresh = gr.Button("重新檢查系統狀態", size="sm")
        refresh.click(fn=check_system, outputs=status)

        with gr.Row():
            with gr.Column(scale=5):
                image_model = gr.Radio(
                    label="生成模型",
                    choices=["SDXL + ControlNet Depth", "Z-Image Turbo"],
                    value="SDXL + ControlNet Depth",
                    info="ControlNet 模式支援姿勢/構圖控制；Z-Image Turbo 模式畫質較佳但不使用參考圖控制。",
                )
                concept = gr.Textbox(
                    label="角色概念",
                    value="賽博龐克煉金術師，使用霓虹符文與機械義肢戰鬥",
                    lines=3,
                )
                style = gr.Dropdown(
                    label="美術風格",
                    choices=list(STYLE_PRESETS.keys()),
                    value="sci-fi",
                )
                extra_notes = gr.Textbox(
                    label="額外條件",
                    placeholder="可選：武器、情緒、色彩、陣營、時代背景...",
                    lines=3,
                )
                reference_image = gr.Image(
                    label="姿勢 / 構圖參考圖",
                    type="pil",
                    sources=["upload", "clipboard"],
                )

                with gr.Accordion("生成設定", open=True):
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        steps = gr.Slider(12, 40, value=25, step=1, label="生成步數")
                    with gr.Row():
                        guidance = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Guidance scale")
                        control = gr.Slider(0.1, 1.5, value=0.55, step=0.05, label="ControlNet 強度")
                    with gr.Row():
                        width = gr.Dropdown([640, 768, 896, 1024], value=768, label="寬度")
                        height = gr.Dropdown([640, 768, 896, 1024], value=768, label="高度")
                    llm_temperature = gr.Slider(0.1, 1.2, value=0.75, step=0.05, label="Gemma temperature")

                generate = gr.Button("生成角色", variant="primary")

            with gr.Column(scale=6):
                output_image = gr.Image(label="生成角色圖", type="pil", format="png")
                control_preview = gr.Image(label="ControlNet 控制圖預覽", type="pil", format="png")

        spec_markdown = gr.Markdown(label="角色卡")
        with gr.Row():
            card_file = gr.File(label="下載角色卡")
            image_path = gr.Textbox(label="圖片儲存路徑", interactive=False)

        generate.click(
            fn=generate_character,
            inputs=[
                image_model,
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
