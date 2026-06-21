"""SDXL + ControlNet image generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL = "diffusers/controlnet-depth-sdxl-1.0"


@dataclass(frozen=True)
class GenerationConfig:
    width: int = 768
    height: int = 768
    steps: int = 25
    guidance_scale: float = 7.0
    controlnet_conditioning_scale: float = 0.8
    seed: int = 42


class CharacterDiffusionPipeline:
    def __init__(
        self,
        base_model: str = BASE_MODEL,
        controlnet_model: str = CONTROLNET_MODEL,
        device: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe: StableDiffusionXLControlNetPipeline | None = None

    def load(self) -> None:
        if self.pipe is not None:
            return

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe.enable_attention_slicing()
        if self.device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe.to(self.device)
        else:
            pipe.to(self.device)
        self.pipe = pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: Image.Image,
        config: GenerationConfig,
    ) -> Image.Image:
        self.load()
        if self.pipe is None:
            raise RuntimeError("Pipeline failed to load.")

        generator_device = "cuda" if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(config.seed))
        control_image = control_image.resize((config.width, config.height), Image.Resampling.LANCZOS)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            width=config.width,
            height=config.height,
            num_inference_steps=int(config.steps),
            guidance_scale=float(config.guidance_scale),
            controlnet_conditioning_scale=float(config.controlnet_conditioning_scale),
            generator=generator,
        )
        return result.images[0]


def save_image(image: Image.Image, output_dir: str | Path = "outputs") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob("character_*.png"))
    next_id = len(existing) + 1
    out_path = path / f"character_{next_id:03d}.png"
    image.save(out_path)
    return out_path
