"""Z-Image Turbo text-to-image generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import ZImagePipeline
from PIL import Image


Z_IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"


@dataclass(frozen=True)
class ZImageGenerationConfig:
    width: int = 1024
    height: int = 1024
    steps: int = 8
    guidance_scale: float = 0.0
    seed: int = 42


class CharacterZImagePipeline:
    def __init__(self, model_id: str = Z_IMAGE_MODEL, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe: ZImagePipeline | None = None

    def load(self) -> None:
        if self.pipe is not None:
            return

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
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
        config: ZImageGenerationConfig,
    ) -> Image.Image:
        self.load()
        if self.pipe is None:
            raise RuntimeError("Z-Image pipeline failed to load.")

        generator_device = "cuda" if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(config.seed))
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(config.width),
            height=int(config.height),
            num_inference_steps=int(config.steps),
            guidance_scale=float(config.guidance_scale),
            generator=generator,
        )
        return result.images[0]


def save_z_image(image: Image.Image, output_dir: str | Path = "outputs") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob("z_image_character_*.png"))
    next_id = len(existing) + 1
    out_path = path / f"z_image_character_{next_id:03d}.png"
    image.save(out_path)
    return out_path

