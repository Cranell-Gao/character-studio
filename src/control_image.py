"""Reference image preprocessing for ControlNet depth conditioning."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def default_pose_image(size: int = 768) -> Image.Image:
    """Create a soft full-body depth reference when the user has no image."""
    image = Image.new("L", (size, size), 35)
    draw = ImageDraw.Draw(image)
    center_x = size // 2
    head_r = size // 14
    top = int(size * 0.18)
    shoulder_y = int(size * 0.32)
    waist_y = int(size * 0.56)
    foot_y = int(size * 0.86)
    draw.ellipse(
        (center_x - head_r, top - head_r, center_x + head_r, top + head_r),
        fill=210,
    )
    draw.rounded_rectangle(
        (int(size * 0.40), shoulder_y, int(size * 0.60), waist_y),
        radius=size // 20,
        fill=185,
    )
    draw.polygon(
        [
            (int(size * 0.40), shoulder_y),
            (int(size * 0.26), int(size * 0.49)),
            (int(size * 0.32), int(size * 0.55)),
            (int(size * 0.45), int(size * 0.38)),
        ],
        fill=165,
    )
    draw.polygon(
        [
            (int(size * 0.60), shoulder_y),
            (int(size * 0.74), int(size * 0.49)),
            (int(size * 0.68), int(size * 0.55)),
            (int(size * 0.55), int(size * 0.38)),
        ],
        fill=165,
    )
    draw.polygon(
        [
            (int(size * 0.45), waist_y),
            (int(size * 0.35), foot_y),
            (int(size * 0.43), foot_y),
            (int(size * 0.51), waist_y),
        ],
        fill=175,
    )
    draw.polygon(
        [
            (int(size * 0.55), waist_y),
            (int(size * 0.65), foot_y),
            (int(size * 0.57), foot_y),
            (int(size * 0.49), waist_y),
        ],
        fill=175,
    )
    array = cv2.GaussianBlur(np.array(image), (31, 31), 0)
    return Image.fromarray(array).convert("RGB")


def _to_pil(image: Image.Image | np.ndarray | str | Path | None) -> Image.Image:
    if image is None:
        return default_pose_image()
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return Image.open(image).convert("RGB")


def make_depth_control_image(
    image: Image.Image | np.ndarray | str | Path | None,
    width: int = 768,
    height: int = 768,
) -> Image.Image:
    """Build a depth-like control map without requiring an extra depth model.

    This uses grayscale contrast, blur, and edge emphasis. It is intentionally
    lightweight so the app still runs if MiDaS/ZoeDepth weights are unavailable.
    """
    pil = _to_pil(image).resize((width, height), Image.Resampling.LANCZOS)
    if image is None:
        return pil
    array = np.array(pil)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, threshold1=70, threshold2=150)
    depth_like = cv2.addWeighted(blurred, 0.82, 255 - edges, 0.18, 0)
    depth_rgb = cv2.cvtColor(depth_like, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb)
