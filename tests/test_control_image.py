from PIL import Image

from src.control_image import default_pose_image, make_depth_control_image


def test_default_pose_size():
    image = default_pose_image(256)
    assert image.size == (256, 256)
    assert image.mode == "RGB"


def test_make_depth_control_image_from_pil():
    image = Image.new("RGB", (128, 128), "white")
    control = make_depth_control_image(image, width=256, height=256)
    assert control.size == (256, 256)
    assert control.mode == "RGB"

