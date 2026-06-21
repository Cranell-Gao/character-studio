from app import OUTPUT_DIR, PROJECT_DIR


def test_output_dir_is_fixed_to_project_outputs():
    assert OUTPUT_DIR.is_absolute()
    assert OUTPUT_DIR == PROJECT_DIR / "outputs"
