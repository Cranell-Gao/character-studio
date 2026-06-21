# AI Character Design Studio

AI Character Design Studio is an HW7 generative AI project that combines a local LLM with a diffusion model. The app uses Ollama `gemma4:12b` to design a game character and produce an SDXL prompt, then uses SDXL + ControlNet Depth to generate a character concept image from an optional pose or composition reference.

## Architecture

- **LLM layer:** Ollama `gemma4:12b` returns structured JSON with character name, background, abilities, outfit, color palette, diffusion prompt, and negative prompt.
- **Prompt layer:** `src/prompt_engine.py` validates and recovers the JSON output, then formats a downloadable character card.
- **Control layer:** `src/control_image.py` converts an uploaded reference image into a depth-like control image. If no image is uploaded, it creates a simple full-body pose map.
- **Diffusion layer:** `src/diffusion_pipeline.py` loads `stabilityai/stable-diffusion-xl-base-1.0` with `diffusers/controlnet-depth-sdxl-1.0`.
- **Interface:** `app.py` provides a Gradio UI for concept input, style selection, ControlNet reference upload, generation settings, image preview, and character card export.

## Environment Setup

This project was planned for an NVIDIA RTX 4080 16GB machine with Ollama already installed.

```bash
cd /home/cranell/Desktop/HW/HW7
conda env create -f environment.yml
conda activate hw7-character-studio
```

If you prefer pip, install CUDA PyTorch first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Ollama Setup

Start Ollama if it is not already running:

```bash
ollama serve
```

Confirm the local model:

```bash
ollama list
```

The app expects this model id:

```text
gemma4:12b
```

## Run the App

```bash
conda activate hw7-character-studio
cd /home/cranell/Desktop/HW/HW7
python app.py
```

Open the local Gradio URL, usually:

```text
http://127.0.0.1:7860
```

## Testing

Lightweight tests:

```bash
pytest
python scripts_smoke_test.py
```

Ollama smoke test:

```bash
python scripts_smoke_test.py --ollama
```

Diffusion import smoke test:

```bash
python scripts_smoke_test.py --diffusion-import
```

Full generation is run through the Gradio interface. The first SDXL ControlNet run may download Hugging Face weights if they are not already cached.

## Submission Files

- Source code and environment files are in this repository.
- `workflow_log.md` records the agent-assisted development process.
- `314832005_HW7.txt` should contain the public GitHub repository URL before submission.
- Save a screenshot or recording as `314832005_HW7.png` or `314832005_HW7.mp4`.

## Notes

- Default generation size is `768x768`, which is a practical baseline for RTX 4080 16GB.
- If VRAM is tight, lower width and height to `640x640`, reduce steps, or keep CPU offload enabled.
- The Ollama client sends `keep_alive: 0s` so `gemma4:12b` is unloaded after prompt generation and SDXL ControlNet can use the GPU memory.
- `stabilityai/sdxl-turbo` can be used as a future fallback for faster text-to-image generation, but the main project uses ControlNet to satisfy the diffusion pipeline customization requirement.
