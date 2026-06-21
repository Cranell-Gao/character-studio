# HW7 Workflow Log

## 1. Requirement Analysis

The assignment requires a generative AI project with a functional App, source code, README, workflow log, and demonstration material. The technical requirement asks for LLM usage and/or diffusion or flow matching usage, with a strong preference for agent-assisted development.

Initial direction was a local Ollama chat tool. After discussion, the scope was upgraded to an LLM + diffusion system so the final project can demonstrate both major course topics.

## 2. Environment Discovery

- Working root: `/home/cranell/Desktop/HW`
- Student ID: `314832005`
- Ollama is installed.
- Local Ollama models:
  - `gemma4:12b`
  - `llama3.2-vision:latest`
  - `llama3.1:latest`
- GPU:
  - NVIDIA GeForce RTX 4080
  - 16GB VRAM
  - Driver `580.159.03`
- Hugging Face cache already contains SDXL Turbo and SD1.5-related assets. The final plan chooses SDXL + SDXL ControlNet Depth for better quality and stronger diffusion technique demonstration.

## 3. Project Planning with Agent

Selected project:

> AI Character Design Studio

Core idea:

1. User enters a character concept.
2. Gemma4 12B generates a structured character design and SDXL prompt.
3. User optionally uploads a reference pose or composition image.
4. The app converts the reference into a depth-like control image.
5. SDXL + ControlNet Depth generates a full-body game character concept image.
6. The app exports a character card and generated image.

## 4. Key Prompts Used

### User Planning Prompt

```text
我目前的電腦已經安裝了ollama，並已經有Gemma4 12b的模型，我想要做LLM+diffusion結合的題目
```

### Implementation Prompt

```text
PLEASE IMPLEMENT THIS PLAN:
# HW7：LLM + Diffusion 角色設計工作台
...
```

### System Prompt for Gemma

The app asks Gemma to behave as a senior game character art director and prompt engineer. It must return compact JSON with character metadata, visual prompt, and negative prompt.

## 5. Implementation Notes

- `src/ollama_client.py` handles local Ollama API calls and health checks.
- `src/prompt_engine.py` builds the LLM prompt and recovers JSON even when the model returns markdown fences.
- `src/control_image.py` creates a lightweight depth-like control image using grayscale equalization, blur, and edge emphasis. This keeps the app usable even without an extra depth-estimation model.
- `src/diffusion_pipeline.py` loads SDXL + ControlNet with fp16 and memory-saving settings.
- `app.py` wires everything into a Gradio Blocks interface.

## 6. Known Risks and Mitigations

- **First diffusion run may download weights:** README documents that SDXL ControlNet weights may need to download from Hugging Face.
- **VRAM pressure:** Defaults use `768x768`; users can reduce to `640x640`.
- **LLM and diffusion sharing VRAM:** Gemma is unloaded after each Ollama response with `keep_alive: 0s`, and diffusion enables PyTorch expandable CUDA segments.
- **LLM JSON formatting:** The prompt parser extracts JSON from plain text or markdown fences.
- **Ollama service unavailable:** The UI has a system status check that explains whether Ollama or `gemma4:12b` is missing.

## 7. Verification

- Created conda environment `hw7-character-studio`.
- `python -m pip check`: passed.
- `pytest -q`: 4 tests passed.
- CUDA verification with elevated local access: PyTorch sees NVIDIA GeForce RTX 4080.
- Ollama smoke test: `gemma4:12b` generated a valid character card.
- First SDXL ControlNet generation hit CUDA OOM because Gemma was still resident in VRAM.
- Fixed by adding Ollama `keep_alive: 0s`, lowering the default ControlNet strength, and using a softer default depth pose.
- Final diffusion smoke test succeeded and generated `outputs/character_002.png`.
- Copied the final generated sample to `314832005_HW7.png` for demonstration submission.

## 8. Final Deliverables

- `app.py`
- `src/`
- `environment.yml`
- `requirements.txt`
- `README.md`
- `workflow_log.md`
- `314832005_HW7.txt`
- Demonstration image: `314832005_HW7.png`
