# AI Character Design Studio

AI Character Design Studio 是 HW7 的生成式 AI 專題，整合本機大型語言模型與擴散模型，目標是讓使用者用自然語言快速生成「遊戲角色設定」與「角色概念圖」。

GitHub Repository: [Cranell-Gao/character-studio](https://github.com/Cranell-Gao/character-studio)

## 專題摘要

- **LLM：** 使用本機 Ollama `gemma4:12b` 產生角色名稱、定位、背景故事、能力、服裝、色彩配置與英文 diffusion prompt。
- **Diffusion：** 提供兩種生成模式：
  - `SDXL + ControlNet Depth`：支援姿勢 / 構圖參考圖，展示 ControlNet 受控生成。
  - `Z-Image Turbo`：高品質 open-source text-to-image 模式，適合產生更精緻的角色概念圖。
- **App：** 使用 Gradio 建立互動式介面，支援中文角色概念範例、中文美術風格選項、參考圖上傳、參數調整、角色卡匯出與圖片儲存。
- **語言設計：** 角色卡內容使用繁體中文；`Diffusion Prompt` 與 `Negative Prompt` 保持英文，以維持圖像模型生成品質。

## 系統架構

- **LLM 層：** `src/ollama_client.py` 呼叫 Ollama OpenAI-like API，並使用 `keep_alive: 0s` 讓 `gemma4:12b` 回答後釋放 VRAM。
- **Prompt 處理層：** `src/prompt_engine.py` 要求 Gemma 回傳 compact JSON，並在模型輸出包含 markdown code fence 時自動抽取 JSON。
- **ControlNet 控制層：** `src/control_image.py` 將上傳參考圖轉換成 depth-like control image；若使用 SDXL 模式但沒有上傳圖片，系統會產生柔和的預設全身姿勢控制圖。
- **SDXL 生成層：** `src/diffusion_pipeline.py` 載入 `stabilityai/stable-diffusion-xl-base-1.0` 與 `diffusers/controlnet-depth-sdxl-1.0`，用於受控角色生成。
- **Z-Image 生成層：** `src/z_image_pipeline.py` 載入 `Tongyi-MAI/Z-Image-Turbo`，用於高品質文字生圖。
- **展示介面：** `app.py` 使用 Gradio Blocks 串接 LLM、ControlNet、SDXL、Z-Image 與檔案匯出。

## 生成模式說明

### SDXL + ControlNet Depth

此模式適合展示「擴散模型管線客製化」與「受控生成」：

- 可上傳姿勢 / 構圖參考圖。
- 會顯示 ControlNet 控制圖預覽。
- 可調整 ControlNet 強度。
- 若沒有上傳參考圖，會使用內建柔和全身姿勢圖。

### Z-Image Turbo

此模式適合展示更高品質的 open-source text-to-image 生成：

- 使用 `Tongyi-MAI/Z-Image-Turbo`。
- 不使用本專案的 ControlNet 流程。
- 切換到此模式時，介面會隱藏參考圖上傳、ControlNet 強度與控制圖預覽，避免誤導。

> 註：Z-Image 生態已有 ControlNet / Union 類模型，但目前不是本專案採用的標準 diffusers ControlNet 接法。為了維持作業展示穩定性，本專案保留 SDXL + ControlNet 作為受控生成模式，Z-Image Turbo 作為高品質文字生圖模式。

## 環境建置

本專題主要測試環境：

- GPU：NVIDIA GeForce RTX 4080 16GB
- Driver：`580.159.03`
- LLM runtime：Ollama
- LLM model：`gemma4:12b`
- Python environment：conda `hw7-character-studio`

建立環境：

```bash
cd /home/cranell/Desktop/HW/HW7
conda env create -f environment.yml
conda activate hw7-character-studio
```

若改用 pip，請先安裝 CUDA 版 PyTorch，再安裝其他套件：

```bash
pip install -r requirements.txt
```

## Ollama 設定

若 Ollama 尚未啟動：

```bash
ollama serve
```

確認模型：

```bash
ollama list
```

應可看到：

```text
gemma4:12b
```

## 執行 App

```bash
conda activate hw7-character-studio
cd /home/cranell/Desktop/HW/HW7
python app.py
```

開啟：

```text
http://127.0.0.1:7860
```

若 `7860` port 已被占用：

```bash
GRADIO_SERVER_PORT=7861 python app.py
```

## 使用方式

1. 選擇生成模型：
   - `SDXL + ControlNet Depth`
   - `Z-Image Turbo`
2. 在「角色概念範例」選擇預設題材，或直接修改「角色概念」文字。
3. 選擇「美術風格」：
   - 奇幻 RPG
   - 科幻機甲
   - 黑暗奇幻
   - 動漫遊戲美術
   - 寫實概念設計
4. 可在「額外條件」補充武器、陣營、時代背景、色彩或情緒。
5. 若使用 SDXL + ControlNet，可上傳姿勢 / 構圖參考圖。
6. 調整 seed、生成步數、解析度、guidance scale 與 ControlNet 強度。
7. 按「生成角色」。
8. 查看角色卡、生成圖片、控制圖預覽，並下載角色卡。

## 介面欄位與生成設定說明

- **生成模型：** 選擇圖片生成後端。`SDXL + ControlNet Depth` 適合展示姿勢 / 構圖控制；`Z-Image Turbo` 適合產生較高品質的文字生圖結果。
- **角色概念：** 主要創作輸入，描述角色身份、世界觀、能力或視覺特徵，例如「黑暗奇幻時間刺客」或「星際遺跡獵人」。
- **角色概念範例：** 提供多組可直接套用的中文題材，選擇後會自動填入角色概念欄位，使用者仍可再修改細節。
- **美術風格：** 使用中文選項控制整體視覺方向，後端會轉成英文 style prompt，例如「科幻機甲」會轉成 sci-fi game concept art 相關描述。
- **額外條件：** 用來補充角色概念之外的限制，例如武器、陣營、色彩、情緒、年代、材質、禁止元素或構圖需求。這些文字會交給 Gemma 一起產生角色設定與 diffusion prompt。
- **姿勢 / 構圖參考圖：** 只在 `SDXL + ControlNet Depth` 模式使用。上傳後會轉成 depth-like control image，控制角色的大致姿勢與構圖。
- **Seed：** 控制隨機性。相同 prompt、模型與設定下，使用相同 seed 較容易重現類似結果。
- **生成步數：** diffusion denoising steps。步數越高通常越細緻但越慢；Z-Image Turbo 會自動限制在較低步數以符合 turbo 模型設計。
- **Guidance scale：** 控制模型遵守文字 prompt 的程度。數值太低可能偏離描述，太高可能畫面僵硬或產生 artifact。
- **ControlNet 強度：** 只在 SDXL + ControlNet 模式使用。數值越高越貼近控制圖，但太高會壓制 prompt 的創作空間。
- **寬度 / 高度：** 控制輸出解析度。RTX 4080 16GB 下建議從 `768x768` 開始；若 VRAM 不足可降到 `640x640`。
- **Gemma temperature：** 控制 LLM 角色設定的創意程度。較低會更穩定，較高會更有變化。

## 測試方式

輕量測試：

```bash
pytest
python scripts_smoke_test.py
```

Ollama 測試：

```bash
python scripts_smoke_test.py --ollama
```

Diffusion 匯入測試：

```bash
python scripts_smoke_test.py --diffusion-import
python scripts_smoke_test.py --z-image-import
```

完整影像生成請透過 Gradio 介面操作。第一次執行 SDXL ControlNet 或 Z-Image Turbo 時，若本機尚未快取權重，程式會從 Hugging Face 下載模型。

## 重要觀察與修正

- 圖片品質會受到 diffusion model 能力、prompt、解析度、steps、ControlNet 強度與是否使用專門微調模型影響。
- SDXL base 的好處是 ControlNet 生態成熟，適合展示受控生成；缺點是角色概念圖細緻度不一定最佳。
- Z-Image Turbo 在實測中能產生更精緻的角色概念圖，因此新增為高品質模式。
- 一開始 SDXL + ControlNet 曾發生 CUDA OOM，原因是 `gemma4:12b` 仍常駐 VRAM。後續加入 Ollama `keep_alive: 0s`，讓 Gemma 回答後釋放 GPU 記憶體。
- Z-Image Turbo 不使用本專案的 ControlNet，因此 UI 已改為切換到 Z-Image 時隱藏參考圖、ControlNet 強度與控制圖預覽。

## Repository 內容

GitHub repository 只追蹤程式碼與說明文件，不追蹤生成輸出：

- 追蹤：
  - `app.py`
  - `src/`
  - `tests/`
  - `environment.yml`
  - `requirements.txt`
  - `README.md`
  - `workflow_log.md`
  - `314832005_HW7.txt`
- 不追蹤：
  - `outputs/*.png`
  - `outputs/*.md`
  - `outputs/*.json`
  - `314832005_HW7.png`
  - 展示講稿與本機 demo 筆記

`314832005_HW7.txt` 內容為公開 GitHub repository 連結。
