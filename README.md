# AI Character Design Studio

AI Character Design Studio 是 HW7 的生成式 AI 專題，整合本機大型語言模型與擴散模型。系統使用 Ollama `gemma4:12b` 產生遊戲角色設定與 SDXL prompt，再透過 SDXL + ControlNet Depth 依照上傳的姿勢或構圖參考圖生成角色概念圖。

GitHub Repository: [Cranell-Gao/character-studio](https://github.com/Cranell-Gao/character-studio)

## 系統架構

- **LLM 層：** Ollama `gemma4:12b` 回傳結構化 JSON，內容包含角色名稱、背景、能力、服裝、色彩配置、diffusion prompt 與 negative prompt。
- **Prompt 處理層：** `src/prompt_engine.py` 負責驗證並修復 Gemma 的 JSON 輸出，最後整理成可下載的角色卡。
- **ControlNet 控制層：** `src/control_image.py` 將上傳的參考圖轉換成 depth-like control image；若未上傳圖片，系統會自動產生柔和的全身姿勢深度圖。
- **Diffusion 層：** `src/diffusion_pipeline.py` 載入 `stabilityai/stable-diffusion-xl-base-1.0` 與 `diffusers/controlnet-depth-sdxl-1.0`，使用 ControlNet 進行受控影像生成；`src/z_image_pipeline.py` 則提供 `Tongyi-MAI/Z-Image-Turbo` 高品質快速生成模式。
- **展示介面：** `app.py` 使用 Gradio 建立互動式 App，支援角色概念輸入、風格選擇、參考圖上傳、參數調整、圖片預覽與角色卡匯出。

## 環境建置

本專題以 NVIDIA RTX 4080 16GB 與已安裝 Ollama 的本機環境為主要測試平台。

```bash
cd /home/cranell/Desktop/HW/HW7
conda env create -f environment.yml
conda activate hw7-character-studio
```

若要改用 pip，請先安裝 CUDA 版 PyTorch，再安裝其他套件：

```bash
pip install -r requirements.txt
```

## Ollama 設定

若 Ollama 尚未啟動，請先執行：

```bash
ollama serve
```

確認本機模型清單：

```bash
ollama list
```

本專題預設使用以下模型：

```text
gemma4:12b
```

## 執行 App

```bash
conda activate hw7-character-studio
cd /home/cranell/Desktop/HW/HW7
python app.py
```

開啟本機 Gradio 網址，預設為：

```text
http://127.0.0.1:7860
```

若 `7860` port 已被占用，可改用：

```bash
GRADIO_SERVER_PORT=7861 python app.py
```

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

## 生成模式

- **SDXL + ControlNet Depth：** 預設模式，支援上傳姿勢或構圖參考圖，適合作為 diffusion pipeline 客製化與 ControlNet 技術展示。
- **Z-Image Turbo：** 高品質快速生成模式，使用 `Tongyi-MAI/Z-Image-Turbo`。此模式不使用參考圖控制，但通常可產生比 SDXL base 更精緻的角色概念圖。
- **美術風格：** 介面提供「奇幻 RPG」、「科幻機甲」、「黑暗奇幻」、「動漫遊戲美術」、「寫實概念設計」等中文選項；後端會自動轉換成英文 prompt style，維持模型生成品質。

## 繳交檔案

- 原始碼與環境設定檔皆包含於此 repository。
- `README.md` 說明專題名稱、系統架構與本機執行方式。
- `workflow_log.md` 記錄 Agent 協作流程、關鍵 prompt、環境探索、除錯與驗證紀錄。
- `314832005_HW7.txt` 已填入公開 GitHub repository 連結。
- `314832005_HW7.png` 為展示用生成圖片。

## 技術重點

- 使用本機 LLM `gemma4:12b` 進行角色設定生成與 prompt engineering。
- 使用 SDXL + ControlNet Depth 實作 diffusion pipeline 客製化。
- 新增 Z-Image Turbo 作為高品質 open-source text-to-image 模型選項。
- 使用 Gradio 封裝為可互動 App。
- Ollama client 會送出 `keep_alive: 0s`，讓 Gemma 回覆後釋放 VRAM，避免與 SDXL ControlNet 搶佔 RTX 4080 16GB 的 GPU 記憶體。
- 預設生成尺寸為 `768x768`；若 VRAM 壓力較高，可降至 `640x640` 或減少 inference steps。
- `stabilityai/sdxl-turbo` 可作為未來快速生成 fallback，但本專題主線採用 ControlNet，以符合擴散模型管線客製化的作業要求。
