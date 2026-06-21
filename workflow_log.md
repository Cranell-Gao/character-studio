# HW7 Agent 協作紀錄

## 1. 需求分析

本作業要求完成一個生成式 AI 專題，必須包含可互動 App、原始碼、README、Agent workflow log 與展示素材。技術面需具體應用 LLM、Diffusion Models 或 Flow Matching，並鼓勵使用 Agent 輔助完成題目發想、架構設計、程式碼生成與除錯。

最初方向是本機 Ollama 聊天工具。後續依照課程要求與展示完整度，將題目升級為 LLM + Diffusion 結合的角色設計工作台。

## 2. 環境探索

- 工作根目錄：`/home/cranell/Desktop/HW`
- HW7 專案目錄：`/home/cranell/Desktop/HW/HW7`
- 學號：`314832005`
- GitHub Repository：[Cranell-Gao/character-studio](https://github.com/Cranell-Gao/character-studio)
- Ollama 已安裝。
- 本機 Ollama 模型：
  - `gemma4:12b`
  - `llama3.2-vision:latest`
  - `llama3.1:latest`
- GPU：
  - NVIDIA GeForce RTX 4080
  - 16GB VRAM
  - Driver `580.159.03`
- Hugging Face cache 已包含 SDXL Turbo 與 SD1.5 相關資源。最初主線選擇 SDXL + SDXL ControlNet Depth，以取得較佳影像品質與更明確的 diffusion 技術展示；後續依據 Arena open-source text-to-image leaderboard 與 Hugging Face 模型資訊，新增 `Tongyi-MAI/Z-Image-Turbo` 作為高品質快速生成模式。

## 3. 專題規劃

選定專題：

> AI Character Design Studio

核心流程：

1. 使用者輸入角色概念。
2. Gemma4 12B 產生結構化角色設定與 SDXL prompt。
3. 使用者可上傳姿勢或構圖參考圖。
4. App 將參考圖轉換為 depth-like control image。
5. 使用者可選擇 SDXL + ControlNet Depth 或 Z-Image Turbo 生成角色概念圖。
6. App 匯出角色卡 Markdown 與生成圖片。

## 4. 關鍵 Prompt 紀錄

### 題目規劃 Prompt

```text
我目前的電腦已經安裝了ollama，並已經有Gemma4 12b的模型，我想要做LLM+diffusion結合的題目
```

### 實作 Prompt

```text
PLEASE IMPLEMENT THIS PLAN:
# HW7：LLM + Diffusion 角色設計工作台
...
```

### App 內部 Gemma System Prompt 摘要

App 要求 Gemma 扮演資深遊戲角色美術總監與 prompt engineer，並回傳 compact JSON。JSON 內容包含角色名稱、角色定位、背景故事、能力、服裝、色彩配置、SDXL visual prompt 與 negative prompt。

## 5. 實作重點

- `src/ollama_client.py`：處理本機 Ollama API 呼叫與健康檢查。
- `src/prompt_engine.py`：建立 LLM prompt，並在模型輸出含 markdown fence 時仍能抽取 JSON。
- `src/prompt_engine.py` 同時提供中文美術風格選項與英文 style preset 的對應，讓介面中文化但不犧牲 prompt 品質。
- `src/control_image.py`：產生 depth-like control image；若沒有上傳參考圖，會產生柔和的預設全身姿勢圖。
- `src/diffusion_pipeline.py`：載入 SDXL + ControlNet，使用 fp16、attention slicing 與 CPU offload 降低 VRAM 壓力。
- `src/z_image_pipeline.py`：載入 Z-Image Turbo，提供不依賴 ControlNet 的高品質快速生成模式。
- `app.py`：使用 Gradio Blocks 封裝完整互動式介面。

## 6. 風險與解法

- **第一次 diffusion 可能下載權重：** README 已註明 SDXL ControlNet 權重可能會從 Hugging Face 下載。
- **VRAM 壓力：** 預設使用 `768x768`；若 GPU 記憶體不足，可降到 `640x640` 或減少 steps。
- **LLM 與 diffusion 共用 VRAM：** Ollama client 在回覆後送出 `keep_alive: 0s`，讓 `gemma4:12b` 釋放 VRAM；diffusion 端也啟用 PyTorch expandable CUDA segments。
- **LLM JSON 格式不穩：** prompt parser 可從純 JSON 或 markdown code fence 中抽取 JSON。
- **Ollama 服務未啟動：** App 介面提供系統狀態檢查，會顯示 Ollama 或 `gemma4:12b` 是否可用。

## 7. 驗證紀錄

- 已建立 conda 環境：`hw7-character-studio`。
- `python -m pip check`：通過。
- `pytest -q`：4 個測試通過。
- CUDA 驗證：PyTorch 可偵測 NVIDIA GeForce RTX 4080。
- Ollama smoke test：`gemma4:12b` 成功產生有效角色卡。
- 第一次 SDXL ControlNet 生成曾發生 CUDA OOM，原因是 Gemma 仍常駐 VRAM。
- 修正方式：加入 Ollama `keep_alive: 0s`、降低預設 ControlNet strength，並改用較柔和的預設 depth pose。
- 最終 diffusion smoke test 成功，生成 `outputs/character_002.png`。
- 已將最終展示圖複製為 `314832005_HW7.png`。
- 新增 Z-Image Turbo 匯入測試，確認目前 diffusers 環境支援 `ZImagePipeline`。
- Z-Image Turbo 實際生成測試成功，輸出 `outputs/z_image_character_001.png`。
- 因 Z-Image Turbo 展示圖品質較佳，已更新 `314832005_HW7.png` 作為最終展示素材。

## 8. 最終交付項目

- `app.py`
- `src/`
- `environment.yml`
- `requirements.txt`
- `README.md`
- `workflow_log.md`
- `314832005_HW7.txt`
- `314832005_HW7.png`
