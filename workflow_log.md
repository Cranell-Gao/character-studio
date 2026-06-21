# HW7 Agent 協作紀錄

## 1. 作業需求

本作業要求完成一個生成式 AI 專題，內容需包含：

- 可互動展示介面 App
- 原始碼與依賴環境設定
- `README.md`
- Agent workflow log
- 展示素材
- 公開 GitHub repository 連結，並寫入 `314832005_HW7.txt`

技術要求需具體應用 LLM、Diffusion Models 或 Flow Matching。本專題最後選擇同時展示 LLM 與 Diffusion。

## 2. 初始環境與目錄要求

使用者指定：

```text
請協助我完成本次的作業，以HW7為根目錄，創建一個全新的python環境(conda)用以執行作業的程式需求，作業要求如目錄中的Requirements，並與我規劃詳細作業內容
```

探索結果：

- 工作根目錄：`/home/cranell/Desktop/HW`
- 作業需求檔案：`/home/cranell/Desktop/HW/Requirements.txt`
- 專案根目錄：`/home/cranell/Desktop/HW/HW7`
- 學號：`314832005`
- GitHub Repository：[Cranell-Gao/character-studio](https://github.com/Cranell-Gao/character-studio)

本機環境：

- Conda：`/home/cranell/miniforge3/bin/conda`
- 新環境：`hw7-character-studio`
- Ollama 已安裝。
- 本機 Ollama 模型：
  - `gemma4:12b`
  - `llama3.2-vision:latest`
  - `llama3.1:latest`
- GPU：
  - NVIDIA GeForce RTX 4080
  - 16GB VRAM
  - Driver `580.159.03`

## 3. 題目形成過程

一開始討論過本機 Ollama 聊天工具，但使用者後續提出：

```text
我目前的電腦已經安裝了ollama，並已經有Gemma4 12b的模型，我想要做LLM+diffusion結合的題目
```

接著補充：

```text
顯示卡為RTX 4080 16GB
```

因此專題改為 LLM + Diffusion 角色設計系統。最終題目：

> AI Character Design Studio

核心流程：

1. 使用者輸入角色概念或選擇範例。
2. Ollama `gemma4:12b` 生成角色設定與英文 diffusion prompt。
3. 使用者選擇生成模式：
   - SDXL + ControlNet Depth
   - Z-Image Turbo
4. App 生成角色圖。
5. App 匯出角色卡與圖片。

## 4. 關鍵使用者 Prompt 紀錄

### 建立 HW7 專案與 conda 環境

```text
請協助我完成本次的作業，以HW7為根目錄，創建一個全新的python環境(conda)用以執行作業的程式需求，作業要求如目錄中的Requirements，並與我規劃詳細作業內容
```

### 提供學號

```text
314832005
```

### 指定本機 Ollama 與 LLM + Diffusion 題目

```text
我目前的電腦已經安裝了ollama，並已經有Gemma4 12b的模型，我想要做LLM+diffusion結合的題目
```

### 指定硬體

```text
顯示卡為RTX 4080 16GB
```

### 要求實作計畫

```text
PLEASE IMPLEMENT THIS PLAN:
# HW7：LLM + Diffusion 角色設計工作台
...
```

### 整體核心流程 Prompt

以下整理為本專題最終可重現的核心任務 Prompt，等同於將本次作業需求、環境條件、模型選擇與實作流程整合後交給 Agent 的完整指令：

```text
請以 /home/cranell/Desktop/HW/HW7 作為專案根目錄，完成一個 HW7 生成式 AI 專題，專題名稱為「AI Character Design Studio」。

請建立並使用新的 conda 環境 hw7-character-studio，Python 3.11，並提供 requirements.txt 與 environment.yml，讓專案可以在本機重現。

本機條件如下：
- 使用者學號：314832005
- 作業根目錄：/home/cranell/Desktop/HW/HW7
- 本機已安裝 Ollama
- 本機已有 Ollama 模型 gemma4:12b
- GPU 為 NVIDIA RTX 4080 16GB
- GitHub repository 為 https://github.com/Cranell-Gao/character-studio

請實作一個 Gradio App，核心流程如下：
1. 使用者輸入角色概念，或從內建中文角色概念範例中選擇。
2. 使用者選擇中文美術風格，例如科幻機甲、奇幻 RPG、黑暗奇幻、動漫遊戲美術、寫實概念設計。
3. App 呼叫本機 Ollama gemma4:12b，請 Gemma 扮演資深遊戲角色美術總監與 prompt engineer。
4. Gemma 必須回傳結構化 JSON，包含角色名稱、角色定位、背景故事、角色能力、服裝設計、色彩配置、英文 visual prompt 與英文 negative prompt。
5. 角色卡內容使用繁體中文；diffusion prompt 與 negative prompt 使用英文，以提升圖像模型穩定性。
6. 使用者可選擇兩種影像生成模式：
   - SDXL + ControlNet Depth：使用 stabilityai/stable-diffusion-xl-base-1.0 與 diffusers/controlnet-depth-sdxl-1.0，支援上傳姿勢 / 構圖參考圖，並轉為 depth-like control image。
   - Z-Image Turbo：使用 Tongyi-MAI/Z-Image-Turbo 作為高品質文字生圖模式，不使用本專案的 ControlNet 參考圖控制。
7. 當選擇 Z-Image Turbo 時，UI 必須隱藏參考圖、ControlNet 強度與控制圖預覽，避免誤導使用者。
8. 使用者可調整 seed、生成步數、guidance scale、ControlNet 強度、寬度、高度與 Gemma temperature。
9. App 必須顯示 Gemma 產生的繁體中文角色卡、生成圖片、ControlNet 控制圖預覽（僅 SDXL 模式）、角色卡下載與圖片下載。
10. 每次生成都必須保存具名輸出，不可覆蓋舊檔：
    - 圖片：<生成模型>_<美術風格>_<角色名稱>.png
    - 角色卡：<生成模型>_<美術風格>_<角色名稱>.md
    - 若同名已存在，請自動加上 _01、_02、_03 等代數編號。
11. Gradio 圖片預覽與「下載生成圖片」必須直接指向具名 PNG 檔案，避免顯示 Gradio 暫存檔名。
12. 輸出目錄必須固定為 HW7/outputs，不受啟動工作目錄影響。
13. app.py 不應固定只使用 7860 port；若未設定 GRADIO_SERVER_PORT，請讓 Gradio 自動尋找可用 port。若使用者指定 GRADIO_SERVER_PORT，則使用指定 port。

請建立以下模組：
- app.py：Gradio UI 與完整生成流程。
- src/ollama_client.py：呼叫 Ollama /api/chat，預設模型 gemma4:12b，並使用 keep_alive: 0s 釋放 VRAM。
- src/prompt_engine.py：建立 Gemma prompt、解析 JSON、處理中文美術風格映射與角色卡 markdown。
- src/control_image.py：將上傳參考圖轉為 depth-like control image；未上傳時產生柔和預設姿勢控制圖。
- src/diffusion_pipeline.py：SDXL + ControlNet Depth pipeline。
- src/z_image_pipeline.py：Z-Image Turbo pipeline。
- src/output_naming.py：負責輸出檔名清理、具名輸出與同名代數編號。
- scripts_smoke_test.py：提供 Ollama、diffusion import 與 Z-Image import 的 smoke test。
- tests/：加入針對 prompt parsing、control image、輸出命名、Gradio 輸出路徑與 port 設定的測試。

請撰寫繁體中文 README.md，內容需包含：
- 專題名稱
- GitHub repository 連結
- 系統架構
- 模型選擇與兩種生成模式說明
- conda 與 pip 安裝方式
- Ollama 設定
- 本地端執行步驟
- Gradio 使用方式
- 生成設定說明
- 輸出檔案命名規則
- 測試方式
- GitHub 不追蹤 outputs 的說明

請撰寫 workflow_log.md，記錄：
- 作業需求與目錄指定
- 使用者關鍵 Prompt
- Agent 協助規劃與實作過程
- 使用的工具與模型
- Gemma prompt 設計
- SDXL + ControlNet 與 Z-Image Turbo 的技術取捨
- 遇到的技術問題與修正，例如 CUDA / VRAM、Ollama keep_alive、Z-Image 不接本專案 ControlNet、Gradio 控制圖顯示、輸出檔名覆蓋、Gradio port 被占用等

請將 GitHub repository 連結寫入 314832005_HW7.txt。GitHub repository 只需要推送程式碼與說明文件，不需要追蹤 outputs 生成結果、314832005_HW7.png 或展示講稿。

最後請執行測試並確認：
- pytest 通過
- Gradio app 可以 build
- Ollama gemma4:12b 可用
- SDXL + ControlNet 與 Z-Image Turbo 至少可透過 App 完成生成流程
```

### 補上 GitHub 並要求繁體中文文件

```text
可以將專案的說明等等都換成繁體中文嗎，補上我的github，[Cranell-Gao/character-studio](https://github.com/Cranell-Gao/character-studio)
```

### 詢問模型品質並指定 Arena leaderboard

```text
圖片的品質是受到diffusion model能力的限制對嗎，可以幫我看一下這個網站中https://arena.ai/leaderboard/text-to-image?license=open-source，有什麼更強且適合我們專案的模型嗎
```

### 要求新增 Z-Image Turbo 並上傳 GitHub

```text
好，可以嘗試幫我增加，並一併將整個專案上傳到github
```

### 要求中文化 UI 與角色輸出

```text
整體我覺得非常好，不過美術風格選擇的那欄希望可以改成中文這樣
```

```text
所以如果是用Z-Image Turbo就不能接controlnet嗎，還有最下面的LLM角色介紹也都幫我弄成中文的形式，角色概念那邊也幫我多加幾組可選擇的選項進去
```

### 修正 Z-Image Turbo 模式的 ControlNet 預覽

```text
我有發現一個問題，假如說我是使用SD的話controlnet的預覽圖假如說我沒有上傳，會有預設的圖出現在控制圖預覽，但是我用Z-Image Turbo是不能接controlnet的，但是還是會顯示出來這樣，可以改掉哪
```

### 最終整理文件與 GitHub 內容

```text
可以幫我更新一下所有的文件敘述成最新的狀態嗎，包含workflow，README等等的，把我們觀察到的等等都寫進去這樣，然後我覺得推上去github的不需要我們的output，314832005_HW7.png也不用，只需要程式跟說明應該就可以了，然後關鍵prompt可以把我叫你設定的環境、專案內容的問答等等的，跟目錄指定等等德也寫進去，並最後幫我生成一個我要怎麼從零展示給大家看的流程與講稿(流程與講稿不用上傳到github)
```

## 5. App 內部 Gemma Prompt 設計

App 要求 Gemma 扮演資深遊戲角色美術總監與 prompt engineer，並回傳 compact JSON。

設計重點：

- `name`、`archetype`、`background`、`abilities`、`outfit`、`color_palette` 使用繁體中文。
- `visual_prompt` 與 `negative_prompt` 使用英文，因為圖像模型對英文 prompt 較穩定。
- prompt 要求包含 full body、centered character、readable silhouette、detailed costume、neutral background。
- parser 可從純 JSON 或 markdown code fence 中抽取 JSON。

## 6. 實作模組

- `app.py`：Gradio UI 與事件串接。
- `src/ollama_client.py`：本機 Ollama API client，並透過 `keep_alive: 0s` 釋放 Gemma VRAM。
- `src/prompt_engine.py`：角色設定 prompt、JSON parsing、中文風格選單對英文 style preset 的映射。
- `src/control_image.py`：參考圖轉 depth-like control image；沒有上傳圖時提供柔和預設姿勢圖。
- `src/diffusion_pipeline.py`：SDXL + ControlNet Depth pipeline。
- `src/z_image_pipeline.py`：Z-Image Turbo pipeline。
- `src/output_naming.py`：依生成模型、美術風格與角色名稱建立輸出檔名，重複時自動加上 `_01`、`_02`。
- `scripts_smoke_test.py`：Ollama、SDXL、Z-Image 輕量 smoke test。
- `tests/`：prompt parsing 與 control image 的單元測試。

## 7. 模型選擇與觀察

### SDXL + ControlNet Depth

優點：

- ControlNet 生態成熟。
- 可展示「受控生成」與「pipeline 客製化」。
- 適合作業要求中的 ControlNet / diffusion pipeline 技術點。

限制：

- `stable-diffusion-xl-base-1.0` 是通用 base model，角色概念圖細緻度不一定最佳。
- ControlNet strength 太高時，控制圖會壓過 prompt 創作空間。

### Z-Image Turbo

加入原因：

- 依據 Arena open-source text-to-image leaderboard 與 Hugging Face 模型資訊，Z-Image Turbo 屬於較新的高品質 open-source text-to-image 模型。
- 實測生成的角色概念圖比 SDXL base 更精緻。

限制：

- 本專案沒有將 Z-Image Turbo 接到 ControlNet。
- Z-Image 生態有 ControlNet / Union 類模型，但不是目前專案採用的標準 diffusers ControlNet 接法。
- 因此 Z-Image Turbo 在本專案中定位為「高品質文字生圖模式」。

## 8. 除錯與修正紀錄

- **Ollama socket 權限：** 沙盒環境一開始無法直接連 `127.0.0.1:11434`，後續使用提權確認 `gemma4:12b` 存在並可呼叫。
- **CUDA 可見性：** 一開始非提權下 PyTorch 看不到 CUDA；提權後確認 RTX 4080 可用。
- **SDXL + ControlNet OOM：** 第一次 diffusion smoke test 發生 CUDA OOM，原因是 Gemma 仍常駐 VRAM。修正為 Ollama request 加上 `keep_alive: 0s`。
- **預設 ControlNet 圖太強：** 早期預設圖像像火柴人，生成結果被控制圖過度影響。後續改為柔和深度人體輪廓，並降低預設 ControlNet strength。
- **Z-Image Turbo UI 誤導：** Z-Image 不使用 ControlNet，但原本仍顯示控制圖預覽。後續切換到 Z-Image Turbo 時隱藏參考圖、ControlNet 強度與控制圖預覽。
- **角色卡覆蓋問題：** 原本 `outputs/latest_character_card.md` 每次生成都會覆蓋，且相對路徑 `outputs/` 可能因啟動位置不同寫到非預期目錄。後續改為固定寫入 `HW7/outputs/`，每次生成都保存一份以模型、風格、角色名命名的 `.md` 與 `.png`，同名時自動加代數編號，不再覆寫 latest 檔。
- **Gradio 圖片檔名顯示：** 圖片預覽若直接回傳 PIL image，Gradio 會使用暫存檔名，容易看起來仍是舊命名。後續改為先保存具名 `.png`，再把檔案路徑回傳給圖片預覽與「下載生成圖片」元件。
- **Gradio port 被占用：** 原本 `app.py` 固定使用 `7860`，若該 port 已被其他服務占用會啟動失敗。後續改為未設定 `GRADIO_SERVER_PORT` 時讓 Gradio 自動尋找可用 port；需要固定 port 時仍可用環境變數指定。
- **中文化：** README、workflow log、UI、美術風格、角色卡輸出皆改為繁體中文；diffusion prompt 保持英文。

## 9. 驗證紀錄

- 建立 conda 環境：`hw7-character-studio`
- `python -m pip check`：通過
- `pytest -q`：12 tests passed
- CUDA：PyTorch 可偵測 NVIDIA GeForce RTX 4080
- Ollama smoke test：`gemma4:12b` 成功產生繁體中文角色卡
- SDXL + ControlNet smoke test：成功生成 `outputs/character_002.png`
- Z-Image Turbo import test：成功載入 `ZImagePipeline`
- Z-Image Turbo 實際生成測試：成功生成 `outputs/z_image_character_001.png`
- Gradio App build：通過
- GitHub push：已推送至 `Cranell-Gao/character-studio`

## 10. Repository 整理

最後依使用者要求，GitHub repository 只保留程式與說明，不追蹤生成輸出。

追蹤內容：

- `app.py`
- `src/`
- `tests/`
- `environment.yml`
- `requirements.txt`
- `README.md`
- `workflow_log.md`
- `314832005_HW7.txt`

不追蹤內容：

- `outputs/*.png`
- `outputs/*.md`
- `outputs/*.json`
- `314832005_HW7.png`
- 本機展示流程與講稿

## 11. 最終專案狀態

目前專案具備：

- 本機 LLM 角色設定生成
- SDXL + ControlNet 受控生成
- Z-Image Turbo 高品質文字生圖
- 繁體中文 UI
- 中文美術風格選項
- 中文角色概念範例
- 繁體中文角色卡輸出
- GitHub repository 連結檔 `314832005_HW7.txt`
