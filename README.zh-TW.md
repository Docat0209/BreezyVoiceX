[English](README.md) | [繁體中文](README.zh-TW.md)

# BreezyVoiceX

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

> 基於聯發科 [BreezyVoice](https://github.com/mtkresearch/BreezyVoice)。

## 什麼是 BreezyVoiceX？

零樣本語音克隆 TTS 系統，專為**台灣腔中文**設計。只需一段短音訊，即可生成該說話者聲音的自然語音 — 支援注音（bopomofo）音素控制。

BreezyVoiceX 封裝了聯發科的 [BreezyVoice](https://github.com/mtkresearch/BreezyVoice)，提供簡化的兩步驟流程（快取說話者 → 合成語音）、Windows 支援與效能分析工具。無需 Linux 限定的依賴套件。

## 與 BreezyVoice 的差異
- 透過 Prompt 快取實現快速零樣本語音合成
- 內建各推論步驟的時間分析器
- 完全移除 Linux 限定的 ttsfrd 依賴，跨平台可用

## 安裝

> 需要 Python 3.11。GPU 使用者建議 CUDA 12.1。

### 取得原始碼
```bash
git clone https://github.com/Docat0209/BreezyVoiceX.git
cd BreezyVoiceX
```

### Linux
```bash
pip install -r requirements.txt
```

### Windows
```bash
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install WeTextProcessing --no-deps
```

## 推論

需要 UTF8 編碼：

```sh
export PYTHONUTF8=1
```

---
> 此版本將流程拆分為兩個明確步驟

**以下列參數執行 single_inference.py：**

### `--mode cache`（產生說話者 Prompt 快取）
| 參數 | 說明 |
|------|------|
| `--speaker_prompt_audio_path` | 必填。說話者參考音訊路徑。 |
| `--speaker_prompt_text_transcription` | 選填。手動轉錄文字。未提供時使用 Whisper 自動辨識。 |
| `--prompt_feature_path` | 選填。快取輸出路徑。預設：`cache/prompt.pt`。 |
| `--model_path` | 選填。HuggingFace 模型 ID 或本地路徑。預設：`MediaTek-Research/BreezyVoice-300M`。 |

### `--mode synthesize`（合成語音）

| 參數 | 說明 |
|------|------|
| `--content_to_synthesize` | 必填。要合成的目標文字。 |
| `--prompt_feature_path` | 必填。先前儲存的說話者快取（`.pt`）路徑。 |
| `--output_path` | 選填。輸出 WAV 檔路徑。預設：`results/output.wav`。 |
| `--model_path` | 選填。HuggingFace 模型 ID 或本地路徑。預設：`MediaTek-Research/BreezyVoice-300M`。 |

**使用範例：**

### 步驟一：快取說話者 Prompt
```bash
python single_inference.py --mode cache --speaker_prompt_audio_path data/example.wav --prompt_feature_path cache/example.pt
```

### 步驟二：從文字合成語音
```bash
python single_inference.py --mode synthesize --content_to_synthesize "您好，這是一段生成測試語音。" --prompt_feature_path cache/example.pt --output_path results/output.wav
```

## 致謝

本專案基於聯發科的 [BreezyVoice](https://github.com/mtkresearch/BreezyVoice)，
一套針對台灣中文設計的語音克隆 TTS 系統，支援注音（bopomofo）音素控制。
原始專案部分衍生自 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)，屬於 [Breeze2 模型家族](https://huggingface.co/collections/MediaTek-Research/breeze2-family-67863158443a06a72dd29900)。

感謝原始作者的貢獻。本儲存庫在此基礎上提供部署就緒的基礎設施、Windows 相容性與模組化服務增強。

官方展示、模型與論文：
- [BreezyVoice Playground](https://huggingface.co/spaces/Splend1dchan/BreezyVoice-Playground)
- [HuggingFace 官方模型](https://huggingface.co/MediaTek-Research/BreezyVoice)
- [論文](https://arxiv.org/abs/2501.17790)
