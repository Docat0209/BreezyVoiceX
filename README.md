# BreezyVoiceX

> Based on [BreezyVoice](https://github.com/mtkresearch/BreezyVoice) by MediaTek Labs.  

BreezyVoiceX is an enhanced version of MediaTek [BreezyVoice](https://github.com/mtkresearch/BreezyVoice), focused on usability.

## Key Improvements
- Fast zero-shot voice synthesis via prompt caching
- Built-in time profiler for each major inference step
- Fully runnable without Linux-only ttsfrd dependency

## Install

> Python 3.11 is required. CUDA 12.1 recommended for GPU users.

### Clone the repo
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

## Inference

UTF8 encoding is required:

``` sh
export PYTHONUTF8=1
```

---
> This version separates the process into two explicit steps

**Run single_inference.py with the following arguments:**

### `--mode cache`（Generate speaker prompt cache）
| Argument                              | Description                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| `--speaker_prompt_audio_path`         | Required. Path to the speaker reference audio.                         |
| `--speaker_prompt_text_transcription` | Optional. Manual transcription. If not provided, Whisper will be used.             |
| `--prompt_feature_path`               | Optional. Output cache file path. Default: `cache/prompt.pt`.                      |
| `--model_path`                        | Optional. HF model ID or directory. Default: `MediaTek-Research/BreezyVoice-300M`. |


### `--mode synthesize`（Generate Audio）

| Argument | Description |
|----------|-------------|
| `--content_to_synthesize` | Required. The target text for TTS. |
| `--prompt_feature_path` | Required. Path to previously saved speaker cache (`.pt`). |
| `--output_path` | Optional. Output WAV file path. Default: `results/output.wav`. |
| `--model_path` | Optional. HF model ID or directory. Default: `MediaTek-Research/BreezyVoice-300M`. |

**Example Usage:**

### Step 1: Cache Speaker Prompt
```bash
python single_inference.py --mode cache --speaker_prompt_audio_path data/example.wav --prompt_feature_path cache/example.pt
```

### Step 2: Synthesize Voice from Text
```bash
python single_inference.py --mode synthesize --content_to_synthesize "您好，這是一段生成測試語音。" --prompt_feature_path cache/example.pt --output_path results/output.wav
```


## Credits & Acknowledgement

This project is based on [BreezyVoice](https://github.com/mtkresearch/BreezyVoice) by MediaTek Labs,  
a voice-cloning TTS system tailored for Taiwanese Mandarin with phonetic control via 注音 (bopomofo).  
The original project was derived in part from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), and is part of the [Breeze2 model family](https://huggingface.co/collections/MediaTek-Research/breeze2-family-67863158443a06a72dd29900).

We appreciate the efforts of the original authors, and this repository continues that work by providing deployment-ready infrastructure, Windows compatibility, and modular serving enhancements.

For official demo, model, and paper, please refer to:
- [BreezyVoice Playground](https://huggingface.co/spaces/Splend1dchan/BreezyVoice-Playground)
- [Official Model on HuggingFace](https://huggingface.co/MediaTek-Research/BreezyVoice)
- [Paper](https://arxiv.org/abs/2501.17790)