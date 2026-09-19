# ai_srt

[English](README.md) | 中文

本地离线字幕生成：用 [whisper.cpp](https://github.com/ggml-org/whisper.cpp)（Vulkan GPU）识别语音（默认 `large-v3-turbo`），日语等片源可选用 [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) Small（FunASR 的 llama.cpp/ggml 运行时），需要翻译时再调用本机 [Ollama](https://ollama.com/)。

默认只识别、不翻译，源语言由 Whisper 自动检测。加上 `--to` 才会翻译。翻译默认完全离线（`qwen3.5`）；`--engine trans` 会走 Google，文本会上网。

顶层脚本是纯 Python 3 标准库，不用 `pip install`。SenseVoice 首次使用会下载 FunASR 的 `llama-funasr-sensevoice` 二进制和 GGUF 权重。

## 依赖

- Python 3、`ffmpeg`
- 本仓库自带 `whisper.cpp` 源码。克隆后若还没有 `whisper.cpp/build/bin/whisper-cli`，需要编译一次：

```bash
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- 翻译还需要本机 Ollama 已启动，并拉好模型：

```bash
ollama pull qwen3.5
```

Whisper 权重会在首次使用对应 `--model` 时自动下载到 `whisper.cpp/models/`。若 Voxtype 已有 `ggml-large-v3-turbo.bin`，脚本会直接链接过去，不再下第二份。

## 用法

```bash
# 只识别（输出视频同目录的 video.<源语言>.srt）
python3 gen_srt.py /path/to/video.mp4

# 指定源语言
python3 gen_srt.py /path/to/video.mp4 --from ja

# 日语用 SenseVoice（FunASR llama.cpp/ggml；Vulkan 可用则用，否则 CPU）
python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice

# 翻译成中文（默认双语：译文 + 原文）
python3 gen_srt.py /path/to/video.mp4 --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual

# 换识别模型 / 翻译模型
python3 gen_srt.py /path/to/video.mp4 --model small
python3 gen_srt.py /path/to/video.mp4 --to zh --ollama-model gemma4

# 用 translate-shell / Google（联网）
python3 gen_srt.py /path/to/video.mp4 --to zh --engine trans

python3 gen_srt.py --list-langs
python3 gen_srt.py --list-models
python3 gen_srt.py --help
```

任意 ffmpeg 能读的视频或音频都可以。翻译模式下输出是 `video.<目标语言>.srt`。

## 许可证

本项目使用 [MIT License](LICENSE)。

`whisper.cpp/` 是上游源码树，同样为 MIT（版权归 The ggml authors，见 `whisper.cpp/LICENSE`）。其中的贡献说明只针对上游 PR，不适用于本仓库的包装脚本。
