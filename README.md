# ai_srt

本地离线字幕生成：用 [whisper.cpp](https://github.com/ggml-org/whisper.cpp)（Vulkan GPU）识别语音，需要翻译时再调用本机 [Ollama](https://ollama.com/)。

默认只识别、不翻译，源语言由 Whisper 自动检测。加上 `--to` 才会翻译。翻译默认完全离线（`qwen3.5`）；`--engine trans` 会走 Google，文本会上网。

顶层脚本是纯 Python 3 标准库，不用 `pip install`。

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

Whisper 权重会在首次使用对应 `--model` 时自动下载到 `whisper.cpp/models/`。

## 用法

```bash
# 只识别（输出视频同目录的 video.<源语言>.srt）
python3 gen_srt.py /path/to/video.mp4

# 指定源语言
python3 gen_srt.py /path/to/video.mp4 --from ja

# 翻译成中文（默认双语：译文 + 原文）
python3 gen_srt.py /path/to/video.mp4 --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual

# 换识别模型 / 翻译模型
python3 gen_srt.py /path/to/video.mp4 --model large-v3
python3 gen_srt.py /path/to/video.mp4 --to zh --ollama-model gemma4

# 用 translate-shell / Google（联网）
python3 gen_srt.py /path/to/video.mp4 --to zh --engine trans

python3 gen_srt.py --list-langs
python3 gen_srt.py --list-models
python3 gen_srt.py --help
```

任意 ffmpeg 能读的视频或音频都可以。翻译模式下输出是 `video.<目标语言>.srt`。
