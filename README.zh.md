# ai_srt

[English](README.md) | 中文

本地离线字幕生成：用 [whisper.cpp](https://github.com/ggml-org/whisper.cpp)（Vulkan GPU）识别语音（默认 `large-v3-turbo`），日语等片源可选用 [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) Small（FunASR 的 llama.cpp/ggml 运行时），需要翻译时再调用本机 [Ollama](https://ollama.com/)。

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

- SenseVoice（`--asr sensevoice`）需要本地编译 FunASR 的 llama.cpp 运行时：

```bash
cd funasr-llamacpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-funasr-sensevoice
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

# 日语 / 中 / 英 / 韩 / 粤 用 SenseVoice（本地编译的 FunASR llama.cpp + Vulkan）
python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --asr sensevoice
python3 gen_srt.py /path/to/dir --from ja --to zh --asr sensevoice
python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice --sv-backend cpu

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

任意 ffmpeg 能读的视频或音频都可以。翻译模式写出 `video.<目标语言>.srt`（默认双语：译文在上、原文在下）。

## 批处理、跳过、续跑

把**目录**传给脚本，会递归处理其中及子目录里所有 `*.mp4` / `*.mkv` / `*.webm` / `*.mov` / `*.m4v` / `*.avi`（按路径排序）。某个文件失败不会跳过后面的。只要当前这一层时加 `--no-recursive`。

```bash
python3 gen_srt.py /path/to/videos --from ja --to zh --asr sensevoice
python3 gen_srt.py /path/to/videos --from ja --to zh --asr sensevoice --no-recursive
```

命令行、时间和全部输出会追加到该目录的 `gen_srt.log`（`--log PATH` 可改路径）。

字幕**完整就不会覆盖**。对 `--from ja --to zh`：

| 磁盘上已有 | 行为 |
|---|---|
| `video.zh.srt` 条数 ≥ 日文识别条数 | **跳过**（不识别、不翻译）。多余的 `video.ja.srt` 会删掉，除非 `--keep-src-srt`。 |
| `video.zh.srt` 不完整 | **从下一条接着译**。有 `video.ja.srt` 就复用，不重跑 ASR。 |

某一条对 Ollama 超时（最多 3 次、每次 60 秒）时，该行先按原文写入，后面的句子继续译。同一条命令再跑即可接着做剩余条目。
| 只有 `video.ja.srt` | 复用识别，只做翻译。 |
| 都没有 | 从头识别 + 翻译。 |

`--force` 才会全部重来。运行中 `video.ja.srt` 只是崩溃检查点，等 `video.zh.srt` 写完会删掉（除非 `--keep-src-srt`）。

## SenseVoice

默认识别引擎是 Whisper（Vulkan 上的 `large-v3-turbo`）。日语，以及中文、粤语、韩语、英语，可以改用 [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) Small（FunASR 的五语种模型）。这是独立后端，不调用 Voxtype：仓库把 FunASR 的 `runtime/llama.cpp` 放在 `funasr-llamacpp/`，用 `GGML_VULKAN=ON` 本地编译 `llama-funasr-sensevoice`，方式和 whisper.cpp 一样。

`--asr` 选识别器；`--engine` 仍是翻译器（`ollama` / `trans`）。`--model` 只作用于 Whisper，和 `--asr sensevoice` 无关。

```bash
cd funasr-llamacpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-funasr-sensevoice
```

第一次跑 `--asr sensevoice` 时，会把 GGUF 权重（`sensevoice-small-q8.gguf`、`fsmn-vad.gguf`）下到 `funasr-llamacpp/gguf/`。时间轴由二进制自带的 FSMN-VAD 切段并输出 SRT。

`--sv-backend auto`（默认）先试 Vulkan，崩溃则回退 CPU。本机用 RADV 编出来的二进制可以在 RX 9070 XT 上跑 Vulkan；FunASR 官方 Windows Vulkan 预编译包是另一套东西，这里不用。也可用 `--sv-backend vulkan` 或 `--sv-backend cpu` 强制指定。

## 许可证

本项目使用 [MIT License](LICENSE)。

`whisper.cpp/` 是上游源码树，同样为 MIT（版权归 The ggml authors，见 `whisper.cpp/LICENSE`）。其中的贡献说明只针对上游 PR，不适用于本仓库的包装脚本。

`funasr-llamacpp/` 来自 FunASR 的 `runtime/llama.cpp`（见 [FunASR](https://github.com/modelscope/FunASR)）；编译时 CMake 会把 llama.cpp 拉到 `build/`。
