# ai_srt

English | [中文](README.zh.md)

Offline subtitle generator: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (Vulkan GPU) for speech recognition (default `large-v3-turbo`), optional [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) Small via FunASR's llama.cpp/ggml runtime for Japanese (and zh/en/ko/yue), and optional local [Ollama](https://ollama.com/) for translation.

Default is transcribe-only. Whisper auto-detects the source language. Pass `--to` to translate. Translation is offline by default (`qwen3.5`); `--engine trans` uses Google and sends text over the network.

The top-level script is Python 3 stdlib only — no `pip install`.

## Requirements

- Python 3 and `ffmpeg`
- This repo vendors `whisper.cpp` source. After cloning, if `whisper.cpp/build/bin/whisper-cli` is missing, build it once:

```bash
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- SenseVoice (`--asr sensevoice`) needs a local FunASR llama.cpp build:

```bash
cd funasr-llamacpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-funasr-sensevoice
```

- Translation also needs a running Ollama daemon and the model:

```bash
ollama pull qwen3.5
```

Whisper weights download automatically to `whisper.cpp/models/` on first use of a given `--model` size. If Voxtype already has `ggml-large-v3-turbo.bin`, the script links that copy instead of downloading again.

## Usage

```bash
# transcribe only → video.<src>.srt next to the input
python3 gen_srt.py /path/to/video.mp4

# pin the source language
python3 gen_srt.py /path/to/video.mp4 --from ja

# Japanese / zh / en / ko / yue via SenseVoice (local FunASR llama.cpp + Vulkan)
python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --asr sensevoice
python3 gen_srt.py /path/to/dir --from ja --to zh --asr sensevoice
python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice --sv-backend cpu

# translate to Chinese (bilingual by default: translation + original)
python3 gen_srt.py /path/to/video.mp4 --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual

# swap ASR / translation models
python3 gen_srt.py /path/to/video.mp4 --model small
python3 gen_srt.py /path/to/video.mp4 --to zh --ollama-model gemma4

# translate-shell / Google (online)
python3 gen_srt.py /path/to/video.mp4 --to zh --engine trans

python3 gen_srt.py --list-langs
python3 gen_srt.py --list-models
python3 gen_srt.py --help
```

Any ffmpeg-readable video or audio works. Translate mode writes `video.<tgt>.srt` (bilingual by default: translation, then original).

## Batch, skip, resume

Pass a **directory** to process every `*.mp4` / `*.mkv` / `*.webm` / `*.mov` / `*.m4v` / `*.avi` in it **and subdirectories** (sorted by path). One file failing does not skip the rest.

```bash
python3 gen_srt.py /path/to/videos --from ja --to zh --asr sensevoice
python3 gen_srt.py /path/to/videos --from ja --to zh --asr sensevoice --no-recursive
```

Command line, timestamps, and all output are appended to `gen_srt.log` in that directory (`--log PATH` to override).

Existing subtitles are **not overwritten** if they look complete. For `--from ja --to zh`:

| Already on disk | What happens |
|---|---|
| `video.zh.srt` has at least as many cues as the Japanese transcript | **Skip** (no ASR, no translation). Leftover `video.ja.srt` is deleted unless `--keep-src-srt`. |
| `video.zh.srt` is partial | **Resume translation** from the next cue. Reuses `video.ja.srt` if present. |

If a single cue times out against Ollama (three tries, 60s each), that line is written untranslated and the rest of the file continues. Rerun the same command to pick up remaining cues.
| Only `video.ja.srt` | Reuse ASR, translate only. |
| Neither | Full ASR + translation. |

`--force` redos ASR and translation from scratch. During a run, `video.ja.srt` is a crash checkpoint and is removed when `video.zh.srt` is finished (unless `--keep-src-srt`).

## SenseVoice

Default ASR is Whisper (`large-v3-turbo` on Vulkan). For Japanese — and Chinese, Cantonese, Korean, or English — you can switch to [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) Small, FunASR’s five-language model. It is a separate engine, not Voxtype: this repo vendors FunASR’s `runtime/llama.cpp` as `funasr-llamacpp/` and compiles `llama-funasr-sensevoice` with `GGML_VULKAN=ON`, same idea as whisper.cpp.

`--asr` picks the recognizer; `--engine` still picks the translator (`ollama` / `trans`). `--model` is Whisper-only and is ignored with `--asr sensevoice`.

```bash
cd funasr-llamacpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-funasr-sensevoice
```

On first `--asr sensevoice` run, GGUF weights (`sensevoice-small-q8.gguf`, `fsmn-vad.gguf`) download into `funasr-llamacpp/gguf/`. The binary emits SRT with its own FSMN-VAD.

`--sv-backend auto` (default) tries Vulkan, then CPU if Vulkan crashes. On this machine a local RADV build runs on the RX 9070 XT; FunASR’s official Windows Vulkan zip is a different binary and is not used. Force a backend with `--sv-backend vulkan` or `--sv-backend cpu`.

## License

This project is licensed under the [MIT License](LICENSE).

`whisper.cpp/` is a vendored upstream tree, also MIT (copyright The ggml authors; see `whisper.cpp/LICENSE`). Its contribution docs apply to upstream PRs, not this wrapper.

`funasr-llamacpp/` is FunASR's `runtime/llama.cpp` (see [FunASR](https://github.com/modelscope/FunASR)); CMake fetches llama.cpp into `build/` at compile time.
