# ai_srt

English | [中文](README.zh.md)

Offline subtitle generator: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (Vulkan GPU) for speech recognition (default `large-v3-turbo`), optional [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) Small via FunASR's llama.cpp/ggml runtime for Japanese (and zh/en/ko/yue), and optional local [Ollama](https://ollama.com/) for translation.

Default is transcribe-only. Whisper auto-detects the source language. Pass `--to` to translate. Translation is offline by default (`qwen3.5`); `--engine trans` uses Google and sends text over the network.

The top-level script is Python 3 stdlib only — no `pip install`. SenseVoice downloads FunASR's `llama-funasr-sensevoice` binary and GGUF weights on first use.

## Requirements

- Python 3 and `ffmpeg`
- This repo vendors `whisper.cpp` source. After cloning, if `whisper.cpp/build/bin/whisper-cli` is missing, build it once:

```bash
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
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

# Japanese via SenseVoice (FunASR llama.cpp/ggml; Vulkan if it works, else CPU)
python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice

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

Any ffmpeg-readable video or audio works. In translate mode the output is `video.<tgt>.srt`.

## License

This project is licensed under the [MIT License](LICENSE).

`whisper.cpp/` is a vendored upstream tree, also MIT (copyright The ggml authors; see `whisper.cpp/LICENSE`). Its contribution docs apply to upstream PRs, not this wrapper.
