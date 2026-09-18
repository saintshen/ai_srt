# ai_srt

English | [中文](README.zh.md)

Offline subtitle generator: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (Vulkan GPU) for speech recognition, optional local [Ollama](https://ollama.com/) for translation.

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

- Translation also needs a running Ollama daemon and the model:

```bash
ollama pull qwen3.5
```

Whisper weights download automatically to `whisper.cpp/models/` on first use of a given `--model` size.

## Usage

```bash
# transcribe only → video.<src>.srt next to the input
python3 gen_srt.py /path/to/video.mp4

# pin the source language
python3 gen_srt.py /path/to/video.mp4 --from ja

# translate to Chinese (bilingual by default: translation + original)
python3 gen_srt.py /path/to/video.mp4 --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual

# swap ASR / translation models
python3 gen_srt.py /path/to/video.mp4 --model large-v3
python3 gen_srt.py /path/to/video.mp4 --to zh --ollama-model gemma4

# translate-shell / Google (online)
python3 gen_srt.py /path/to/video.mp4 --to zh --engine trans

python3 gen_srt.py --list-langs
python3 gen_srt.py --list-models
python3 gen_srt.py --help
```

Any ffmpeg-readable video or audio works. In translate mode the output is `video.<tgt>.srt`.

`whisper.cpp/` is a vendored upstream tree (MIT). It has its own `LICENSE` and contribution docs; those apply to upstream PRs, not this wrapper.
