# ai_srt

A personal offline subtitle generator: local `whisper.cpp` (Vulkan backend)
does speech recognition (default `large-v3-turbo`), optional SenseVoice
(FunASR, via Voxtype ONNX) for Japanese / zh/en/ja/ko/yue, and local Ollama
translates when asked. `whisper.cpp/`
is a vendored upstream clone (has its own build system and its own
`AGENTS.md` contribution policy — that policy is for upstream PRs, not
relevant here since we only consume its prebuilt binary).

Default is transcribe-only (SRT in the spoken language). Translation is
opt-in via `--to`. Source language defaults to Whisper auto-detect, override
with `--from`. The original Japanese→Chinese path is `--from ja --to zh`.

## Dev environment

- Pure-stdlib Python 3 (`argparse, json, os, re, subprocess, sys, tempfile,
  urllib.request, pathlib`) — no venv, no `requirements.txt`, no `pip
  install` needed for the top-level script.
- System deps: `ffmpeg` (already on PATH at `/usr/bin/ffmpeg`), a running
  Ollama daemon (`http://localhost:11434`) if translating.
- GPU: AMD Radeon RX 9070 XT (gfx1201), Vulkan only — no ROCm/HIP/CUDA
  involved in this script's path.

## Build & run

`whisper.cpp/build/bin/whisper-cli` is already compiled (CMake, Release,
`GGML_VULKAN=ON`) — do not rebuild unless the binary is missing or you need
to bump whisper.cpp itself. If a rebuild is needed:
```
cd whisper.cpp && cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Translation uses local Ollama `qwen3.5` by default (override with
`--ollama-model`). Ensure the model is pulled (`ollama pull qwen3.5`) if
it is missing.

Run the pipeline:
```
python3 gen_srt.py /path/to/video.mp4
python3 gen_srt.py /path/to/video.mp4 --from ja
python3 gen_srt.py /path/to/video.mp4 --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual
python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice
python3 gen_srt.py /path/to/video.mp4 --model small
python3 gen_srt.py /path/to/video.mp4 --to zh --ollama-model gemma4
python3 gen_srt.py /path/to/video.mp4 --engine trans --to zh   # translate-shell/Google (sends text online)
python3 gen_srt.py --list-langs
python3 gen_srt.py --list-models
python3 gen_srt.py --help
```

Output is written next to the input: `video.<src>.srt` (transcribe) or
`video.<tgt>.srt` (translate). Whisper models auto-download to
`whisper.cpp/models/ggml-<size>.bin` on first use of a given `--model` size.
If `~/.local/share/voxtype/models/ggml-large-v3-turbo.bin` already exists,
`--model large-v3-turbo` symlinks that file instead of downloading again.

`--asr sensevoice` is for Japanese (also zh/en/ko/yue). It does **not** use
the Vulkan `voxtype` binary (that build has no SenseVoice). It calls
`/usr/lib/voxtype/voxtype-onnx-avx512` (fallback: `voxtype-onnx-avx2`) on
Silero VAD clips so the SRT still has timestamps. VAD for this path must
run on CPU: `whisper-vad-speech-segments --use-gpu` aborts on this AMD
Vulkan setup.

There is no test suite for the top-level script. `whisper.cpp/tests/`
(`tests/run-tests.sh`) covers the vendored engine, not this repo's code.

## Conventions

- All user-facing strings, docstrings, comments, and `--help` text are in
  English. Keep a Chinese README at `README.zh.md` in sync with `README.md`.
  Language aliases in `LANG_ALIASES` may still accept CJK input.
- Progress is printed as numbered stages (`[1/3] ...`, `[2/3] ...`, `[3/3]
  ...`); follow the same style for new pipeline steps. Transcribe-only runs
  are `[1/2]` + `[2/2]`.
- `whisper-cli` is invoked with `LD_LIBRARY_PATH` pointed at
  `whisper.cpp/build/bin` (for `libggml*`, `libwhisper*`) — required any
  time you shell out to that binary directly.
- Ollama calls go through the raw `/api/generate` HTTP endpoint (`urllib`),
  not the `ollama` Python package — keep new Ollama calls consistent with
  that (`stream: false`, explicit `keep_alive`).
- Default Ollama model: `qwen3.5` for every language pair. Override with
  `--ollama-model`.
- Default Whisper model: `large-v3-turbo`. ASR backend is `--asr whisper`
  (default) or `--asr sensevoice`. `--engine` remains the *translation*
  engine (`ollama` / `trans`).

## Pitfalls

- `whisper.cpp/build/` is a generated CMake build tree — never hand-edit
  files under it; regenerate via the cmake command above.
- `whisper-cli` subprocess output is captured as raw bytes and decoded with
  `errors="replace"` because its stderr/progress output can contain
  non-UTF-8 bytes — don't switch that call to `text=True`, it will crash
  intermittently.
- Auto-detected source language is parsed from whisper.cpp's
  `auto-detected language: xx (p = ...)` log line on stderr. If that format
  changes upstream, `--from` still works; filename/`--to` source may fall
  back to `unk`.
- VAD model (`whisper.cpp/models/ggml-silero-v6.2.0.bin`) must exist for
  `--vad` (on by default); if missing, the script prints a warning and
  silently processes full audio instead of failing.
- `--engine trans` sends subtitle text to Google Translate over the network;
  `--engine ollama` (default) is fully offline — don't switch the default
  without calling that out, per user's stated offline/privacy preference.
- Ollama `/api/generate` calls always send `think: false`. Reasoning models
  (qwen3.5, deepseek-r1, gpt-oss) otherwise spend the token budget on a
  hidden chain-of-thought and often return an empty translation. If an old
  Ollama rejects the field with HTTP 400, the call retries without it.
- Do not call `whisper-vad-speech-segments --use-gpu` on this AMD Vulkan
  setup; it aborts. The SenseVoice path always runs that helper on CPU.
- `/usr/bin/voxtype` is the Vulkan build and cannot run SenseVoice. The
  SenseVoice path must use `voxtype-onnx-avx512` (or `voxtype-onnx-avx2`).
