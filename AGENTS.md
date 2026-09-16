# ai_srt

A personal offline subtitle generator: local `whisper.cpp` (Vulkan backend)
does speech recognition, local Ollama translates when asked. `whisper.cpp/`
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

Ensure the `trans-ja` Ollama model exists (one-time, from `Modelfile` in repo
root) if you will translate Japanese→Chinese:
```
ollama create trans-ja -f Modelfile
```
Other language pairs use `qwen2.5:7b` directly (the same base as `trans-ja`)
with a generic translation prompt. Override with `--ollama-model`.

Run the pipeline:
```
python3 gen_srt.py /path/to/video.mp4
python3 gen_srt.py /path/to/video.mp4 --from ja
python3 gen_srt.py /path/to/video.mp4 --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh
python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual
python3 gen_srt.py /path/to/video.mp4 --model large-v3
python3 gen_srt.py /path/to/video.mp4 --engine trans --to zh   # translate-shell/Google (sends text online)
python3 gen_srt.py --list-langs
python3 gen_srt.py --help
```

Output is written next to the input: `video.<src>.srt` (transcribe) or
`video.<tgt>.srt` (translate). Whisper models auto-download to
`whisper.cpp/models/ggml-<size>.bin` on first use of a given `--model` size.

There is no test suite for the top-level script. `whisper.cpp/tests/`
(`tests/run-tests.sh`) covers the vendored engine, not this repo's code.

## Conventions

- All user-facing strings, docstrings, and `--help` text are in Chinese
  (中文) — match this when editing or adding CLI options.
- Progress is printed as numbered stages (`[1/3] ...`, `[2/3] ...`, `[3/3]
  ...`); follow the same style for new pipeline steps. Transcribe-only runs
  are `[1/2]` + `[2/2]`.
- `whisper-cli` is invoked with `LD_LIBRARY_PATH` pointed at
  `whisper.cpp/build/bin` (for `libggml*`, `libwhisper*`) — required any
  time you shell out to that binary directly.
- Ollama calls go through the raw `/api/generate` HTTP endpoint (`urllib`),
  not the `ollama` Python package — keep new Ollama calls consistent with
  that (`stream: false`, explicit `keep_alive`).
- Default Ollama model: `trans-ja` for ja→zh, `qwen2.5:7b` for every other
  pair. Do not send non-ja→zh text through `trans-ja` (its Modelfile system
  prompt is Japanese→Chinese only).

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
