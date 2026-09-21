#!/usr/bin/env python3
"""
Generate subtitles from video or audio: local whisper.cpp (Vulkan GPU) for
speech recognition (default model large-v3-turbo), optional SenseVoice
Small via FunASR's llama.cpp/ggml runtime (Vulkan when it works, else CPU)
for Japanese / zh/en/ja/ko/yue, and optional local Ollama or translate-shell
for translation.

Any ffmpeg-readable file works. Default is transcribe-only (no translation);
Whisper auto-detects the source language, override with --from. Pass --to to
translate. For Japanese video, --asr sensevoice uses SenseVoice Small
(FunASR llama.cpp, not Voxtype).

Translation engines (--engine):
    ollama (default): fully offline, qwen3.5 with a generic translation prompt.
    trans: translate-shell (trans) via the Google Translate API
        (sends text to Google; not offline).

Usage:
    python3 gen_srt.py /path/to/video.mp4
    python3 gen_srt.py /path/to/video.mp4 --from ja
    python3 gen_srt.py /path/to/video.mp4 --to zh
    python3 gen_srt.py /path/to/video.mp4 --from ja --to zh
    python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual
    python3 gen_srt.py /path/to/video.mp4 --from ja --asr sensevoice
    python3 gen_srt.py /path/to/dir --from ja --to zh --asr sensevoice
    python3 gen_srt.py /path/to/video.mp4 --model small
    python3 gen_srt.py /path/to/video.mp4 --to zh --ollama-model gemma4
    python3 gen_srt.py /path/to/video.mp4 --engine trans --to zh
    python3 gen_srt.py --list-langs
    python3 gen_srt.py --list-models
    python3 gen_srt.py --help

Output:
    transcribe-only: /path/to/video.<src>.srt
    translate:       /path/to/video.<tgt>.srt
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER_CPP_DIR = ROOT / "whisper.cpp"
WHISPER_CLI = WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"
MODELS_DIR = WHISPER_CPP_DIR / "models"
VAD_MODEL = MODELS_DIR / "ggml-silero-v6.2.0.bin"
VOXTYPE_TURBO = Path.home() / ".local/share/voxtype/models/ggml-large-v3-turbo.bin"

FUNASR_DIR = ROOT / "funasr-llamacpp"
FUNASR_BIN = FUNASR_DIR / "build" / "bin" / "llama-funasr-sensevoice"
FUNASR_GGUF_DIR = FUNASR_DIR / "gguf"
FUNASR_MODEL = FUNASR_GGUF_DIR / "sensevoice-small-q8.gguf"
FUNASR_VAD = FUNASR_GGUF_DIR / "fsmn-vad.gguf"
FUNASR_VULKAN_MARKER = FUNASR_DIR / ".vulkan-broken"
FUNASR_MODEL_URL = (
    "https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF/resolve/main/"
    "sensevoice-small-q8.gguf"
)
FUNASR_VAD_URL = (
    "https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF/resolve/main/fsmn-vad.gguf"
)

DEFAULT_WHISPER_MODEL = "large-v3-turbo"
DEFAULT_OLLAMA_MODEL = "qwen3.5"
DEFAULT_VAD_MAX_SPEECH_S = 8.0
OLLAMA_RETRIES = 3
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
SENSEVOICE_LANGS = {"zh", "en", "ja", "ko", "yue"}

# whisper.cpp language codes → English names (aligned with src/whisper.cpp g_lang)
WHISPER_LANGS = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

# Names used in the translation prompt; unlisted codes fall back to WHISPER_LANGS
LANG_PROMPT_NAMES = {
    "zh": "Simplified Chinese", "zh-CN": "Simplified Chinese",
    "yue": "Cantonese",
}

# User-facing aliases → whisper code (CJK aliases kept as input convenience)
LANG_ALIASES = {
    "auto": "auto", "detect": "auto",
    "zh-cn": "zh", "zh-hans": "zh", "zh-tw": "zh", "zh-hant": "zh",
    "cmn": "zh", "chinese": "zh", "中文": "zh", "汉语": "zh",
    "简体": "zh", "简体中文": "zh", "繁体": "zh", "繁体中文": "zh",
    "japanese": "ja", "日语": "ja", "日文": "ja", "日本語": "ja",
    "english": "en", "英语": "en", "英文": "en", "英語": "en",
    "korean": "ko", "韩语": "ko", "韩文": "ko", "韓語": "ko",
    "朝鲜语": "ko", "한국어": "ko",
    "french": "fr", "法语": "fr", "法文": "fr",
    "german": "de", "德语": "de", "德文": "de",
    "spanish": "es", "西班牙语": "es", "西班牙文": "es",
    "russian": "ru", "俄语": "ru", "俄文": "ru",
    "vietnamese": "vi", "越南语": "vi", "越南文": "vi",
    "thai": "th", "泰语": "th", "泰文": "th",
    "arabic": "ar", "阿拉伯语": "ar", "阿拉伯文": "ar",
    "cantonese": "yue", "粤语": "yue", "广东话": "yue", "廣東話": "yue",
    "portuguese": "pt", "葡萄牙语": "pt", "pt-br": "pt", "pt-pt": "pt",
}

# translate-shell wants zh-CN for Chinese; other codes match whisper
TRANS_CODES = {"zh": "zh-CN"}

AUTO_LANG_RE = re.compile(r"auto-detected language:\s+(\S+)\s+\(p\s*=")
SRT_BLOCK_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s*(.*)"
)
MUSIC_PLACEHOLDER_RE = re.compile(r"^[\(（][^)）]*[\)）]$|^[♪\s]+$")
SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]*\|>")
SENSEVOICE_LANG_TAG_RE = re.compile(r"<\|(zh|en|ja|ko|yue)\|>")
SRT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def lang_display(code: str) -> str:
    if code in LANG_PROMPT_NAMES:
        return LANG_PROMPT_NAMES[code]
    if code in WHISPER_LANGS:
        return WHISPER_LANGS[code]
    return code


def normalize_lang(value: str, allow_auto: bool = False) -> str:
    raw = value.strip()
    key = raw.lower().replace("_", "-")
    if key in ("auto", "detect", ""):
        if allow_auto:
            return "auto"
        raise argparse.ArgumentTypeError("target language cannot be auto")
    resolved = LANG_ALIASES.get(raw) or LANG_ALIASES.get(key)
    if resolved == "auto":
        if allow_auto:
            return "auto"
        raise argparse.ArgumentTypeError("target language cannot be auto")
    if resolved:
        return resolved
    if key in WHISPER_LANGS:
        return key
    for code, name in WHISPER_LANGS.items():
        if name == key:
            return code
    hint = " (auto is also allowed)" if allow_auto else ""
    raise argparse.ArgumentTypeError(
        f"unknown language: {value}{hint}. See --list-langs"
    )


def trans_code(code: str) -> str:
    return TRANS_CODES.get(code, code)


def ollama_base_url(ollama_url: str) -> str:
    url = ollama_url.rstrip("/")
    if url.endswith("/api/generate"):
        url = url[: -len("/api/generate")]
    return url


def list_ollama_models(ollama_url: str = "http://localhost:11434/api/generate"):
    """Return [(name, details, size), ...]; exit if Ollama is unreachable."""
    tags_url = ollama_base_url(ollama_url) + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=10) as resp:
            data = json.load(resp)
    except Exception as e:
        sys.exit(f"cannot reach Ollama ({tags_url}): {e}")
    models = []
    for m in data.get("models", []):
        name = m.get("name") or ""
        if name:
            models.append((name, m.get("details") or {}, m.get("size") or 0))
    models.sort(key=lambda x: x[0])
    return models


def print_ollama_models():
    models = list_ollama_models()
    if not models:
        print("No local Ollama models installed. Download one with: ollama pull <name>")
        return
    print(f"{'model':<24} {'params':<10} size")
    for name, details, size in models:
        params = details.get("parameter_size") or ""
        size_gb = f"{size / 1e9:.1f} GB" if size else ""
        print(f"{name:<24} {params:<10} {size_gb}")
    print()
    print("usage: python3 gen_srt.py video.mp4 --to zh --ollama-model <name>")
    print(f"default: {DEFAULT_OLLAMA_MODEL}")


def ollama_model_installed(name: str, installed: list) -> bool:
    wanted = {name, f"{name}:latest"}
    if name.endswith(":latest"):
        wanted.add(name[:-len(":latest")])
    have = set()
    for inst in installed:
        have.add(inst)
        if inst.endswith(":latest"):
            have.add(inst[:-len(":latest")])
    return bool(wanted & have)


def ensure_ollama_model(name: str):
    installed = [m[0] for m in list_ollama_models()]
    if ollama_model_installed(name, installed):
        return
    listing = "\n".join(f"    {n}" for n in installed) or "    (none)"
    sys.exit(
        f"Ollama model not found: {name}\ninstalled:\n{listing}\n"
        f"list: python3 gen_srt.py --list-models\n"
        f"pull: ollama pull {name}"
    )


def ollama_generate(ollama_model: str, prompt: str, keep_alive: str = "10m",
                    ollama_url: str = "http://localhost:11434/api/generate",
                    options=None, timeout: int = 180,
                    retries: int = OLLAMA_RETRIES) -> dict:
    """POST /api/generate. Always send think=false so reasoning models emit
    the translation instead of burning the token budget on a hidden CoT."""
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "think": False,
    }
    if options:
        payload["options"] = options

    def _post(body: dict):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            ollama_url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    delay = 2
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return _post(payload)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            # Older Ollama / models that reject `think` return HTTP 400; retry once.
            if e.code == 400 and "think" in payload:
                payload.pop("think", None)
                try:
                    return _post(payload)
                except urllib.error.HTTPError as e2:
                    err_body = e2.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Ollama request failed HTTP {e2.code}: {err_body[:500]}"
                    ) from e2
            raise RuntimeError(
                f"Ollama request failed HTTP {e.code}: {err_body[:500]}"
            ) from e
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
            if attempt == retries:
                raise RuntimeError(
                    f"Ollama failed after {retries} tries ({type(e).__name__}: {e})"
                ) from e
            print(f"    Ollama {type(e).__name__}: {e}; "
                  f"retry {attempt}/{retries} in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"Ollama failed: {last_err}") from last_err


def whisper_env() -> dict:
    env = os.environ.copy()
    lib_dir = str(WHISPER_CPP_DIR / "build" / "bin")
    env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def extract_audio(media_path: Path, wav_path: Path, stage: str):
    cmd = [
        "ffmpeg", "-y", "-i", str(media_path),
        "-vn", "-ar", "16000", "-ac", "1",
        str(wav_path),
    ]
    print(f"{stage} extracting audio...")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        sys.exit(f"ffmpeg failed:\n{err[-2000:]}")


def ensure_model(model_size: str) -> Path:
    model_path = MODELS_DIR / f"ggml-{model_size}.bin"
    if model_path.exists():
        return model_path
    if model_size == "large-v3-turbo" and VOXTYPE_TURBO.is_file():
        try:
            model_path.symlink_to(VOXTYPE_TURBO)
            print(f"    linked Whisper {model_size} from {VOXTYPE_TURBO}")
            return model_path
        except OSError as e:
            print(f"    could not link Voxtype {model_size} ({e}); downloading...")
    print(f"    downloading Whisper model {model_size} ...")
    subprocess.run(
        ["bash", str(WHISPER_CPP_DIR / "models" / "download-ggml-model.sh"), model_size],
        check=True, cwd=WHISPER_CPP_DIR,
    )
    return model_path


def ts_to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def transcribe(wav_path: Path, model_path: Path, language: str,
               use_vad: bool = True, suppress_music: bool = True,
               vad_threshold: float = 0.5, vad_min_silence_ms: int = 100,
               vad_max_speech_s: float = DEFAULT_VAD_MAX_SPEECH_S,
               stage: str = "[2/3]"):
    if language == "auto":
        print(f"{stage} transcribing with whisper-cli (Vulkan GPU), auto-detecting language...")
    else:
        print(f"{stage} transcribing {lang_display(language)} with whisper-cli (Vulkan GPU)...")
    cmd = [
        str(WHISPER_CLI),
        "-m", str(model_path),
        "-f", str(wav_path),
        "-l", language,
        "--output-txt",
    ]
    if use_vad and VAD_MODEL.exists():
        cmd += [
            "--vad", "-vm", str(VAD_MODEL),
            "-vt", str(vad_threshold),
            "-vsd", str(vad_min_silence_ms),
            "-vmsd", str(vad_max_speech_s),
        ]
        print(f"    VAD enabled, skipping non-speech "
              f"(threshold={vad_threshold}, min silence={vad_min_silence_ms}ms, "
              f"max speech={vad_max_speech_s}s)")
    elif use_vad:
        print(f"    warning: VAD model missing ({VAD_MODEL}), processing full audio")
    if suppress_music:
        # -sns drops the model's non-speech special tokens
        # --suppress-regex also filters placeholders like (音楽)/(拍手)/(music)
        cmd += [
            "-sns",
            "--suppress-regex", r"[\(（][^)）]*[\)）]|♪+",
        ]
        print("    non-speech placeholder suppression enabled (e.g. (音楽)/(拍手)/(music))")
    env = whisper_env()
    # Capture raw bytes and decode loosely: whisper-cli stderr/progress can
    # contain non-UTF-8 bytes (progress bars, terminal control chars) that
    # would crash with text=True.
    result = subprocess.run(cmd, capture_output=True, text=False, env=env)
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        sys.exit(f"whisper-cli failed:\n{stderr_text[-2000:]}")

    detected = language
    if language == "auto":
        matches = AUTO_LANG_RE.findall(stderr_text)
        if matches:
            detected = matches[-1]
            print(f"    detected source language: {detected} ({lang_display(detected)})")
        else:
            detected = "unk"
            print("    warning: could not parse source language from whisper log; treating as unknown")

    segments = []
    skipped = 0
    for line in stdout_text.splitlines():
        m = SRT_BLOCK_RE.search(line)
        if m:
            h1, m1, s1, ms1, h2, m2, s2, ms2, text = m.groups()
            start = ts_to_seconds(h1, m1, s1, ms1)
            end = ts_to_seconds(h2, m2, s2, ms2)
            text = text.strip()
            if not text:
                continue
            if suppress_music and MUSIC_PLACEHOLDER_RE.match(text):
                skipped += 1
                continue
            segments.append((start, end, text))
    extra = f" (filtered {skipped} non-speech placeholders)" if skipped else ""
    print(f"    {len(segments)} segments{extra}")
    return segments, detected


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"    downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ai_srt"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def ensure_funasr_runtime():
    """Require a locally built SenseVoice binary; download GGUF weights if missing."""
    if not FUNASR_BIN.is_file():
        sys.exit(
            f"llama-funasr-sensevoice missing: {FUNASR_BIN}\n"
            "Build it like whisper.cpp (Vulkan):\n"
            "  cd funasr-llamacpp && cmake -B build -DGGML_VULKAN=ON "
            "-DCMAKE_BUILD_TYPE=Release && cmake --build build -j "
            "--target llama-funasr-sensevoice"
        )
    if not FUNASR_MODEL.is_file():
        download_file(FUNASR_MODEL_URL, FUNASR_MODEL)
    if not FUNASR_VAD.is_file():
        download_file(FUNASR_VAD_URL, FUNASR_VAD)


def strip_sensevoice_text(text: str) -> str:
    return SENSEVOICE_TAG_RE.sub("", text).strip()


def parse_srt_blocks(text: str):
    """Parse SRT into (start, end, body_lines). Incomplete trailing blocks are dropped."""
    blocks = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        ts_line = lines[0]
        body_from = 1
        if not SRT_TS_RE.search(ts_line) and len(lines) >= 2:
            ts_line = lines[1]
            body_from = 2
        m = SRT_TS_RE.search(ts_line)
        if not m:
            continue
        start = ts_to_seconds(*m.groups()[:4])
        end = ts_to_seconds(*m.groups()[4:])
        body = lines[body_from:]
        if body:
            blocks.append((start, end, body))
    return blocks


def parse_srt_cues(text: str):
    """Parse standard SRT from llama-funasr-sensevoice --srt stdout."""
    cues = []
    for start, end, body in parse_srt_blocks(text):
        joined = strip_sensevoice_text(" ".join(body))
        if joined:
            cues.append((start, end, joined))
    return cues


def load_srt_cues(path: Path):
    return parse_srt_cues(path.read_text(encoding="utf-8"))


def load_srt_blocks(path: Path):
    return parse_srt_blocks(path.read_text(encoding="utf-8"))


def media_srt_path(media_path: Path, tag: str) -> Path:
    return Path(str(media_path.with_suffix("")) + f".{tag}.srt")


def write_srt_cue(fh, idx: int, start: float, end: float, body_lines):
    fh.write(f"{idx}\n")
    fh.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
    for line in body_lines:
        fh.write(f"{line}\n")
    fh.write("\n")
    fh.flush()


def write_srt_file(path: Path, cues):
    """cues: iterable of (start, end, body_lines_or_str)."""
    with open(path, "w", encoding="utf-8") as fh:
        for idx, item in enumerate(cues, 1):
            start, end, body = item
            if isinstance(body, str):
                body = [body]
            write_srt_cue(fh, idx, start, end, body)


def pick_sensevoice_backend(requested: str) -> str:
    if requested in ("cpu", "vulkan"):
        return requested
    if FUNASR_VULKAN_MARKER.exists():
        return "cpu"
    return "vulkan"


def run_funasr_sensevoice(wav_path: Path, backend: str, use_vad: bool,
                          vad_max_speech_s: float):
    cmd = [
        str(FUNASR_BIN),
        "-m", str(FUNASR_MODEL),
        "-a", str(wav_path),
        "--backend", backend,
        "--srt",
    ]
    if use_vad:
        cmd += [
            "--vad", str(FUNASR_VAD),
            "--vad-maxseg", str(int(vad_max_speech_s * 1000)),
        ]
    result = subprocess.run(cmd, capture_output=True, text=False, timeout=7200)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    return result.returncode, stdout, stderr


def transcribe_sensevoice(wav_path: Path, language: str,
                          use_vad: bool = True, suppress_music: bool = True,
                          vad_max_speech_s: float = DEFAULT_VAD_MAX_SPEECH_S,
                          backend: str = "auto",
                          stage: str = "[2/3]"):
    """SenseVoice Small via FunASR llama.cpp/ggml. Tries Vulkan, falls back to CPU."""
    ensure_funasr_runtime()
    if language not in ("auto", "unk") and language not in SENSEVOICE_LANGS:
        print(f"    warning: SenseVoice covers zh/en/ja/ko/yue; '{language}' "
              f"will still run (no language flag on this binary)")

    chosen = pick_sensevoice_backend(backend)
    print(f"{stage} transcribing with SenseVoice Small "
          f"(FunASR llama.cpp/ggml, backend={chosen})...")
    print(f"    binary: {FUNASR_BIN}")
    print(f"    model:  {FUNASR_MODEL.name}")

    code, stdout, stderr = run_funasr_sensevoice(
        wav_path, chosen, use_vad, vad_max_speech_s,
    )
    if code != 0 and chosen == "vulkan" and backend == "auto":
        print("    Vulkan SenseVoice failed "
              f"(exit {code}); falling back to CPU. "
              "This is expected on some AMD GPUs (e.g. RX 9070 XT).")
        FUNASR_VULKAN_MARKER.write_text("vulkan backend failed\n", encoding="utf-8")
        chosen = "cpu"
        print("    retrying with backend=cpu ...")
        code, stdout, stderr = run_funasr_sensevoice(
            wav_path, chosen, use_vad, vad_max_speech_s,
        )
    if code != 0:
        sys.exit(
            f"llama-funasr-sensevoice failed (backend={chosen}, exit {code}):\n"
            f"{stderr[-2000:]}"
        )

    segments = parse_srt_cues(stdout)
    skipped = 0
    if suppress_music:
        kept = []
        for start, end, text in segments:
            if MUSIC_PLACEHOLDER_RE.match(text):
                skipped += 1
                continue
            kept.append((start, end, text))
        segments = kept

    detected = language if language != "auto" else "unk"
    if language == "auto":
        m = SENSEVOICE_LANG_TAG_RE.search(stdout + "\n" + stderr)
        if m:
            detected = m.group(1)
            print(f"    detected source language: {detected} ({lang_display(detected)})")

    extra = f" (filtered {skipped} non-speech placeholders)" if skipped else ""
    print(f"    {len(segments)} segments{extra}")
    return segments, detected


def warmup_ollama(ollama_model: str, keep_alive: str = "10m",
                  ollama_url: str = "http://localhost:11434/api/generate"):
    """Load the model into VRAM and pin keep_alive so it is not unloaded mid-run."""
    print(f"    warming up Ollama model {ollama_model} (keep_alive={keep_alive}) ...")
    ollama_generate(ollama_model, "hello", keep_alive=keep_alive, ollama_url=ollama_url)
    print("    warmup done, model is resident in VRAM")


def translate_ollama(text: str, src: str, tgt: str, ollama_model: str,
                     keep_alive: str = "10m",
                     ollama_url: str = "http://localhost:11434/api/generate") -> str:
    src_name = lang_display(src) if src not in ("auto", "unk") else "source text"
    tgt_name = lang_display(tgt)
    prompt = (
        f"Translate the following {src_name} into {tgt_name}. "
        f"Output only the translation itself, with no explanations, "
        f"pinyin, romanization, or quotation marks:\n{text}"
    )
    data = ollama_generate(
        ollama_model, prompt, keep_alive=keep_alive, ollama_url=ollama_url,
        options={"temperature": 0.2, "presence_penalty": 0},
    )
    return (data.get("response") or "").strip()


def translate_trans(text: str, src: str, tgt: str) -> str:
    """Call Google Translate via translate-shell (`trans`).
    Sends text to Google; this is not offline translation."""
    src_code = trans_code(src) if src not in ("auto", "unk") else ""
    pair = f"{src_code}:{trans_code(tgt)}"
    result = subprocess.run(
        ["trans", "-b", pair, text],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"trans failed: {result.stderr.strip()}")
    return result.stdout.strip()


def translate_text(text: str, src: str, tgt: str, engine: str,
                   ollama_model: str, keep_alive: str = "10m") -> str:
    if engine == "trans":
        return translate_trans(text, src, tgt)
    return translate_ollama(text, src, tgt, ollama_model, keep_alive=keep_alive)


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def list_langs():
    print(f"{'code':<6} {'name':<20}")
    for code, name in sorted(WHISPER_LANGS.items()):
        print(f"{code:<6} {name:<20}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def list_media_files(directory: Path, recursive: bool = True):
    files = []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for path in iterator:
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            files.append(path)
    return sorted(files)


def process_one(media_path: Path, args):
    media_path = media_path.resolve()
    if not media_path.exists():
        sys.exit(f"file not found: {media_path}")

    do_translate = args.tgt_lang is not None
    src_hint = args.src_lang
    src_srt_guess = None
    if src_hint not in ("auto", "unk"):
        src_srt_guess = media_srt_path(media_path, src_hint)
    reuse_asr = (
        not args.force
        and src_srt_guess is not None
        and src_srt_guess.is_file()
        and src_srt_guess.stat().st_size > 0
    )

    if reuse_asr:
        segments = load_srt_cues(src_srt_guess)
        if not segments:
            reuse_asr = False
        else:
            src_lang = src_hint
            print(f"    using existing transcription {src_srt_guess} "
                  f"({len(segments)} cues)")

    if not reuse_asr:
        n_stages = 3 if do_translate else 2
        model_path = None
        if args.asr == "whisper":
            model_path = ensure_model(args.model)
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "audio.wav"
            extract_audio(media_path, wav_path, stage=f"[1/{n_stages}]")
            asr_kwargs = dict(
                language=args.src_lang,
                use_vad=not args.no_vad,
                suppress_music=not args.no_suppress_music,
                vad_threshold=args.vad_threshold,
                vad_min_silence_ms=args.vad_min_silence,
                vad_max_speech_s=args.vad_max_speech,
                stage=f"[2/{n_stages}]",
            )
            if args.asr == "sensevoice":
                segments, detected_lang = transcribe_sensevoice(
                    wav_path,
                    language=args.src_lang,
                    use_vad=not args.no_vad,
                    suppress_music=not args.no_suppress_music,
                    vad_max_speech_s=args.vad_max_speech,
                    backend=args.sv_backend,
                    stage=f"[2/{n_stages}]",
                )
            else:
                segments, detected_lang = transcribe(
                    wav_path, model_path, **asr_kwargs,
                )
        src_lang = detected_lang if args.src_lang == "auto" else args.src_lang
        src_srt = media_srt_path(media_path, src_lang)
        write_srt_file(src_srt, segments)
        print(f"    wrote transcription {src_srt}")
    else:
        src_srt = src_srt_guess
        n_stages = 2 if do_translate else 1

    actually_translate = do_translate and src_lang != args.tgt_lang
    if do_translate and not actually_translate:
        print(f"    source and target are the same ({src_lang}), skipping translation")

    out_tag = args.tgt_lang if actually_translate else src_lang
    out_srt = media_srt_path(media_path, out_tag)

    if not actually_translate:
        if reuse_asr and not args.force:
            print(f"done. already have {src_srt}")
            return
        print(f"[{n_stages}/{n_stages}] writing subtitles: {out_srt}")
        write_srt_file(out_srt, segments)
        print(f"done. subtitles saved to: {out_srt}")
        return

    existing = []
    if not args.force and out_srt.is_file() and out_srt.stat().st_size > 0:
        existing = load_srt_blocks(out_srt)
        if len(existing) >= len(segments):
            print(f"skip: {out_srt} already has {len(existing)} cues")
            if not args.keep_src_srt and src_srt != out_srt and src_srt.is_file():
                src_srt.unlink()
                print(f"    removed checkpoint {src_srt}")
            print(f"done. subtitles saved to: {out_srt}")
            return
        if existing:
            print(f"    resuming translation at cue {len(existing) + 1}/"
                  f"{len(segments)} ({out_srt})")

    print(f"[{n_stages}/{n_stages}] writing subtitles: {out_srt}")
    ollama_model = args.ollama_model
    if args.engine == "ollama":
        if not ollama_model:
            ollama_model = DEFAULT_OLLAMA_MODEL
        ensure_ollama_model(ollama_model)
        print(f"    translation model: {ollama_model}")
        warmup_ollama(ollama_model, keep_alive=args.keep_alive)
    else:
        print("    using trans (translate-shell/Google); text will be sent to Google")

    n_done = len(existing)
    with open(out_srt, "w", encoding="utf-8") as f:
        idx = 1
        for start, end, body in existing:
            write_srt_cue(f, idx, start, end, body)
            idx += 1
        for start, end, src_text in segments[n_done:]:
            tgt_text = translate_text(
                src_text, src_lang, args.tgt_lang, args.engine,
                ollama_model=ollama_model, keep_alive=args.keep_alive,
            )
            body = [tgt_text, src_text] if args.bilingual else [tgt_text]
            write_srt_cue(f, idx, start, end, body)
            print(f"  [{idx}] {start:.1f}-{end:.1f}: (translated)")
            idx += 1

    if not args.keep_src_srt and src_srt != out_srt and src_srt.is_file():
        src_srt.unlink()
        print(f"    removed checkpoint {src_srt}")
    print(f"done. subtitles saved to: {out_srt}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate subtitles from video or audio "
                    "(local Whisper or SenseVoice, optional translation)",
    )
    ap.add_argument("media", type=Path, nargs="?",
                    help="video/audio file, or a directory of videos (*.mp4, *.mkv, …); "
                         "directories are scanned recursively")
    ap.add_argument("--from", dest="src_lang", default="auto", metavar="LANG",
                    type=lambda s: normalize_lang(s, allow_auto=True),
                    help="source language code, default auto (Whisper detect). Common: ja/en/zh/ko")
    ap.add_argument("--to", dest="tgt_lang", default=None, metavar="LANG",
                    type=lambda s: normalize_lang(s, allow_auto=False),
                    help="target language. Omit to transcribe only. Common: zh/en/ja")
    ap.add_argument("--asr", choices=["whisper", "sensevoice"], default="whisper",
                    help="speech recognition: whisper=whisper.cpp Vulkan (default), "
                         "sensevoice=SenseVoice Small via FunASR llama.cpp/ggml "
                         "(Japanese / zh/en/ja/ko/yue)")
    ap.add_argument("--sv-backend", choices=["auto", "cpu", "vulkan"], default="auto",
                    help="SenseVoice compute backend. auto tries Vulkan then CPU")
    ap.add_argument("--model", default=DEFAULT_WHISPER_MODEL,
                    help="Whisper ASR model: tiny/base/small/medium/large-v3/"
                         "large-v3-turbo (default large-v3-turbo; ignored with "
                         "--asr sensevoice). Not the translator; see --ollama-model")
    ap.add_argument("--ollama-model", default=None, metavar="NAME",
                    help="Ollama translation model, default qwen3.5. "
                         "See --list-models for locally installed models")
    ap.add_argument("--keep-alive", default="10m",
                    help="how long to keep the Ollama model in VRAM, to avoid reload")
    ap.add_argument("--bilingual", dest="bilingual", action="store_true", default=True,
                    help="bilingual SRT: translation plus original (on by default when translating)")
    ap.add_argument("--no-bilingual", dest="bilingual", action="store_false",
                    help="translation only, no original line")
    ap.add_argument("--no-vad", action="store_true",
                    help="disable VAD and process the full audio (do not skip silence)")
    ap.add_argument("--vad-threshold", type=float, default=0.5,
                    help="VAD speech threshold, default 0.5, range 0-1. "
                         "Lower (e.g. 0.3) catches more weak speech, with more false positives")
    ap.add_argument("--vad-min-silence", type=int, default=100,
                    help="VAD minimum silence in ms, default 100, used to split segments")
    ap.add_argument("--vad-max-speech", type=float, default=DEFAULT_VAD_MAX_SPEECH_S,
                    help="VAD max speech span in seconds, default 8, splits long "
                         "talk into shorter subtitle cues")
    ap.add_argument("--no-suppress-music", action="store_true",
                    help="keep non-speech placeholders such as (音楽)/(拍手)/(music)")
    ap.add_argument("--engine", choices=["ollama", "trans"], default="ollama",
                    help="translation engine: ollama=local offline (default), "
                         "trans=translate-shell/Google (faster, sends text online)")
    ap.add_argument("--force", action="store_true",
                    help="redo ASR and translation even if sidecar .srt files exist")
    ap.add_argument("--keep-src-srt", action="store_true",
                    help="when translating, also keep video.<src>.srt; default is to "
                         "delete that checkpoint after video.<tgt>.srt is complete")
    ap.add_argument("--log", default=None, metavar="PATH",
                    help="append stdout/stderr and the command line to this file "
                         "(default: <dir>/gen_srt.log next to the media)")
    ap.add_argument("--no-recursive", action="store_true",
                    help="when media is a directory, only process videos in that "
                         "folder, not subdirectories")
    ap.add_argument("--list-langs", action="store_true",
                    help="list Whisper language codes and exit")
    ap.add_argument("--list-models", action="store_true",
                    help="list local Ollama translation models and exit")
    args = ap.parse_args()

    if args.list_langs:
        list_langs()
        return
    if args.list_models:
        print_ollama_models()
        return

    if args.media is None:
        ap.error("please provide a video, audio file, or directory")

    media_path = args.media.expanduser()
    if not media_path.exists():
        sys.exit(f"file not found: {media_path.resolve()}")
    media_path = media_path.resolve()

    if args.log:
        log_path = Path(args.log).expanduser().resolve()
    elif media_path.is_dir():
        log_path = media_path / "gen_srt.log"
    else:
        log_path = media_path.parent / "gen_srt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_out, log_file)
    sys.stderr = Tee(orig_err, log_file)
    print(f"--- {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} ---")
    print(f"command: {sys.executable} {' '.join(sys.argv)}")
    print(f"cwd: {Path.cwd()}")
    print(f"log: {log_path}")

    try:
        if media_path.is_dir():
            files = list_media_files(media_path, recursive=not args.no_recursive)
            if not files:
                sys.exit(f"no video files in {media_path}")
            print(f"found {len(files)} video(s) in {media_path}")
            failed = []
            for i, path in enumerate(files, 1):
                print(f"\n==== [{i}/{len(files)}] {path.name} ====")
                try:
                    process_one(path, args)
                except KeyboardInterrupt:
                    print("interrupted")
                    raise
                except SystemExit as e:
                    msg = e.code if isinstance(e.code, str) else (e.code or "")
                    print(f"FAIL {path}: {msg}")
                    failed.append(path)
                except Exception:
                    traceback.print_exc()
                    print(f"FAIL {path}")
                    failed.append(path)
            print(f"\n==== summary: {len(files) - len(failed)} ok, {len(failed)} failed ====")
            for path in failed:
                print(f"  FAIL {path}")
            if failed:
                sys.exit(1)
            return
        process_one(media_path, args)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code)
        raise
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_file.close()
        print(f"log saved to {log_path}", file=orig_err)


if __name__ == "__main__":
    main()
