#!/usr/bin/env python3
"""
Generate subtitles from video or audio: local whisper.cpp (Vulkan GPU) for
speech recognition, optional local Ollama or translate-shell for translation.

Any ffmpeg-readable file works. Default is transcribe-only (no translation);
Whisper auto-detects the source language, override with --from. Pass --to to
translate.

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
    python3 gen_srt.py /path/to/video.mp4 --model large-v3
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
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER_CPP_DIR = ROOT / "whisper.cpp"
WHISPER_CLI = WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"
MODELS_DIR = WHISPER_CPP_DIR / "models"
VAD_MODEL = MODELS_DIR / "ggml-silero-v6.2.0.bin"

DEFAULT_OLLAMA_MODEL = "qwen3.5"

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
                    options=None, timeout: int = 180) -> dict:
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
    if not model_path.exists():
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
        ]
        print(f"    VAD enabled, skipping non-speech "
              f"(threshold={vad_threshold}, min silence={vad_min_silence_ms}ms)")
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
    env = os.environ.copy()
    lib_dir = str(WHISPER_CPP_DIR / "build" / "bin")
    env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
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


def main():
    ap = argparse.ArgumentParser(
        description="Generate subtitles from video or audio (local Whisper, optional translation)",
    )
    ap.add_argument("media", type=Path, nargs="?",
                    help="video or audio file (anything ffmpeg can read)")
    ap.add_argument("--from", dest="src_lang", default="auto", metavar="LANG",
                    type=lambda s: normalize_lang(s, allow_auto=True),
                    help="source language code, default auto (Whisper detect). Common: ja/en/zh/ko")
    ap.add_argument("--to", dest="tgt_lang", default=None, metavar="LANG",
                    type=lambda s: normalize_lang(s, allow_auto=False),
                    help="target language. Omit to transcribe only. Common: zh/en/ja")
    ap.add_argument("--model", default="small",
                    help="Whisper ASR model: tiny/base/small/medium/large-v3 "
                         "(not the translator; see --ollama-model)")
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
    ap.add_argument("--no-suppress-music", action="store_true",
                    help="keep non-speech placeholders such as (音楽)/(拍手)/(music)")
    ap.add_argument("--engine", choices=["ollama", "trans"], default="ollama",
                    help="translation engine: ollama=local offline (default), "
                         "trans=translate-shell/Google (faster, sends text online)")
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
        ap.error("please provide a video or audio file")

    media_path = args.media.resolve()
    if not media_path.exists():
        sys.exit(f"file not found: {media_path}")

    do_translate = args.tgt_lang is not None
    n_stages = 3 if do_translate else 2

    model_path = ensure_model(args.model)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        extract_audio(media_path, wav_path, stage=f"[1/{n_stages}]")
        segments, detected_lang = transcribe(
            wav_path, model_path,
            language=args.src_lang,
            use_vad=not args.no_vad,
            suppress_music=not args.no_suppress_music,
            vad_threshold=args.vad_threshold,
            vad_min_silence_ms=args.vad_min_silence,
            stage=f"[2/{n_stages}]",
        )
        src_lang = detected_lang if args.src_lang == "auto" else args.src_lang
        actually_translate = do_translate and src_lang != args.tgt_lang
        if do_translate and not actually_translate:
            print(f"    source and target are the same ({src_lang}), skipping translation")

        out_tag = args.tgt_lang if actually_translate else src_lang
        out_srt = Path(str(media_path.with_suffix("")) + f".{out_tag}.srt")

        print(f"[{n_stages}/{n_stages}] writing subtitles: {out_srt}")
        ollama_model = args.ollama_model
        if actually_translate and args.engine == "ollama":
            if not ollama_model:
                ollama_model = DEFAULT_OLLAMA_MODEL
            ensure_ollama_model(ollama_model)
            print(f"    translation model: {ollama_model}")
            warmup_ollama(ollama_model, keep_alive=args.keep_alive)
        elif actually_translate and args.engine == "trans":
            print("    using trans (translate-shell/Google); text will be sent to Google")

        with open(out_srt, "w", encoding="utf-8") as f:
            idx = 1
            for start, end, src_text in segments:
                if not actually_translate:
                    f.write(f"{idx}\n")
                    f.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
                    f.write(f"{src_text}\n\n")
                    print(f"  [{idx}] {start:.1f}-{end:.1f}: (transcribed, {len(src_text)} chars)")
                    idx += 1
                    continue
                tgt_text = translate_text(
                    src_text, src_lang, args.tgt_lang, args.engine,
                    ollama_model=ollama_model, keep_alive=args.keep_alive,
                )
                f.write(f"{idx}\n")
                f.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
                if args.bilingual:
                    f.write(f"{tgt_text}\n{src_text}\n\n")
                else:
                    f.write(f"{tgt_text}\n\n")
                print(f"  [{idx}] {start:.1f}-{end:.1f}: (translated)")
                idx += 1

    print(f"done. subtitles saved to: {out_srt}")


if __name__ == "__main__":
    main()
