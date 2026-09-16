#!/usr/bin/env python3
"""
通用字幕生成：本地 whisper.cpp (Vulkan GPU) 识别语音，可选本地 Ollama /
translate-shell 翻译，写出 .srt。

任意 ffmpeg 能读的视频/音频均可。默认只识别、不翻译；源语言默认由 Whisper
自动检测，可用 --from 指定。加 --to 才翻译。

翻译引擎 (--engine)：
    ollama (默认): 完全本地离线。ja→zh 默认用 trans-ja 模型，其他语言对
        用 qwen2.5:7b + 通用翻译提示词。
    trans: 调用 translate-shell (trans 命令) 走 Google 翻译 API
        （文本会发送到 Google，非离线）。

用法:
    python3 gen_srt.py /path/to/video.mp4
    python3 gen_srt.py /path/to/video.mp4 --from ja
    python3 gen_srt.py /path/to/video.mp4 --to zh
    python3 gen_srt.py /path/to/video.mp4 --from ja --to zh
    python3 gen_srt.py /path/to/video.mp4 --from ja --to zh --no-bilingual
    python3 gen_srt.py /path/to/video.mp4 --model large-v3
    python3 gen_srt.py /path/to/video.mp4 --engine trans --to zh
    python3 gen_srt.py --list-langs
    python3 gen_srt.py --help

输出:
    只识别:   /path/to/video.<源语言>.srt
    翻译模式: /path/to/video.<目标语言>.srt
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER_CPP_DIR = ROOT / "whisper.cpp"
WHISPER_CLI = WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"
MODELS_DIR = WHISPER_CPP_DIR / "models"
VAD_MODEL = MODELS_DIR / "ggml-silero-v6.2.0.bin"

DEFAULT_JA_ZH_MODEL = "trans-ja"
DEFAULT_GENERIC_MODEL = "qwen2.5:7b"

# whisper.cpp 支持的语言代码 → 英文名（与 src/whisper.cpp g_lang 对齐）
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

# 翻译提示词里用的中文名；未列出的回退到英文名
LANG_ZH_NAMES = {
    "en": "英文", "zh": "简体中文", "ja": "日文", "ko": "韩文",
    "de": "德文", "fr": "法文", "es": "西班牙文", "ru": "俄文",
    "pt": "葡萄牙文", "it": "意大利文", "vi": "越南文", "th": "泰文",
    "ar": "阿拉伯文", "hi": "印地文", "id": "印尼文", "nl": "荷兰文",
    "pl": "波兰文", "tr": "土耳其文", "uk": "乌克兰文", "sv": "瑞典文",
    "yue": "粤语", "zh-CN": "简体中文",
}

# 用户输入别名 → whisper 代码
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

# translate-shell 对中文要用 zh-CN，其余与 whisper 代码一致
TRANS_CODES = {"zh": "zh-CN"}

AUTO_LANG_RE = re.compile(r"auto-detected language:\s+(\S+)\s+\(p\s*=")
SRT_BLOCK_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s*(.*)"
)
MUSIC_PLACEHOLDER_RE = re.compile(r"^[\(（][^)）]*[\)）]$|^[♪\s]+$")


def lang_display(code: str) -> str:
    if code in LANG_ZH_NAMES:
        return LANG_ZH_NAMES[code]
    if code in WHISPER_LANGS:
        return WHISPER_LANGS[code]
    return code


def normalize_lang(value: str, allow_auto: bool = False) -> str:
    raw = value.strip()
    key = raw.lower().replace("_", "-")
    if key in ("auto", "detect", ""):
        if allow_auto:
            return "auto"
        raise argparse.ArgumentTypeError("目标语言不能是 auto")
    resolved = LANG_ALIASES.get(raw) or LANG_ALIASES.get(key)
    if resolved == "auto":
        if allow_auto:
            return "auto"
        raise argparse.ArgumentTypeError("目标语言不能是 auto")
    if resolved:
        return resolved
    if key in WHISPER_LANGS:
        return key
    for code, name in WHISPER_LANGS.items():
        if name == key:
            return code
    hint = "（也可用 auto）" if allow_auto else ""
    raise argparse.ArgumentTypeError(f"未知语言: {value}{hint}。查看 --list-langs")


def trans_code(code: str) -> str:
    return TRANS_CODES.get(code, code)


def default_ollama_model(src: str, tgt: str) -> str:
    if src == "ja" and tgt == "zh":
        return DEFAULT_JA_ZH_MODEL
    return DEFAULT_GENERIC_MODEL


def extract_audio(media_path: Path, wav_path: Path, stage: str):
    cmd = [
        "ffmpeg", "-y", "-i", str(media_path),
        "-vn", "-ar", "16000", "-ac", "1",
        str(wav_path),
    ]
    print(f"{stage} 提取音频...")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        sys.exit(f"ffmpeg 失败:\n{err[-2000:]}")


def ensure_model(model_size: str) -> Path:
    model_path = MODELS_DIR / f"ggml-{model_size}.bin"
    if not model_path.exists():
        print(f"    下载模型 {model_size} ...")
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
        print(f"{stage} 用 whisper-cli (Vulkan GPU) 自动检测语言并识别语音...")
    else:
        print(f"{stage} 用 whisper-cli (Vulkan GPU) 识别{lang_display(language)}语音...")
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
        print(f"    已启用 VAD (语音活动检测)，跳过无人声段落 "
              f"(阈值={vad_threshold}, 最小静音时长={vad_min_silence_ms}ms)")
    elif use_vad:
        print(f"    警告: VAD 模型不存在 ({VAD_MODEL})，跳过 VAD，处理完整音频")
    if suppress_music:
        # -sns 抑制模型内置的非语音特殊 token
        # --suppress-regex 额外过滤 (音楽)/(拍手)/(music) 等占位符文本
        cmd += [
            "-sns",
            "--suppress-regex", r"[\(（][^)）]*[\)）]|♪+",
        ]
        print(f"    已启用非语音占位符抑制 (如 (音楽)/(拍手)/(music) 等)")
    env = os.environ.copy()
    lib_dir = str(WHISPER_CPP_DIR / "build" / "bin")
    env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    # 用 text=False 拿原始字节，再手动宽松解码，避免 whisper-cli 日志中
    # 偶发的非 UTF-8 字节 (进度条/终端控制字符等) 导致 UnicodeDecodeError 崩溃
    result = subprocess.run(cmd, capture_output=True, text=False, env=env)
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        sys.exit(f"whisper-cli 失败:\n{stderr_text[-2000:]}")

    detected = language
    if language == "auto":
        matches = AUTO_LANG_RE.findall(stderr_text)
        if matches:
            detected = matches[-1]
            print(f"    检测到源语言: {detected} ({lang_display(detected)})")
        else:
            detected = "unk"
            print("    警告: 未能从 whisper 日志解析到源语言，按未知语言处理")

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
    print(f"    识别到 {len(segments)} 段" + (f" (另过滤掉 {skipped} 段非语音占位符)" if skipped else ""))
    return segments, detected


def warmup_ollama(ollama_model: str, keep_alive: str = "10m",
                  ollama_url: str = "http://localhost:11434/api/generate"):
    """提前把模型加载进显存并设置保留时间，避免翻译时中途卸载重载。"""
    print(f"    预热 Ollama 模型 {ollama_model} (keep_alive={keep_alive}) ...")
    payload = json.dumps({
        "model": ollama_model,
        "prompt": "你好",
        "stream": False,
        "keep_alive": keep_alive,
    }).encode("utf-8")
    req = urllib.request.Request(
        ollama_url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        json.load(resp)
    print("    预热完成，模型已常驻显存")


def translate_ollama(text: str, src: str, tgt: str, ollama_model: str,
                     keep_alive: str = "10m",
                     ollama_url: str = "http://localhost:11434/api/generate") -> str:
    src_name = lang_display(src) if src not in ("auto", "unk") else "原文"
    tgt_name = lang_display(tgt)
    prompt = (
        f"请将下面的{src_name}翻译成{tgt_name}，只输出译文本身，"
        f"不要添加任何解释、拼音、罗马音或引号：\n{text}"
    )
    payload = json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(
        ollama_url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    return data["response"].strip()


def translate_trans(text: str, src: str, tgt: str) -> str:
    """用 translate-shell (trans 命令) 调用 Google 翻译 API。
    注意: 这会把文本发送到 Google 服务器, 不是本地离线翻译。"""
    src_code = trans_code(src) if src not in ("auto", "unk") else ""
    pair = f"{src_code}:{trans_code(tgt)}"
    result = subprocess.run(
        ["trans", "-b", pair, text],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"trans 命令失败: {result.stderr.strip()}")
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
    print(f"{'代码':<6} {'英文名':<20} 中文名")
    for code, name in sorted(WHISPER_LANGS.items()):
        zh = LANG_ZH_NAMES.get(code, "")
        print(f"{code:<6} {name:<20} {zh}")


def main():
    ap = argparse.ArgumentParser(
        description="从视频/音频生成字幕（本地 Whisper 识别，可选翻译）",
    )
    ap.add_argument("media", type=Path, nargs="?",
                    help="视频或音频文件（ffmpeg 能读即可）")
    ap.add_argument("--from", dest="src_lang", default="auto", metavar="LANG",
                    type=lambda s: normalize_lang(s, allow_auto=True),
                    help="源语言代码，默认 auto（Whisper 自动检测）。常用: ja/en/zh/ko")
    ap.add_argument("--to", dest="tgt_lang", default=None, metavar="LANG",
                    type=lambda s: normalize_lang(s, allow_auto=False),
                    help="目标语言。省略则不翻译，只输出识别结果。常用: zh/en/ja")
    ap.add_argument("--model", default="small",
                    help="whisper 模型: tiny/base/small/medium/large-v3")
    ap.add_argument("--ollama-model", default=None,
                    help="Ollama 模型名。默认 ja→zh 用 trans-ja，其他语言对用 qwen2.5:7b")
    ap.add_argument("--keep-alive", default="10m",
                    help="Ollama 模型在显存中保留的时间，避免频繁卸载重载")
    ap.add_argument("--bilingual", dest="bilingual", action="store_true", default=True,
                    help="双语字幕：每条同时显示译文和原文（翻译模式下默认开启）")
    ap.add_argument("--no-bilingual", dest="bilingual", action="store_false",
                    help="关闭双语字幕，只输出译文")
    ap.add_argument("--no-vad", action="store_true",
                    help="禁用 VAD，处理完整音频（不跳过静音段）")
    ap.add_argument("--vad-threshold", type=float, default=0.5,
                    help="VAD 判定为'有人声'的阈值，默认0.5，范围0-1。"
                         "调低(如0.3)对能量突变更敏感，能识别更多叫声/弱语音混合片段，但误检也会增多")
    ap.add_argument("--vad-min-silence", type=int, default=100,
                    help="VAD 最小静音时长(ms)，默认100，用于切分语音段落。调低能更快响应短促的声音变化")
    ap.add_argument("--no-suppress-music", action="store_true",
                    help="禁用 (音楽)/(拍手)/(music) 等非语音占位符抑制")
    ap.add_argument("--engine", choices=["ollama", "trans"], default="ollama",
                    help="翻译引擎: ollama=本地离线(默认), trans=translate-shell/Google(快，联网发送文本)")
    ap.add_argument("--list-langs", action="store_true",
                    help="列出 Whisper 支持的语言代码后退出")
    args = ap.parse_args()

    if args.list_langs:
        list_langs()
        return

    if args.media is None:
        ap.error("请提供视频或音频文件")

    media_path = args.media.resolve()
    if not media_path.exists():
        sys.exit(f"文件不存在: {media_path}")

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
            print(f"    源语言与目标语言相同 ({src_lang})，跳过翻译")

        out_tag = args.tgt_lang if actually_translate else src_lang
        out_srt = Path(str(media_path.with_suffix("")) + f".{out_tag}.srt")

        print(f"[{n_stages}/{n_stages}] 写入字幕: {out_srt}")
        ollama_model = args.ollama_model
        if actually_translate and args.engine == "ollama":
            if not ollama_model:
                ollama_model = default_ollama_model(src_lang, args.tgt_lang)
            warmup_ollama(ollama_model, keep_alive=args.keep_alive)
        elif actually_translate and args.engine == "trans":
            print("    使用 trans (translate-shell/Google) 翻译引擎 (文本将发送到 Google 服务器)")

        with open(out_srt, "w", encoding="utf-8") as f:
            idx = 1
            for start, end, src_text in segments:
                if not actually_translate:
                    f.write(f"{idx}\n")
                    f.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
                    f.write(f"{src_text}\n\n")
                    print(f"  [{idx}] {start:.1f}-{end:.1f}: (识别完成，长度 {len(src_text)} 字符)")
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
                print(f"  [{idx}] {start:.1f}-{end:.1f}: (已翻译)")
                idx += 1

    print(f"完成! 字幕已保存到: {out_srt}")


if __name__ == "__main__":
    main()
