"""Create ASR timing transcripts from video audio without reading subtitle files.

Run this script with a virtual environment containing FunASR or faster-whisper.
One model instance is reused for every episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import subprocess
import wave
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def episode_key(path: Path) -> int | None:
    match = re.fullmatch(r"0*(\d+)", re.sub(r"\s+", "", path.stem))
    return int(match.group(1)) if match else None


def discover_videos(root: Path) -> list[tuple[int, Path]]:
    videos: dict[int, Path] = {}
    for path in root.rglob("*.mp4"):
        if any(part.lower().startswith("ocr") for part in path.parts):
            continue
        key = episode_key(path)
        if key is not None:
            videos.setdefault(key, path.resolve())
    return sorted(videos.items())


def parse_episode_filter(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    selected: set[int] = set()
    for raw in values:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, last = map(int, part.split("-", 1))
                selected.update(range(min(first, last), max(first, last) + 1))
            else:
                selected.add(int(part))
    return selected


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as target:
            json.dump(document, target, ensure_ascii=False, indent=2)
            target.write("\n")
            temporary = Path(target.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def timing_segments(raw_segments: Any) -> list[dict[str, Any]]:
    """Split ASR words into subtitle-sized timing windows without external text."""
    output: list[dict[str, Any]] = []
    punctuation = set("。！？!?；;")
    for raw in raw_segments:
        words = list(raw.words or [])
        if not words:
            text = raw.text.strip()
            if text and raw.end > raw.start:
                output.append({
                    "start_ms": round(raw.start * 1000),
                    "end_ms": round(raw.end * 1000),
                    "asr_text": text,
                })
            continue
        group: list[Any] = []

        def flush() -> None:
            if not group:
                return
            text = "".join(word.word for word in group).strip()
            start_ms = round(group[0].start * 1000)
            end_ms = round(group[-1].end * 1000)
            if text and end_ms > start_ms:
                output.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "asr_text": text,
                })
            group.clear()

        for word in words:
            if word.start is None or word.end is None:
                continue
            gap_ms = round((word.start - group[-1].end) * 1000) if group else 0
            duration_ms = round((word.end - group[0].start) * 1000) if group else 0
            if group and (gap_ms >= 450 or duration_ms > 2600):
                flush()
            group.append(word)
            if any(mark in word.word for mark in punctuation):
                flush()
        flush()
    # Remove exact duplicate windows produced around Whisper chunk boundaries.
    deduplicated: list[dict[str, Any]] = []
    for item in output:
        if deduplicated and item["asr_text"] == deduplicated[-1]["asr_text"] \
                and item["start_ms"] < deduplicated[-1]["end_ms"]:
            if item["end_ms"] > deduplicated[-1]["end_ms"]:
                deduplicated[-1]["end_ms"] = item["end_ms"]
            continue
        deduplicated.append(item)
    return deduplicated


def grouped_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    punctuation = set("。！？!?；;")
    group: list[dict[str, Any]] = []

    def flush() -> None:
        if not group:
            return
        text = "".join(item["text"] for item in group).strip()
        if text and group[-1]["end_ms"] > group[0]["start_ms"]:
            output.append({
                "start_ms": group[0]["start_ms"],
                "end_ms": group[-1]["end_ms"],
                "asr_text": text,
            })
        group.clear()

    for token in tokens:
        gap_ms = token["start_ms"] - group[-1]["end_ms"] if group else 0
        duration_ms = token["end_ms"] - group[0]["start_ms"] if group else 0
        if group and (gap_ms >= 450 or duration_ms > 2600):
            flush()
        group.append(token)
        if any(mark in token["text"] for mark in punctuation):
            flush()
    flush()
    return output


def transcribe_funasr(model: Any, video: Path) -> tuple[list[dict[str, Any]], str, float]:
    """Run 20 s overlapping windows and retain character timestamps."""
    with tempfile.TemporaryDirectory(prefix="short-drama-asr-") as temp_name:
        temp = Path(temp_name)
        audio = temp / "audio.wav"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(audio),
        ], check=True)
        tokens: list[dict[str, Any]] = []
        with wave.open(str(audio), "rb") as source:
            rate = source.getframerate()
            total_ms = round(source.getnframes() * 1000 / rate)
            start_ms = 0
            window_index = 0
            while start_ms < total_ms:
                end_ms = min(start_ms + 20_000, total_ms)
                start_frame = round(start_ms * rate / 1000)
                end_frame = round(end_ms * rate / 1000)
                source.setpos(start_frame)
                frames = source.readframes(end_frame - start_frame)
                clip = temp / f"clip-{window_index:05d}.wav"
                with wave.open(str(clip), "wb") as target:
                    target.setparams(source.getparams())
                    target.writeframes(frames)
                result = model.generate(
                    input=[str(clip)], cache={}, batch_size=1,
                    language="中文", itn=False,
                )[0]
                timestamps = result.get("timestamp", []) or []
                recognized_tokens = str(result.get("text", "")).split()
                for token_index, item in enumerate(timestamps):
                    if len(item) >= 3:
                        token_text = str(item[0])
                        local_start, local_end = round(item[1]), round(item[2])
                    elif len(item) == 2 and token_index < len(recognized_tokens):
                        token_text = recognized_tokens[token_index]
                        local_start, local_end = round(item[0]), round(item[1])
                    else:
                        continue
                    # The first 2 s of every later window overlap the prior one.
                    if window_index and (local_start + local_end) / 2 < 2000:
                        continue
                    tokens.append({
                        "text": token_text,
                        "start_ms": start_ms + local_start,
                        "end_ms": start_ms + local_end,
                    })
                if end_ms == total_ms:
                    break
                start_ms = end_ms - 2000
                window_index += 1
        return grouped_tokens(tokens), "zh", 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从视频音轨批量生成独立 ASR 时间轴")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--engine", choices=("funasr", "whisper"), default="funasr")
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = args.root.expanduser().resolve()
    output = (args.output or root / "ocr" / "asr").expanduser().resolve()
    selected = parse_episode_filter(args.episodes)
    videos = discover_videos(root)
    if selected is not None:
        videos = [item for item in videos if item[0] in selected]
    if not videos:
        raise SystemExit("没有找到视频")

    model_id = args.model or (
        "FunAudioLLM/Fun-ASR-Nano-2512" if args.engine == "funasr" else "large-v3"
    )
    started = time.perf_counter()
    print(json.dumps({"event": "model_loading", "engine": args.engine, "model": model_id}, ensure_ascii=False), flush=True)
    if args.engine == "funasr":
        from funasr import AutoModel
        model = AutoModel(
            model=model_id, trust_remote_code=True,
            device="cuda:0" if args.device == "cuda" else "cpu",
            hub="ms", disable_update=True,
        )
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_id, device=args.device, compute_type=args.compute_type)
    print(json.dumps({"event": "model_ready", "episodes": len(videos)}, ensure_ascii=False), flush=True)
    failures = []
    for position, (episode, video) in enumerate(videos, 1):
        path = output / f"episode_{episode:04d}" / "transcript.json"
        if path.exists() and not args.overwrite:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("source_file") == str(video) and existing.get("asr", {}).get("model") == model_id:
                    print(json.dumps({"event": "skip", "episode": episode}, ensure_ascii=False), flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        episode_started = time.perf_counter()
        try:
            print(json.dumps({
                "event": "episode_start", "episode": episode,
                "position": position, "total": len(videos), "video": str(video),
            }, ensure_ascii=False), flush=True)
            if args.engine == "funasr":
                segments, language, language_probability = transcribe_funasr(model, video)
            else:
                raw_segments, info = model.transcribe(
                    str(video), language="zh", word_timestamps=True,
                    vad_filter=False, condition_on_previous_text=False,
                    beam_size=5,
                )
                segments = timing_segments(raw_segments)
                language, language_probability = info.language, info.language_probability
            document = {
                "schema_version": "1.0",
                "media_id": sha256_file(video),
                "source_file": str(video),
                "timebase": "video_ms",
                "asr": {
                    "engine": args.engine,
                    "model": model_id,
                    "language": language,
                    "language_probability": round(language_probability, 4),
                    "subtitle_files_read": False,
                },
                "segments": segments,
                "elapsed_seconds": round(time.perf_counter() - episode_started, 2),
            }
            atomic_json(path, document)
            print(json.dumps({
                "event": "episode_complete", "episode": episode,
                "segments": len(segments),
                "elapsed_seconds": document["elapsed_seconds"],
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {"episode": episode, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps({"event": "episode_failed", **failure}, ensure_ascii=False), flush=True)
    report = {
        "engine": args.engine,
        "model": model_id,
        "episodes_requested": len(videos),
        "failures": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "subtitle_files_read": False,
    }
    atomic_json(output / "asr_report.json", report)
    print(json.dumps({"event": "batch_complete", **report}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
