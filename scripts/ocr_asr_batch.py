"""Run subtitle OCR for a directory of independently generated ASR transcripts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.asr_adapter import save_evidence  # noqa: E402
from app.contracts import validate_segments  # noqa: E402
from app.ocr_engine import RapidOCRProvider  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.video_pipeline import process_video  # noqa: E402


def format_timestamp(value_ms: int) -> str:
    hours, remainder = divmod(max(0, value_ms), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(path: Path, segments: list[dict[str, Any]], detections: list[dict[str, Any]]) -> None:
    predictions = {row["segment_index"]: row["ocr_text"].strip() for row in detections}
    blocks = []
    for segment_index, segment in enumerate(segments):
        text = predictions.get(segment_index, "")
        if not text:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n{format_timestamp(segment['start_ms'])} --> "
            f"{format_timestamp(segment['end_ms'])}\n{text}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def episode_number(path: Path) -> int:
    return int(path.parent.name.removeprefix("episode_"))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量执行 ASR 时间窗引导的 OCR")
    parser.add_argument("asr_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--profile", choices=("auto", "win11", "linux", "linux-low-vram"), default="auto")
    parser.add_argument("--suppression", choices=("off", "spatial", "adaptive"), default="spatial")
    parser.add_argument("--roi", type=float, nargs=4, default=(0.12, 0.62, 0.88, 0.76))
    parser.add_argument("--center-band", type=float, nargs=2, default=(0.66, 0.74))
    parser.add_argument("--grid-ms", type=int, default=60_000)
    parser.add_argument("--dense-ms", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asr_dir = args.asr_dir.expanduser().resolve()
    output = (args.output or asr_dir.parent).expanduser().resolve()
    transcripts = sorted(asr_dir.glob("episode_*/transcript.json"), key=episode_number)
    selected = parse_episode_filter(args.episodes)
    if selected is not None:
        transcripts = [path for path in transcripts if episode_number(path) in selected]
    if not transcripts:
        raise SystemExit("没有找到 ASR transcript.json")
    base = Settings.for_profile(args.profile, background_suppression=args.suppression)
    settings = replace(
        base, roi=tuple(args.roi), subtitle_center_band=tuple(args.center_band),
        grid_ms=args.grid_ms, dense_ms=args.dense_ms,
    )
    provider = RapidOCRProvider(settings)
    provider.ensure_ready()
    locator = {
        "method": "visual_model",
        "roi": list(settings.roi),
        "subtitle_center_band": list(settings.subtitle_center_band),
        "sample_episodes": [1, 30, 59],
        "sample_frames": [
            str(path) for path in sorted((output / "locator_frames").glob("*.png"))
        ],
    }
    save_evidence(output / "subtitle_roi.json", locator)
    summaries = []
    failures = []
    started = time.perf_counter()
    print(json.dumps({"event": "provider_ready", **provider.status.to_dict()}, ensure_ascii=False), flush=True)
    for position, transcript_path in enumerate(transcripts, 1):
        episode = episode_number(transcript_path)
        evidence_path = output / "evidence" / f"episode_{episode:04d}_ocr.json"
        if evidence_path.exists() and not args.overwrite:
            try:
                existing = json.loads(evidence_path.read_text(encoding="utf-8"))
                if existing.get("guidance", {}).get("type") == "independent_asr_timing":
                    summary = existing["summary"]
                    summaries.append({"episode": episode, **summary, "status": "skipped"})
                    print(json.dumps({"event": "skip", "episode": episode}, ensure_ascii=False), flush=True)
                    continue
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        episode_started = time.perf_counter()
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            segments = validate_segments(transcript)
            video = Path(transcript["source_file"]).resolve()
            print(json.dumps({
                "event": "episode_start", "episode": episode,
                "position": position, "total": len(transcripts), "segments": len(segments),
            }, ensure_ascii=False), flush=True)
            detections = process_video(video, segments, provider, settings)
            summary = {
                "segments": len(segments),
                "detected_segments": len(detections),
                "coverage": round(len(detections) / len(segments), 4) if segments else None,
                "elapsed_seconds": round(time.perf_counter() - episode_started, 2),
            }
            evidence = {
                "schema_version": "1.0",
                "media_id": transcript["media_id"],
                "source_file": str(video),
                "timebase": "video_ms",
                "guidance": {
                    "type": "independent_asr_timing",
                    "source_file": str(transcript_path),
                    "asr_engine": transcript.get("asr"),
                    "asr_text_used_as_ocr_prompt": False,
                    "directory_srt_used": False,
                },
                "locator": locator,
                "settings": asdict(settings),
                "provider": provider.status.to_dict(),
                "segments": segments,
                "detections": detections,
                "summary": summary,
            }
            save_evidence(evidence_path, evidence)
            write_srt(output / "subtitles" / f"{episode:02d}.srt", segments, detections)
            summaries.append({"episode": episode, **summary, "status": "complete"})
            print(json.dumps({"event": "episode_complete", "episode": episode, **summary}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {"episode": episode, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps({"event": "episode_failed", **failure}, ensure_ascii=False), flush=True)
    total_segments = sum(row["segments"] for row in summaries)
    aggregate = {
        "episodes": len(summaries),
        "segments": total_segments,
        "detected_segments": sum(row["detected_segments"] for row in summaries),
        "coverage": round(sum(row["detected_segments"] for row in summaries) / total_segments, 4)
                    if total_segments else None,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report = {
        "guidance": "independent_asr_timing",
        "directory_srt_used": False,
        "provider": provider.status.to_dict(),
        "settings": asdict(settings),
        "aggregate": aggregate,
        "episodes": summaries,
        "failures": failures,
    }
    save_evidence(output / "ocr_report.json", report)
    print(json.dumps({"event": "batch_complete", **aggregate, "failures": len(failures)}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
