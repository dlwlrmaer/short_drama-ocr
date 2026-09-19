"""ASR-guided OCR frame-candidate evaluation on 《瑶瑶如她》 episodes 1/16/31.

Compares two OCR candidate-frame sets on the same frozen 1-second video grid and
the same inherited ROI:

  baseline   -> every fixed-grid frame (no ASR timing)
  asr_guided -> only fixed-grid frames that fall inside a real ASR segment +/- pad

Ground-truth text for scoring is always the human-verified SRT.  ASR text is only
used to choose *when* to look, never as OCR truth (``asr_text_used_as_truth=false``).

The script never rewrites the source video, SRT, existing ROI, or ASR transcript
data.  All artifacts go under --out.  Frame decoding records the true decoded
frame PTS via FFmpeg showinfo, matching short_drama-asr/docs/OCR_CONTRACT.md.

Reproduce (after the ASR transcripts exist):

  data/.venv/bin/python scripts/yaoyao_asr_guided_eval.py \
    --asr-transcript data/yaoyao_asr_guided_eval_fallback/asr/01-*/transcript.json \
    --asr-transcript data/yaoyao_asr_guided_eval_fallback/asr/16-*/transcript.json \
    --asr-transcript data/yaoyao_asr_guided_eval_fallback/asr/31-*/transcript.json \
    --out data/yaoyao_asr_guided_eval_fallback
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BENCHMARK = Path(__file__).with_name("benchmark_ocr_win.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_ocr_win", BENCHMARK)
_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_bench)

FROZEN_ROI = [0.1, 0.44, 0.85, 0.78]
FROZEN_PAD_MS = 500          # ASR segment padding for candidate windows
FROZEN_TOLERANCE_MS = 500    # cue<->frame intersection tolerance
FROZEN_GRID_MS = 1000        # fixed video time grid
PTS_RE = re.compile(r"pts_time:([-+]?\d+(?:\.\d+)?)")
SHA_RE = re.compile(r"^\s*([0-9a-f]{64})\s")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_box(roi, width, height):
    x0 = int(width * roi[0])
    y0 = int(height * roi[1])
    x1 = int(width * roi[2])
    y1 = int(height * roi[3])
    return x0, y0, x1 - x0, y1 - y0


def decode_frame(video: Path, requested_ms: int, crop, timeout=120):
    """Decode one frame at >= requested_ms and report the true decoded PTS."""
    seconds = requested_ms / 1000.0
    seek = max(0.0, seconds - 1.0)
    x, y, w, h = crop
    vf = f"select=gte(t\\,{seconds:.3f}),crop={w}:{h}:{x}:{y},showinfo"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info",
               "-ss", f"{seek:.3f}", "-copyts", "-i", str(video),
               "-frames:v", "1", "-vf", vf,
               "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=False, timeout=timeout)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"FFmpeg could not decode frame at {requested_ms} ms")
    match = PTS_RE.search(result.stderr.decode("utf-8", errors="replace"))
    if not match:
        raise RuntimeError(f"FFmpeg returned no frame PTS at {requested_ms} ms")
    return result.stdout, round(float(match.group(1)) * 1000)


def load_transcript(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def asr_windows(document: dict, pad_ms: int) -> list[tuple[int, int]]:
    windows = []
    for segment in document["segments"]:
        windows.append((max(0, segment["start_ms"] - pad_ms), segment["end_ms"] + pad_ms))
    return windows


def in_windows(moment: int, windows: list[tuple[int, int]]) -> bool:
    return any(start <= moment <= end for start, end in windows)


def build_candidates(duration_ms: int, windows: list[tuple[int, int]], grid_ms: int):
    baseline = list(range(0, max(0, duration_ms) + 1, grid_ms))
    guided = [moment for moment in baseline if in_windows(moment, windows)]
    return baseline, guided


def assign_frames(frames, cues, tolerance_ms):
    """Map each frame to the nearest cue within tolerance; return assignments + invalid."""
    assignment = {}
    for frame in frames:
        best = None
        for index, cue in enumerate(cues):
            if cue["start_ms"] - tolerance_ms <= frame["pts_ms"] <= cue["end_ms"] + tolerance_ms:
                middle = (cue["start_ms"] + cue["end_ms"]) / 2
                distance = abs(frame["pts_ms"] - middle)
                if best is None or distance < best[0]:
                    best = (distance, index)
        if best is not None:
            assignment[frame["candidate"]] = best[1]
    return assignment


def cue_text(cue_index, frames, assignment, cues):
    """Pick OCR text from the candidate frame nearest the cue midpoint.

    Deterministic, references only cue timing (never the SRT text), and mirrors the
    project's own convention of sampling one frame near each cue midpoint.  Ties on
    distance resolve to the earlier candidate.
    """
    return cue_text_and_frame(cue_index, frames, assignment, cues)[0]


def cue_text_and_frame(cue_index, frames, assignment, cues):
    midpoint = (cues[cue_index]["start_ms"] + cues[cue_index]["end_ms"]) / 2
    best = None
    for frame in frames:
        if assignment.get(frame["candidate"]) != cue_index:
            continue
        distance = abs(frame["pts_ms"] - midpoint)
        if best is None or distance < best[0]:
            best = (distance, frame)
    return (best[1]["ocr_text"], best[1]) if best else ("", None)


def cue_text_majority(cue_index, frames, assignment):
    texts = [frame["ocr_text"] for frame in frames
             if assignment.get(frame["candidate"]) == cue_index and frame["ocr_text"]]
    if not texts:
        return ""
    counts = {}
    for text in texts:
        counts[text] = counts.get(text, 0) + 1
    top = max(counts.values())
    winners = {text for text, count in counts.items() if count == top}
    for frame in frames:
        if assignment.get(frame["candidate"]) == cue_index and frame["ocr_text"] in winners:
            return frame["ocr_text"]
    return ""


def metrics_for_set(cues, frames, assignment, rule="midpoint"):
    candidate_ids = {frame["candidate"] for frame in frames}
    valid = {fid for fid in candidate_ids if fid in assignment}
    invalid = candidate_ids - valid
    covered = {assignment[fid] for fid in valid}
    rows = []
    outside = 0
    for index, cue in enumerate(cues):
        if rule == "midpoint":
            prediction, selected = cue_text_and_frame(index, frames, assignment, cues)
        else:
            prediction, selected = cue_text_majority(index, frames, assignment), None
        if selected is not None and not (cue["start_ms"] <= selected["pts_ms"] <= cue["end_ms"]):
            outside += 1
        row = _bench.score_result({"reference": cue["reference"], "prediction": prediction,
                                   "latency_ms": 0})
        row["covered"] = index in covered
        rows.append(row)
    total_chars = sum(row["reference_chars"] for row in rows)
    total_edits = sum(row["edit_distance"] for row in rows)
    latencies = sorted(frame["ocr_ms"] for frame in frames)
    p95 = latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)] if latencies else None
    return {
        "candidates": len(candidate_ids),
        "valid_frames": len(valid),
        "invalid_frames": len(invalid),
        "coverage_cues": len(covered),
        "coverage_rate": round(len(covered) / len(cues), 4) if cues else None,
        "uncovered_cues": len(cues) - len(covered),
        "cues_selected_outside_own_window": outside,
        "exact_matches": sum(row["edit_distance"] == 0 for row in rows),
        "exact_match_rate": round(sum(row["edit_distance"] == 0 for row in rows) / len(rows), 4)
        if rows else None,
        "exact_1_2": sum(row["edit_distance"] == 0 and row["length_band"] == "1-2" for row in rows),
        "exact_3_4": sum(row["edit_distance"] == 0 and row["length_band"] == "3-4" for row in rows),
        "band_1_2": sum(row["length_band"] == "1-2" for row in rows),
        "band_3_4": sum(row["length_band"] == "3-4" for row in rows),
        "proxy_cer": round(total_edits / total_chars, 4) if total_chars else None,
        "empty_results": sum(not row["detected"] for row in rows),
        "inserted_noise_results": sum(row["difference_kind"] == "contains_inserted_noise" for row in rows),
        "inserted_noise_chars": sum(row["edit_counts"]["insertions"] for row in rows),
        "ocr_latency_median_ms": latencies[len(latencies) // 2] if latencies else None,
        "ocr_latency_p95_ms": p95,
    }


def evaluate_episode(transcript_path: Path, srt_path: Path, video_path: Path, out_dir: Path,
                     roi, pad_ms, tolerance_ms, grid_ms, engine_name, rescore=False):
    document = load_transcript(transcript_path)
    duration_ms = document["audio_start_ms"] + document["duration_ms"]
    cues = _bench.read_srt(srt_path)
    if not cues:
        raise RuntimeError(f"No SRT cues in {srt_path}")
    windows = asr_windows(document, pad_ms)
    baseline, guided = build_candidates(duration_ms, windows, grid_ms)
    frames_dir = out_dir / "frames" / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True)
    width, height = (int(value) for value in probe.stdout.strip().split(","))
    crop = crop_box(roi, width, height)

    if rescore:
        cached = json.loads((out_dir / f"frames_{video_path.stem}.json").read_text(encoding="utf-8"))
        frames = cached["frames"]
        decode_times = cached.get("decode_times_ms") or [None] * len(frames)
        total_seconds = cached.get("frame_ocr_total_seconds")
        if total_seconds is None:
            total_seconds = round(sum(frame["ocr_ms"] for frame in frames) / 1000, 2)
    else:
        infer = _bench.make_engine(engine_name)
        frames = []
        decode_times = []
        started = time.perf_counter()
        for index, requested in enumerate(baseline):
            begin = time.perf_counter()
            payload, pts_ms = decode_frame(video_path, requested, crop)
            decode_times.append(round((time.perf_counter() - begin) * 1000))
            image_path = frames_dir / f"frame_{index:05d}_{requested:06d}.png"
            image_path.write_bytes(payload)
            begin = time.perf_counter()
            texts = infer(image_path)
            ocr_ms = round((time.perf_counter() - begin) * 1000)
            frames.append({
                "candidate": index, "requested_ms": requested, "pts_ms": pts_ms,
                "image": str(image_path.relative_to(out_dir)), "ocr_text": "".join(texts), "ocr_ms": ocr_ms,
            })
            if (index + 1) % 40 == 0:
                print(f"  {video_path.stem}: {index + 1}/{len(baseline)} frames", flush=True)
        total_seconds = round(time.perf_counter() - started, 2)
        (out_dir / f"frames_{video_path.stem}.json").write_text(
            json.dumps({"frames": frames, "windows": windows, "decode_times_ms": decode_times,
                        "frame_ocr_total_seconds": total_seconds},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    baseline_frames = frames
    guided_frames = [frame for frame in frames if frame["candidate"] in _guided_ids(baseline, guided)]
    assignment_all = assign_frames(frames, cues, tolerance_ms)
    baseline_metrics = metrics_for_set(cues, baseline_frames, assignment_all)
    guided_metrics = metrics_for_set(cues, guided_frames, assignment_all)
    baseline_metrics_majority = metrics_for_set(cues, baseline_frames, assignment_all, rule="majority")
    guided_metrics_majority = metrics_for_set(cues, guided_frames, assignment_all, rule="majority")

    decode_sorted = sorted(value for value in decode_times if value is not None)
    reports = {
        "episode": video_path.stem,
        "srt": str(srt_path),
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "srt_sha256": sha256_file(srt_path),
        "transcript": str(transcript_path),
        "transcript_sha256": sha256_file(transcript_path),
        "media_id": document["media_id"],
        "duration_ms": duration_ms,
        "asr": {
            "engine": document["configuration"]["engine"],
            "model_id": document["configuration"].get("model_id"),
            "model_revision": document["runtime"].get("model_revision"),
            "elapsed_seconds": document["runtime"].get("elapsed_seconds"),
            "realtime_factor": document["runtime"].get("realtime_factor"),
            "segments": len(document["segments"]),
            "coarse_window_segments": sum(
                segment.get("timestamp_granularity") == "window" for segment in document["segments"]),
            "window_count": len(document.get("windows", [])),
        },
        "roi": roi,
        "roi_source": "data/ocr_eval/ep14/manifest.json (reused, unchanged)",
        "crop_pixels": {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[3],
                        "canvas": [width, height]},
        "tolerances": {"asr_pad_ms": pad_ms, "cue_frame_tolerance_ms": tolerance_ms,
                       "grid_ms": grid_ms},
        "ocr_engine": engine_name,
        "frames_total": len(baseline),
        "frames_guided": len(guided_frames),
        "decode_latency_median_ms": decode_sorted[len(decode_sorted) // 2] if decode_sorted else None,
        "decode_latency_p95_ms": decode_sorted[max(0, (95 * len(decode_sorted) + 99) // 100 - 1)]
        if decode_sorted else None,
        "frame_ocr_total_seconds": total_seconds,
        "cue_text_rule": "frame nearest cue midpoint that is in the candidate set (no SRT text used)",
        "baseline": baseline_metrics,
        "asr_guided": guided_metrics,
        "baseline_majority_rule": baseline_metrics_majority,
        "asr_guided_majority_rule": guided_metrics_majority,
        "invalid_frame_reduction": baseline_metrics["invalid_frames"] - guided_metrics["invalid_frames"],
        "uncovered_cue_reduction": baseline_metrics["uncovered_cues"] - guided_metrics["uncovered_cues"],
        "asr_text_used_as_truth": False,
        "ocr_reference": "human_verified_srt",
    }
    (out_dir / f"episode_{video_path.stem}.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return reports


def _guided_ids(baseline, guided):
    guided_set = set(guided)
    return {index for index, moment in enumerate(baseline) if moment in guided_set}


def asr_not_ocr_truth_checks(reports, transcript_paths, out_dir):
    """Show ASR text is a timing source only; scoring reference stays the SRT.

    Cross-tabulates OCR text against ASR text and against the human SRT per cue to
    make explicit that ASR text is not reused as OCR ground truth.
    """
    checks = []
    for report, transcript_path in zip(reports, transcript_paths):
        document = load_transcript(transcript_path)
        frames = json.loads((out_dir / f"frames_{report['episode']}.json")
                            .read_text(encoding="utf-8"))["frames"]
        cues = _bench.read_srt(Path(report["srt"]))
        # map every frame to a cue for OCR-vs-ASR-vs-SRT comparison
        assignment = {}
        for frame in frames:
            for index, cue in enumerate(cues):
                if cue["start_ms"] - report["tolerances"]["cue_frame_tolerance_ms"] <= frame["pts_ms"] \
                        <= cue["end_ms"] + report["tolerances"]["cue_frame_tolerance_ms"]:
                    assignment[frame["candidate"]] = index
                    break
        ocr_matches_asr = 0
        ocr_matches_srt = 0
        asr_matches_srt = 0
        compared = 0
        for segment in document["segments"]:
            asr_text = _bench.normalize(segment["final_text"])
            overlap = [index for index, cue in enumerate(cues)
                       if cue["start_ms"] < segment["end_ms"] and cue["end_ms"] > segment["start_ms"]]
            if not overlap:
                continue
            index = overlap[0]
            ocr_text = cue_text(index, frames, assignment, cues)
            compared += 1
            if _bench.normalize(ocr_text) == asr_text:
                ocr_matches_asr += 1
            if _bench.normalize(ocr_text) == _bench.normalize(cues[index]["reference"]):
                ocr_matches_srt += 1
            if asr_text == _bench.normalize(cues[index]["reference"]):
                asr_matches_srt += 1
        checks.append({
            "episode": report["episode"],
            "segments_compared": compared,
            "ocr_text_equals_asr_text": ocr_matches_asr,
            "ocr_text_equals_human_srt": ocr_matches_srt,
            "asr_text_equals_human_srt": asr_matches_srt,
            "note": "OCR is scored against the human SRT; ASR text is only a timing guide.",
        })
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-transcript", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=["rapidocr", "rapidocr_v6_medium",
                                             "rapidocr_v5_mobile", "rapidocr_v5_server"],
                        default="rapidocr")
    parser.add_argument("--roi", type=float, nargs=4, default=FROZEN_ROI)
    parser.add_argument("--pad-ms", type=int, default=FROZEN_PAD_MS)
    parser.add_argument("--tolerance-ms", type=int, default=FROZEN_TOLERANCE_MS)
    parser.add_argument("--grid-ms", type=int, default=FROZEN_GRID_MS)
    parser.add_argument("--rescore", action="store_true",
                        help="reuse cached frames_*.json OCR text instead of re-running FFmpeg/OCR")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for transcript in args.asr_transcript:
        document = load_transcript(transcript)
        video = Path(document["source_file"])
        srt = Path("/vol1/@apphome/trim.openclaw/data/workspace/瑶瑶如她58/5-字幕SRT")
        srt_path = srt / f"{video.stem}.srt"
        print(f"=== {video.stem} ===", flush=True)
        reports.append(evaluate_episode(transcript, srt_path, video, args.out, list(args.roi),
                                        args.pad_ms, args.tolerance_ms, args.grid_ms, args.engine,
                                        rescore=args.rescore))
    checks = asr_not_ocr_truth_checks(reports, args.asr_transcript, args.out)
    tz = timezone(timedelta(hours=8))
    combined = {
        "schema_version": "1.0",
        "generated_at": datetime.now(tz).isoformat(),
        "asr_source": "short_drama-asr transcribe --engine whisper --model-id large-v3 --device cpu --no-align",
        "fallback_note": "ASR ran on a fallback model (faster-whisper large-v3) because the documented "
                         "Qwen/GPU path needs an RTX-class GPU; text is a timing guide only.",
        "roi": list(args.roi),
        "tolerances": {"asr_pad_ms": args.pad_ms, "cue_frame_tolerance_ms": args.tolerance_ms,
                       "grid_ms": args.grid_ms},
        "ocr_engine": args.engine,
        "asr_text_is_ocr_truth": False,
        "episodes": reports,
        "asr_not_ocr_truth_checks": checks,
    }
    (args.out / "eval_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": [(r["episode"], r["baseline"]["candidates"],
                                    r["asr_guided"]["candidates"]) for r in reports]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
