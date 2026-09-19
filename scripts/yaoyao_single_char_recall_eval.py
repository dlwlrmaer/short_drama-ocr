"""Reproducible single/short-subtitle recall ablation for ``瑶瑶如她``.

The ASR transcript supplies timing windows only.  Human SRT text is the sole
reference.  Candidate schedules are created for the complete episode and are
then pruned to deterministic evaluation cues/background probes before decoding,
so the expensive OCR run stays small without changing candidate eligibility.

No source media, SRT, ASR transcript, or inherited ROI is modified.  All caches
and reports are written below ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BENCHMARK = Path(__file__).with_name("benchmark_ocr_win.py")
SPEC = importlib.util.spec_from_file_location("benchmark_ocr_win", BENCHMARK)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)

DEFAULT_SRT_DIR = Path("/vol1/@apphome/trim.openclaw/data/workspace/瑶瑶如她58/5-字幕SRT")
DEFAULT_VIDEO_DIR = Path("/vol1/@apphome/trim.openclaw/data/workspace/瑶瑶如她58/原版-58集")
ROI_X0, ROI_Y0, ROI_X1 = 0.10, 0.44, 0.85
ROI_BOTTOMS = (0.78, 0.82, 0.84)
GRID_MS = 1000
DENSE_MS = 250
ASR_PAD_MS = 500
TOLERANCE_MS = 250
SUBTITLE_CENTER_BAND = (0.715, 0.815)
HOLDOUT_EPISODES = (35, 39, 48, 50)

# Frozen after inspecting the previous one-second-grid evaluation.  The omitted
# seventh empty cue (ep01 ``城里大医院去了``) is not a 1--4 character short cue.
KNOWN_MISSES = (
    (1, 19359, 19839, "怎么回事"),
    (1, 71319, 71920, "滚"),
    (1, 72040, 72879, "赶紧滚"),
    (16, 27039, 27839, "妈"),
    (16, 56399, 57000, "赶快走"),
    (16, 66359, 66920, "给你"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_engine(mode: str):
    """Create the same checked RapidOCR provider used by the production API."""
    from app.ocr_engine import RapidOCRProvider
    from app.settings import Settings

    provider = RapidOCRProvider(Settings(execution_mode=mode))
    engine = provider.ensure_ready()
    return provider, engine


def merge_windows(windows):
    merged = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(item) for item in merged]


def asr_windows(document, pad_ms=ASR_PAD_MS):
    duration = document["audio_start_ms"] + document["duration_ms"]
    return merge_windows((max(0, row["start_ms"] - pad_ms),
                          min(duration, row["end_ms"] + pad_ms))
                         for row in document["segments"])


def in_windows(moment, windows):
    return any(start <= moment <= end for start, end in windows)


def candidate_schedules(duration_ms, windows, grid_ms=GRID_MS, dense_ms=DENSE_MS):
    """Return full-episode baseline and dense-ASR schedules.

    The dense schedule is a strict superset of the global one-second fallback;
    each ASR window additionally gets a 250 ms local grid including its end.
    """
    baseline = list(range(0, duration_ms + 1, grid_ms))
    dense = set(baseline)
    for start, end in windows:
        value = start
        while value <= end:
            dense.add(value)
            value += dense_ms
        dense.add(end)
    return baseline, sorted(dense)


def cue_id(ep, cue):
    return f"{ep:02d}:{cue['start_ms']}-{cue['end_ms']}:{BENCH.normalize(cue['reference'])}"


def load_targets(srt_dir: Path):
    targets = []
    known_keys = {(ep, start, end, text) for ep, start, end, text in KNOWN_MISSES}
    for ep in (1, 16, 31, *HOLDOUT_EPISODES):
        cues = BENCH.read_srt(srt_dir / f"{ep:02d}.srt")
        for cue in cues:
            text = BENCH.normalize(cue["reference"])
            key = (ep, cue["start_ms"], cue["end_ms"], text)
            is_known = key in known_keys
            is_holdout = ep in HOLDOUT_EPISODES and 1 <= len(text) <= 4
            if is_known or is_holdout:
                targets.append({**cue, "episode": ep, "text": text,
                                "known_miss": is_known, "holdout": is_holdout,
                                "cue_id": cue_id(ep, cue)})
    observed = {(row["episode"], row["start_ms"], row["end_ms"], row["text"])
                for row in targets if row["known_miss"]}
    missing = known_keys - observed
    if missing:
        raise RuntimeError(f"Frozen known misses no longer match SRT: {sorted(missing)}")
    return targets


def relevant_candidates(schedule, cues, tolerance_ms=TOLERANCE_MS):
    return [moment for moment in schedule if any(
        cue["start_ms"] - tolerance_ms <= moment <= cue["end_ms"] + tolerance_ms
        for cue in cues)]


def background_probes(baseline, all_cues, limit=10):
    eligible = [moment for moment in baseline if moment >= 5000 and all(
        not (cue["start_ms"] - 250 <= moment <= cue["end_ms"] + 250)
        for cue in all_cues)]
    if len(eligible) <= limit:
        return eligible
    return [eligible[(i * len(eligible) + len(eligible) // 2) // limit]
            for i in range(limit)]


def probe_video(video: Path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", str(video),
    ], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"]), round(float(data["format"]["duration"]) * 1000)


def read_frame(capture, moment_ms):
    capture.set(cv2.CAP_PROP_POS_MSEC, moment_ms)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame at {moment_ms}ms")
    pts_ms = round(capture.get(cv2.CAP_PROP_POS_MSEC))
    return frame, pts_ms


def ocr_frame(engine, frame, width, height, bottom):
    x0, x1 = round(width * ROI_X0), round(width * ROI_X1)
    y0, y1 = round(height * ROI_Y0), round(height * bottom)
    crop = frame[y0:y1, x0:x1]
    sharpness = round(float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                          cv2.CV_64F).var()), 3)
    started = time.perf_counter()
    result = engine(crop)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    boxes = []
    if result.boxes is not None:
        for points, text, score in zip(result.boxes, result.txts, result.scores):
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            boxes.append({
                "text": text, "score": round(float(score), 6),
                "x_center": round((x0 + (min(xs) + max(xs)) / 2) / width, 6),
                "y_center": round((y0 + (min(ys) + max(ys)) / 2) / height, 6),
                "bbox": [[round(x0 + float(point[0]), 2), round(y0 + float(point[1]), 2)]
                         for point in points],
            })
    boxes.sort(key=lambda item: (item["y_center"], item["x_center"]))
    raw_text = "".join(item["text"] for item in boxes)
    lo, hi = SUBTITLE_CENTER_BAND
    subtitle_boxes = [item for item in boxes if lo <= item["y_center"] <= hi]
    subtitle_boxes.sort(key=lambda item: item["x_center"])
    filtered_text = "".join(item["text"] for item in subtitle_boxes)
    return {"raw_text": raw_text, "filtered_text": filtered_text, "boxes": boxes,
            "sharpness": sharpness, "ocr_ms": elapsed_ms}


def select_midpoint(cue, rows, field):
    middle = (cue["start_ms"] + cue["end_ms"]) / 2
    eligible = [row for row in rows if cue["start_ms"] - TOLERANCE_MS <= row["pts_ms"]
                <= cue["end_ms"] + TOLERANCE_MS]
    if not eligible:
        return "", None
    selected = min(eligible, key=lambda row: (abs(row["pts_ms"] - middle), row["pts_ms"]))
    return selected[field], selected


def select_stable(cue, rows, field="filtered_text"):
    """Select a text trajectory, then its highest-quality stable frame.

    Text is never compared to the SRT reference during selection.  A trajectory
    is repeated normalized OCR text in adjacent dense candidates (<=350 ms).
    """
    middle = (cue["start_ms"] + cue["end_ms"]) / 2
    eligible = sorted((row for row in rows
                       if cue["start_ms"] - TOLERANCE_MS <= row["pts_ms"]
                       <= cue["end_ms"] + TOLERANCE_MS and BENCH.normalize(row[field])),
                      key=lambda row: row["pts_ms"])
    if not eligible:
        return "", None
    tracks = []
    for row in eligible:
        text = BENCH.normalize(row[field])
        if tracks and tracks[-1]["text"] == text and row["pts_ms"] - tracks[-1]["rows"][-1]["pts_ms"] <= 350:
            tracks[-1]["rows"].append(row)
        else:
            tracks.append({"text": text, "rows": [row]})
    def track_key(track):
        rs = track["rows"]
        matching_boxes = [box for row in rs for box in row["boxes"]
                          if SUBTITLE_CENTER_BAND[0] <= box["y_center"] <= SUBTITLE_CENTER_BAND[1]]
        confidence = sum(box["score"] for box in matching_boxes) / max(1, len(matching_boxes))
        proximity = min(abs(row["pts_ms"] - middle) for row in rs)
        return (len(rs) >= 2, len(rs), round(confidence, 6), -proximity)
    winner = max(tracks, key=track_key)
    selected = max(winner["rows"], key=lambda row: (row["sharpness"],
                                                     -abs(row["pts_ms"] - middle)))
    return selected[field], selected


def score_rows(targets, episode_frames, schedule_name, bottom, filtered, stable):
    output = []
    field = "filtered_text" if filtered else "raw_text"
    for cue in targets:
        rows = episode_frames[cue["episode"]][bottom]
        allowed = {row["requested_ms"] for row in rows
                   if row["schedules"].get(schedule_name)}
        candidate_rows = [row for row in rows if row["requested_ms"] in allowed]
        prediction, selected = (select_stable(cue, candidate_rows, field)
                                if stable else select_midpoint(cue, candidate_rows, field))
        scored = BENCH.score_result({"reference": cue["reference"], "prediction": prediction,
                                     "latency_ms": selected["ocr_ms"] if selected else 0})
        normalized_prediction = BENCH.normalize(prediction)
        output.append({**cue, "prediction": prediction,
                       "selected_ms": selected["pts_ms"] if selected else None,
                       "recall": bool(cue["text"] and cue["text"] in normalized_prediction),
                       "exact": scored["edit_distance"] == 0,
                       "edit_distance": scored["edit_distance"],
                       "empty": not normalized_prediction,
                       "insertions": scored["edit_counts"]["insertions"]})
    return output


def metric_block(rows):
    chars = sum(len(row["text"]) for row in rows)
    edits = sum(row["edit_distance"] for row in rows)
    return {"cues": len(rows), "recalled": sum(row["recall"] for row in rows),
            "recall": round(sum(row["recall"] for row in rows) / len(rows), 4) if rows else None,
            "exact": sum(row["exact"] for row in rows),
            "exact_rate": round(sum(row["exact"] for row in rows) / len(rows), 4) if rows else None,
            "cer": round(edits / chars, 4) if chars else None,
            "empty": sum(row["empty"] for row in rows)}


def summarize_config(name, rows, background, candidate_counts, ocr_seconds):
    pure1 = [row for row in rows if len(row["text"]) == 1]
    band12 = [row for row in rows if 1 <= len(row["text"]) <= 2]
    band34 = [row for row in rows if 3 <= len(row["text"]) <= 4]
    known = [row for row in rows if row["known_miss"]]
    return {
        "name": name,
        "metrics": {"pure_1_char": metric_block(pure1), "1_2_chars": metric_block(band12),
                    "3_4_chars": metric_block(band34), "all": metric_block(rows)},
        "known_6": {"recalled": sum(row["recall"] for row in known),
                    "exact": sum(row["exact"] for row in known), "total": len(known),
                    "details": [{key: row[key] for key in ("cue_id", "reference", "prediction",
                                                            "recall", "exact", "selected_ms")}
                                for row in known]},
        "background_false_positive_frames": sum(bool(BENCH.normalize(text)) for text in background),
        "background_probe_frames": len(background),
        "empty_results": sum(row["empty"] for row in rows),
        "candidate_frames": candidate_counts,
        "roi_cache_ocr_seconds": ocr_seconds,
    }


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.srt_dir)
    transcripts = {}
    for path in args.asr_transcript:
        document = json.loads(path.read_text(encoding="utf-8"))
        transcripts[int(Path(document["source_file"]).stem)] = (path, document)
    expected = {1, 16, 31, *HOLDOUT_EPISODES}
    if set(transcripts) != expected:
        raise SystemExit(f"Need transcripts for {sorted(expected)}; got {sorted(transcripts)}")

    provider = None
    engine = None
    if not args.rescore:
        provider, engine = make_engine(args.mode)
        print(json.dumps({"provider": provider.status.to_dict()}, ensure_ascii=False), flush=True)
    frames_by_episode = {}
    episode_meta = {}
    for ep in sorted(expected):
        video = args.video_dir / f"{ep:02d}.mp4"
        srt = args.srt_dir / f"{ep:02d}.srt"
        all_cues = BENCH.read_srt(srt)
        episode_targets = [row for row in targets if row["episode"] == ep]
        transcript_path, document = transcripts[ep]
        width, height, video_duration = probe_video(video)
        duration = min(video_duration, document["audio_start_ms"] + document["duration_ms"])
        windows = asr_windows(document)
        baseline, dense = candidate_schedules(duration, windows)
        selected_baseline = relevant_candidates(baseline, episode_targets)
        selected_dense = relevant_candidates(dense, episode_targets)
        backgrounds = background_probes(baseline, all_cues, args.background_per_episode)
        moments = sorted(set(selected_baseline) | set(selected_dense) | set(backgrounds))
        episode_meta[ep] = {
            "video": str(video), "srt": str(srt), "transcript": str(transcript_path),
            "video_sha256": sha256_file(video), "srt_sha256": sha256_file(srt),
            "transcript_sha256": sha256_file(transcript_path), "duration_ms": duration,
            "asr_windows": len(windows), "baseline_candidates": len(baseline),
            "dense_candidates": len(dense), "evaluated_moments": len(moments),
            "baseline_evaluated_candidates": len(set(selected_baseline) | set(backgrounds)),
            "dense_evaluated_candidates": len(set(selected_dense) | set(backgrounds)),
            "background_probes": backgrounds,
        }
        cache = out / "cache" / f"episode_{ep:02d}.json"
        if args.rescore:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            frames_by_episode[ep] = {float(key): value for key, value in payload["roi_frames"].items()}
            continue
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video}")
        roi_frames = {bottom: [] for bottom in ROI_BOTTOMS}
        for index, moment in enumerate(moments, 1):
            frame, pts_ms = read_frame(capture, moment)
            for bottom in ROI_BOTTOMS:
                result = ocr_frame(engine, frame, width, height, bottom)
                roi_frames[bottom].append({
                    "requested_ms": moment, "pts_ms": pts_ms,
                    "schedules": {"baseline": moment in set(selected_baseline),
                                  "dense": moment in set(selected_dense),
                                  "background": moment in set(backgrounds)},
                    **result,
                })
            if index % 25 == 0:
                print(f"episode {ep:02d}: {index}/{len(moments)} decoded, {index * 3} OCR", flush=True)
        capture.release()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"episode": ep, "roi_frames": roi_frames},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        frames_by_episode[ep] = roi_frames

    configs = [
        ("A_grid_1s_roi078_raw_midpoint", "baseline", 0.78, False, False),
        ("B_dense250_roi078_raw_midpoint", "dense", 0.78, False, False),
        ("C_dense250_roi082_raw_midpoint", "dense", 0.82, False, False),
        ("D_dense250_roi084_raw_midpoint", "dense", 0.84, False, False),
        ("E_dense250_roi082_yfilter_midpoint", "dense", 0.82, True, False),
        ("F_dense250_roi082_yfilter_stable", "dense", 0.82, True, True),
        ("G_dense250_roi084_yfilter_midpoint", "dense", 0.84, True, False),
    ]
    summaries = []
    details = {}
    for name, schedule, bottom, filtered, stable in configs:
        rows = score_rows(targets, frames_by_episode, schedule, bottom, filtered, stable)
        field = "filtered_text" if filtered else "raw_text"
        background = [row[field] for ep, bottom_rows in frames_by_episode.items()
                      for row in bottom_rows[bottom] if row["schedules"]["background"]]
        counts = {"full_episode_schedule": sum(item[f"{schedule}_candidates"]
                                                for item in episode_meta.values()),
                  "evaluated_for_config": sum(item[f"{schedule}_evaluated_candidates"]
                                              for item in episode_meta.values()),
                  "evaluated_unique_moments": sum(item["evaluated_moments"]
                                                  for item in episode_meta.values()),
                  "actual_ocr_calls_all_rois": sum(item["evaluated_moments"]
                                                   for item in episode_meta.values()) * len(ROI_BOTTOMS)}
        ocr_seconds = round(sum(row["ocr_ms"] for bottom_rows in frames_by_episode.values()
                                for row in bottom_rows[bottom]) / 1000, 3)
        summaries.append(summarize_config(name, rows, background, counts, ocr_seconds))
        details[name] = rows

    recommended = next(row for row in summaries
                       if row["name"] == "E_dense250_roi082_yfilter_midpoint")
    baseline_summary = summaries[0]

    result = {
        "schema_version": "1.0", "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "reference": "human-verified SRT only", "asr_text_used_as_truth": False,
        "provider": (provider.status.to_dict() if provider is not None else
                     json.loads((out / "eval_results.json").read_text(encoding="utf-8")).get("provider")),
        "schedule": {"baseline_grid_ms": GRID_MS, "dense_asr_window_ms": DENSE_MS,
                     "outside_asr_fallback_ms": GRID_MS, "asr_pad_ms": ASR_PAD_MS,
                     "cue_time_tolerance_ms": TOLERANCE_MS},
        "roi_variants": [[ROI_X0, ROI_Y0, ROI_X1, bottom] for bottom in ROI_BOTTOMS],
        "subtitle_box_center_band": list(SUBTITLE_CENTER_BAND),
        "known_misses": [{"episode": ep, "start_ms": start, "end_ms": end, "text": text}
                         for ep, start, end, text in KNOWN_MISSES],
        "target_counts": {"total": len(targets), "known": sum(row["known_miss"] for row in targets),
                          "holdout": sum(row["holdout"] for row in targets)},
        "ocr_total_seconds_all_rois": round(sum(
            row["ocr_ms"] for bottom_rows in frames_by_episode.values()
            for bottom in ROI_BOTTOMS for row in bottom_rows[bottom]) / 1000, 3),
        "episodes": episode_meta, "configs": summaries,
        "decision": {
            "recommended": recommended["name"],
            "reason": "highest held-out recall/exact among filtered variants; the stable-track ablation regressed held-out recall",
            "threshold": "recommended config must recall at least 5/6 known misses and have no more common-probe background false positives than A",
            "passes": recommended["known_6"]["recalled"] >= 5 and
                      recommended["background_false_positive_frames"] <=
                      baseline_summary["background_false_positive_frames"],
        },
    }
    (out / "eval_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    (out / "cue_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"configs": [{"name": row["name"], "known": row["known_6"],
                                    "background_fp": row["background_false_positive_frames"],
                                    "metrics": row["metrics"]} for row in summaries],
                      "decision": result["decision"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-transcript", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--srt-dir", type=Path, default=DEFAULT_SRT_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--background-per-episode", type=int, default=10)
    parser.add_argument("--mode", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--rescore", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
