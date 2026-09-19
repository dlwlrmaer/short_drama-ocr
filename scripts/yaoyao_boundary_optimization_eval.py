"""Boundary-aware frame selection ablation for the Yaoyao short-cue set.

This is intentionally a cache-first follow-up to ``yaoyao_single_char_recall_eval``.
It reuses the frozen recommendation-E ROI cache and adds only deterministic cue-
interior anchors when fewer than three cached frames survive a 100/150 ms trim.
Human SRT text is used only for scoring; neither candidate generation nor frame
selection compares OCR output with the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2


HERE = Path(__file__).parent
BASE_SPEC = importlib.util.spec_from_file_location(
    "yaoyao_single_char_recall_eval", HERE / "yaoyao_single_char_recall_eval.py")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

DEFAULT_SOURCE = Path("data/yaoyao_single_char_recall_eval")
DEFAULT_OUT = Path("data/yaoyao_boundary_optimization_eval")
TRIMS_MS = (100, 150)
MIN_CANDIDATES = 3
ANCHOR_MATCH_MS = 80
SINGLE_CONFIRM_GAP_MS = 100
DECODE_PTS_GUARD_MS = 40


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cue_window(cue, trim_ms):
    start, end = cue["start_ms"] + trim_ms, cue["end_ms"] - trim_ms
    if start > end:
        midpoint = round((cue["start_ms"] + cue["end_ms"]) / 2)
        return midpoint, midpoint
    return start, end


def interior_anchors(cue, trim_ms):
    start, end = cue_window(cue, trim_ms)
    # OpenCV seeks to the nearest decoded frame (20--40 ms in these files).  Keep
    # requested edge anchors one frame farther inward so their actual PTS still
    # satisfies the advertised trim.
    if end - start >= 2 * DECODE_PTS_GUARD_MS:
        start += DECODE_PTS_GUARD_MS
        end -= DECODE_PTS_GUARD_MS
    return sorted({start, round((start + end) / 2), end})


def planned_augmentations(targets, source_frames, trims=TRIMS_MS):
    """Return cue-interior moments needed to reach three candidates per trim.

    Planning uses timings only.  A cached decoded PTS within 80 ms covers an
    anchor; OCR text and the SRT reference never affect whether a frame is added.
    """
    requested = defaultdict(set)
    reasons = defaultdict(lambda: defaultdict(list))
    for cue in targets:
        ep = cue["episode"]
        for trim in trims:
            lo, hi = cue_window(cue, trim)
            available = [row["pts_ms"] for row in source_frames[ep] if lo <= row["pts_ms"] <= hi]
            for moment in interior_anchors(cue, trim):
                if len(available) >= MIN_CANDIDATES:
                    break
                if any(abs(pts - moment) <= ANCHOR_MATCH_MS for pts in available):
                    continue
                requested[ep].add(moment)
                available.append(moment)
                reasons[ep][moment].append({"cue_id": cue["cue_id"], "trim_ms": trim})
    return {ep: sorted(values) for ep, values in requested.items()}, reasons


def subtitle_boxes(row):
    lo, hi = BASE.SUBTITLE_CENTER_BAND
    return [box for box in row.get("boxes", []) if lo <= box["y_center"] <= hi
            and BASE.BENCH.normalize(box["text"])]


def confidence(row, chosen_text=None):
    boxes = subtitle_boxes(row)
    if chosen_text is not None:
        wanted = BASE.BENCH.normalize(chosen_text)
        boxes = [box for box in boxes if BASE.BENCH.normalize(box["text"]) == wanted]
    return sum(box["score"] for box in boxes) / len(boxes) if boxes else 0.0


def stable_box_tracks(rows, min_gap_ms=SINGLE_CONFIRM_GAP_MS):
    """Build exact-text box tracks with temporal and geometric stability stats."""
    grouped = defaultdict(list)
    for row in rows:
        for box in subtitle_boxes(row):
            grouped[BASE.BENCH.normalize(box["text"])].append((row, box))
    tracks = {}
    for text, items in grouped.items():
        # At most one observation per decoded PTS counts as temporal support.
        by_pts = {}
        for row, box in items:
            current = by_pts.get(row["pts_ms"])
            if current is None or box["score"] > current[1]["score"]:
                by_pts[row["pts_ms"]] = (row, box)
        items = [by_pts[key] for key in sorted(by_pts)]
        pts = [row["pts_ms"] for row, _ in items]
        confirmed = len(pts) >= 2 and max(pts) - min(pts) >= min_gap_ms
        xs = [box["x_center"] for _, box in items]
        ys = [box["y_center"] for _, box in items]
        x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
        spatial_std = math.sqrt(sum((x - x_mean) ** 2 + (y - y_mean) ** 2
                                    for x, y in zip(xs, ys)) / len(xs))
        tracks[text] = {
            "text": text, "items": items, "support": len(items), "confirmed": confirmed,
            "span_ms": max(pts) - min(pts) if pts else 0,
            "mean_confidence": sum(box["score"] for _, box in items) / len(items),
            "x_mean": x_mean, "y_mean": y_mean, "spatial_std": spatial_std,
        }
    return tracks


def track_rank(track):
    """Reference-free rank: support, confidence, geometry, then subtitle centre."""
    return (track["confirmed"], track["support"], round(track["mean_confidence"], 6),
            -round(track["spatial_std"], 6), -abs(track["x_mean"] - 0.5),
            -abs(track["y_mean"] - 0.76))


def row_sharpness_score(row, rows):
    values = [item["sharpness"] for item in rows]
    lo, hi = min(values), max(values)
    return (row["sharpness"] - lo) / (hi - lo) if hi > lo else 0.5


def select_quality(cue, inner_rows, fallback_rows, require_single_confirmation=True):
    """Choose stable OCR boxes and the best supporting frame.

    Confirmed box text is preferred over full-frame text, which removes unstable
    same-band clutter such as the episode-50 clothing logo.  For a one-character
    result, two observations at least 100 ms apart are required when available.
    If that evidence does not exist, the highest-quality visible single frame is
    retained.  If trimming removed every visible frame, the same fallback is
    attempted on untrimmed recommendation-E candidates.
    """
    midpoint = (cue["start_ms"] + cue["end_ms"]) / 2

    def choose(rows, stage):
        visible = [row for row in rows if BASE.BENCH.normalize(row["filtered_text"])]
        if not visible:
            return None
        tracks = stable_box_tracks(visible)
        confirmed = [track for track in tracks.values() if track["confirmed"]]
        if confirmed:
            winner = max(confirmed, key=track_rank)
            candidate_rows = [row for row, _ in winner["items"]]
            selected = max(candidate_rows, key=lambda row: (
                round(winner["mean_confidence"], 6),
                round(row_sharpness_score(row, candidate_rows), 6),
                -abs(row["pts_ms"] - midpoint)))
            return winner["text"], selected, {
                "selection_stage": stage, "evidence": "confirmed_box_track",
                "evidence_frames": winner["support"], "evidence_span_ms": winner["span_ms"],
                "mean_confidence": round(winner["mean_confidence"], 6),
                "spatial_std": round(winner["spatial_std"], 6),
            }

        # Explicit short-cue safeguard: do not turn a single visible frame empty.
        selected = max(visible, key=lambda row: (
            round(confidence(row), 6), round(row_sharpness_score(row, visible), 6),
            -abs(row["pts_ms"] - midpoint)))
        text = BASE.BENCH.normalize(selected["filtered_text"])
        evidence = "single_frame_fallback" if require_single_confirmation and len(text) == 1 \
            else "quality_frame_fallback"
        return text, selected, {
            "selection_stage": stage, "evidence": evidence,
            "evidence_frames": 1, "evidence_span_ms": 0,
            "mean_confidence": round(confidence(selected), 6), "spatial_std": None,
        }

    result = choose(inner_rows, "trimmed")
    if result is not None:
        return result
    result = choose(fallback_rows, "untrimmed_fallback")
    if result is not None:
        return result
    return "", None, {"selection_stage": "empty", "evidence": "none",
                       "evidence_frames": 0, "evidence_span_ms": 0,
                       "mean_confidence": 0.0, "spatial_std": None}


def select_trimmed_midpoint(cue, inner_rows, fallback_rows):
    middle = (cue["start_ms"] + cue["end_ms"]) / 2
    pool = inner_rows or fallback_rows
    if not pool:
        return "", None, {"selection_stage": "empty", "evidence": "none"}
    selected = min(pool, key=lambda row: (abs(row["pts_ms"] - middle), row["pts_ms"]))
    return selected["filtered_text"], selected, {
        "selection_stage": "trimmed" if inner_rows else "untrimmed_fallback",
        "evidence": "midpoint",
    }


def score_prediction(cue, prediction, selected, selection, candidate_count):
    scored = BASE.BENCH.score_result({"reference": cue["reference"], "prediction": prediction,
                                      "latency_ms": selected["ocr_ms"] if selected else 0})
    normalized = BASE.BENCH.normalize(prediction)
    return {**cue, "prediction": prediction,
            "selected_ms": selected["pts_ms"] if selected else None,
            "recall": bool(cue["text"] and cue["text"] in normalized),
            "exact": scored["edit_distance"] == 0,
            "edit_distance": scored["edit_distance"], "empty": not normalized,
            "insertions": scored["edit_counts"]["insertions"],
            "candidate_count": candidate_count, **selection}


def score_config(targets, combined_frames, trim_ms, quality):
    output = []
    for cue in targets:
        rows = [row for row in combined_frames[cue["episode"]]
                if "boundary_trims" not in row or trim_ms in row["boundary_trims"]]
        lo, hi = cue_window(cue, trim_ms)
        inner = [row for row in rows if lo <= row["pts_ms"] <= hi]
        fallback = [row for row in rows if cue["start_ms"] - BASE.TOLERANCE_MS <= row["pts_ms"]
                    <= cue["end_ms"] + BASE.TOLERANCE_MS]
        if quality:
            prediction, selected, meta = select_quality(cue, inner, fallback)
        else:
            prediction, selected, meta = select_trimmed_midpoint(cue, inner, fallback)
        output.append(score_prediction(cue, prediction, selected, meta, len(inner)))
    return output


def metric_block(rows):
    return BASE.metric_block(rows)


def summarize(name, rows, trim_ms, background_fp, elapsed_ms):
    pure1 = [row for row in rows if len(row["text"]) == 1]
    band12 = [row for row in rows if 1 <= len(row["text"]) <= 2]
    band34 = [row for row in rows if 3 <= len(row["text"]) <= 4]
    known = [row for row in rows if row["known_miss"]]
    return {
        "name": name, "trim_ms": trim_ms,
        "metrics": {"pure_1_char": metric_block(pure1), "1_2_chars": metric_block(band12),
                    "3_4_chars": metric_block(band34), "all": metric_block(rows)},
        "known_6": {"recalled": sum(row["recall"] for row in known),
                    "exact": sum(row["exact"] for row in known), "total": len(known),
                    "details": [{key: row.get(key) for key in
                                 ("cue_id", "reference", "prediction", "recall", "exact",
                                  "selected_ms", "selection_stage", "evidence")}
                                for row in known]},
        "empty_results": sum(row["empty"] for row in rows),
        "inserted_noise_results": sum(row["insertions"] > 0 for row in rows),
        "background_false_positive_frames": background_fp,
        "single_char_confirmed": sum(row.get("evidence") == "confirmed_box_track"
                                     for row in pure1),
        "single_char_fallbacks": sum("fallback" in row.get("evidence", "") for row in pure1),
        "untrimmed_fallbacks": sum(row.get("selection_stage") == "untrimmed_fallback"
                                   for row in rows),
        "selection_wall_ms": elapsed_ms,
    }


def load_source(source, targets):
    frames = {}
    checksums = {}
    for ep in sorted({cue["episode"] for cue in targets}):
        path = source / "cache" / f"episode_{ep:02d}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        frames[ep] = document["roi_frames"]["0.82"]
        checksums[ep] = {"path": str(path), "sha256": sha256_file(path),
                         "frames": len(frames[ep])}
    return frames, checksums


def augment_cache(args, targets, source_frames, planned, reasons):
    cache_dir = args.out / "augmentation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    augmented = {ep: [] for ep in source_frames}
    if args.rescore:
        for ep in sorted(source_frames):
            path = cache_dir / f"episode_{ep:02d}.json"
            document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"frames": []}
            augmented[ep] = document["frames"]
            for row in augmented[ep]:
                row["boundary_trims"] = sorted({reason["trim_ms"] for reason in
                                                 reasons[ep].get(row["requested_ms"], [])})
            if path.exists():
                document["frames"] = augmented[ep]
                path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        return {ep: [row for row in rows if row["boundary_trims"]]
                for ep, rows in augmented.items()}

    from rapidocr import RapidOCR
    engine = RapidOCR()
    for ep in sorted(source_frames):
        path = cache_dir / f"episode_{ep:02d}.json"
        existing_document = (json.loads(path.read_text(encoding="utf-8"))
                             if path.exists() else {"frames": []})
        stored_rows = existing_document["frames"]
        by_requested = {row["requested_ms"]: row for row in stored_rows}
        for row in stored_rows:
            row["boundary_trims"] = sorted({reason["trim_ms"] for reason in
                                             reasons[ep].get(row["requested_ms"], [])})
        video = args.video_dir / f"{ep:02d}.mp4"
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        seen_pts = {row["pts_ms"] for row in source_frames[ep]}
        for moment in planned.get(ep, []):
            if moment in by_requested:
                continue
            frame, pts_ms = BASE.read_frame(capture, moment)
            if pts_ms in seen_pts:
                continue
            result = BASE.ocr_frame(engine, frame, width, height, 0.82)
            new_row = {"requested_ms": moment, "pts_ms": pts_ms,
                       "schedules": {"boundary_augmentation": True},
                       "boundary_trims": sorted({reason["trim_ms"] for reason in
                                                 reasons[ep][moment]}), **result}
            stored_rows.append(new_row)
            by_requested[moment] = new_row
            seen_pts.add(pts_ms)
        capture.release()
        path.write_text(json.dumps({"episode": ep, "planned_requested_ms": planned.get(ep, []),
                                    "reasons": reasons.get(ep, {}), "frames": stored_rows},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        augmented[ep] = [row for row in stored_rows if row["boundary_trims"]]
        print(f"episode {ep:02d}: {len(augmented[ep])} active boundary frames", flush=True)
    return augmented


def baseline_e_rows(targets, source_frames):
    wrapped = {ep: {0.82: rows} for ep, rows in source_frames.items()}
    return BASE.score_rows(targets, wrapped, "dense", 0.82, True, False)


def key_cases(details):
    wanted = {"35:103200-103959:嗯", "39:1439-2000:你", "50:100480-101239:不是"}
    return {name: [{key: row.get(key) for key in
                    ("cue_id", "reference", "prediction", "selected_ms", "exact", "recall",
                     "selection_stage", "evidence", "evidence_frames", "evidence_span_ms")}
                   for row in rows if row["cue_id"] in wanted]
            for name, rows in details.items()}


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    targets = BASE.load_targets(args.srt_dir)
    source_frames, source_checksums = load_source(args.source, targets)
    planned, reasons = planned_augmentations(targets, source_frames)
    ocr_started = time.perf_counter()
    augmented = augment_cache(args, targets, source_frames, planned, reasons)
    augmentation_wall_seconds = round(time.perf_counter() - ocr_started, 3) if not args.rescore else None
    combined = {ep: sorted(source_frames[ep] + augmented[ep], key=lambda row: row["pts_ms"])
                for ep in source_frames}

    source_eval = json.loads((args.source / "eval_results.json").read_text(encoding="utf-8"))
    previous_result_path = args.out / "eval_results.json"
    previous_result = (json.loads(previous_result_path.read_text(encoding="utf-8"))
                       if args.rescore and previous_result_path.exists() else None)
    if augmentation_wall_seconds is None:
        augmentation_wall_seconds = previous_result.get("augmentation", {}).get(
            "wall_seconds_including_decode_and_engine") if previous_result else None
    base_summary = next(row for row in source_eval["configs"]
                        if row["name"] == "E_dense250_roi082_yfilter_midpoint")
    background_fp = base_summary["background_false_positive_frames"]
    details = {"E_before": baseline_e_rows(targets, source_frames)}
    summaries = [{**base_summary, "name": "E_before", "trim_ms": 0,
                  "selection_wall_ms": 0}]
    for trim in TRIMS_MS:
        for quality, suffix in ((False, "midpoint"), (True, "quality_box_stability_single_confirm")):
            name = f"trim{trim}_{suffix}"
            started = time.perf_counter()
            rows = score_config(targets, combined, trim, quality)
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            details[name] = rows
            summaries.append(summarize(name, rows, trim, background_fp, elapsed))

    # The 100 ms variant is the deployment candidate unless it fails the frozen
    # recall/background guardrail; 150 ms remains an intentionally stricter ablation.
    candidate_name = "trim100_quality_box_stability_single_confirm"
    candidate = next(row for row in summaries if row["name"] == candidate_name)
    passes = (candidate["known_6"]["recalled"] == 6 and
              candidate["metrics"]["pure_1_char"]["recall"] >=
              base_summary["metrics"]["pure_1_char"]["recall"] and
              candidate["background_false_positive_frames"] <= background_fp)
    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "reference": "human-verified SRT only", "asr_text_used_as_truth": False,
        "source_cache": source_checksums,
        "boundary_rules": {"trims_ms": list(TRIMS_MS), "minimum_inner_candidates": MIN_CANDIDATES,
                           "anchor_match_ms": ANCHOR_MATCH_MS,
                           "decode_pts_guard_ms": DECODE_PTS_GUARD_MS,
                           "single_char_confirmation_gap_ms": SINGLE_CONFIRM_GAP_MS,
                           "single_frame_fallback": True},
        "augmentation": {"planned_requested_frames": sum(len(v) for v in planned.values()),
                         "actual_new_frames": sum(len(v) for v in augmented.values()),
                         "actual_ocr_seconds": round(sum(row["ocr_ms"] for rows in augmented.values()
                                                         for row in rows) / 1000, 3),
                         "wall_seconds_including_decode_and_engine": augmentation_wall_seconds,
                         "by_trim": {str(trim): {
                             "planned_requested_frames": sum(
                                 any(reason["trim_ms"] == trim for reason in reasons[ep][moment])
                                 for ep in reasons for moment in reasons[ep]),
                             "actual_new_frames": sum(trim in row.get("boundary_trims", [])
                                                      for rows in augmented.values() for row in rows),
                             "actual_ocr_seconds": round(sum(
                                 row["ocr_ms"] for rows in augmented.values() for row in rows
                                 if trim in row.get("boundary_trims", [])) / 1000, 3),
                         } for trim in TRIMS_MS},
                         "per_episode": {str(ep): {"planned": len(planned.get(ep, [])),
                                                   "actual": len(augmented[ep])}
                                         for ep in sorted(augmented)}},
        "configs": summaries,
        "key_cases": key_cases(details),
        "decision": {"recommended": candidate_name if passes else "E_before",
                     "passes_default_guardrail": passes,
                     "guardrail": "6/6 historical recall, pure-1 recall >= E, background FP <= E",
                     "note": "100 ms preserves more usable evidence than 150 ms; stable exact box tracks remove unstable same-band clutter; a visible single-frame fallback prevents short-cue loss."},
    }
    (args.out / "eval_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "cue_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configs": summaries, "augmentation": result["augmentation"],
                      "key_cases": result["key_cases"], "decision": result["decision"]},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--srt-dir", type=Path, default=BASE.DEFAULT_SRT_DIR)
    parser.add_argument("--video-dir", type=Path, default=BASE.DEFAULT_VIDEO_DIR)
    parser.add_argument("--rescore", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
