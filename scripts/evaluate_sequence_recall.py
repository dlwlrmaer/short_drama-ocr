"""Compare visual subtitle evidence against an SRT used only after inference.

The reference SRT never enters OCR or ASR generation. This tool reports cue
recall by representative frame time, and separately checks rough text agreement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.subtitle_sequence import normalize_ocr_text  # noqa: E402
from app.video_pipeline import probe_duration_ms  # noqa: E402

TIMESTAMP = re.compile(
    r"(?P<start>\d\d:\d\d:\d\d,\d{3})\s+-->\s+"
    r"(?P<end>\d\d:\d\d:\d\d,\d{3})"
)


def milliseconds(value: str) -> int:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)


def read_srt(path: Path) -> list[dict]:
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig")):
        lines = block.strip().splitlines()
        match = next((TIMESTAMP.fullmatch(line.strip()) for line in lines if TIMESTAMP.fullmatch(line.strip())), None)
        if match is None:
            continue
        line_index = next(index for index, line in enumerate(lines) if TIMESTAMP.fullmatch(line.strip()))
        cues.append({"start_ms": milliseconds(match["start"]),
                     "end_ms": milliseconds(match["end"]),
                     "text": "".join(lines[line_index + 1:]).strip()})
    return cues


def reference_paths(directory: Path) -> dict[int, Path]:
    """Accept source names such as ``58 .srt`` without renaming user files."""
    return {int(path.stem.strip()): path for path in directory.glob("*.srt")
            if path.stem.strip().isdecimal()}


def simplify(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return "".join(char for char in normalize_ocr_text(value)
                   if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def compare(reference: list[dict], detections: list[dict], tolerance_ms: int = 100) -> dict:
    """Match each OCR frame to at most one cue, preferring text agreement."""
    candidates = []
    for cue_index, cue in enumerate(reference):
        for detection_index, detection in enumerate(detections):
            point = int(detection["frame_pts_ms"])
            if cue["start_ms"] - tolerance_ms <= point <= cue["end_ms"] + tolerance_ms:
                score = SequenceMatcher(None, simplify(cue["text"]),
                                        simplify(detection["ocr_text"]), autojunk=False).ratio()
                candidates.append((score, cue_index, detection_index))
    matched_cues, matched_detections = {}, set()
    for score, cue_index, detection_index in sorted(candidates, reverse=True):
        if cue_index not in matched_cues and detection_index not in matched_detections:
            matched_cues[cue_index] = score
            matched_detections.add(detection_index)
    rough_text = sum(score >= 0.5 for score in matched_cues.values())
    nearby_candidates: dict[int, list[tuple[float, float, int]]] = {}
    for cue_index, cue in enumerate(reference):
        midpoint = (cue["start_ms"] + cue["end_ms"]) / 2
        for detection_index, detection in enumerate(detections):
            point = int(detection["frame_pts_ms"])
            if cue["start_ms"] - 2500 <= point <= cue["end_ms"] + 2500:
                score = SequenceMatcher(None, simplify(cue["text"]),
                                        simplify(detection["ocr_text"]), autojunk=False).ratio()
                if score >= 0.5:
                    nearby_candidates.setdefault(cue_index, []).append(
                        (score, -abs(point - midpoint), detection_index))
    nearby_assigned: dict[int, int] = {}

    def assign(cue_index: int, visited: set[int]) -> bool:
        for _, _, detection_index in sorted(nearby_candidates.get(cue_index, []), reverse=True):
            if detection_index in visited:
                continue
            visited.add(detection_index)
            previous = nearby_assigned.get(detection_index)
            if previous is None or assign(previous, visited):
                nearby_assigned[detection_index] = cue_index
                return True
        return False

    for cue_index in range(len(reference)):
        assign(cue_index, set())
    nearby_cues = set(nearby_assigned.values())
    return {"reference_cues": len(reference), "detections": len(detections),
            "visual_cues_found": len(matched_cues), "rough_text_matches": rough_text,
            "nearby_text_matches": len(nearby_cues),
            "visual_recall": round(len(matched_cues) / len(reference), 4) if reference else None,
            "rough_text_recall": round(rough_text / len(reference), 4) if reference else None,
            "nearby_text_recall": round(len(nearby_cues) / len(reference), 4) if reference else None,
            "missed": [{"start_ms": cue["start_ms"], "text": cue["text"]}
                       for index, cue in enumerate(reference) if index not in matched_cues],
            "unmatched_nearby": [{"start_ms": cue["start_ms"], "text": cue["text"]}
                                 for index, cue in enumerate(reference) if index not in nearby_cues]}


def main() -> None:
    parser = argparse.ArgumentParser(description="对照 SRT 仅用于事后评估字幕召回率")
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("baseline_evidence_dir", type=Path)
    parser.add_argument("improved_evidence_dir", type=Path)
    parser.add_argument("--episodes", nargs="*", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = reference_paths(args.reference_dir)
    numbers = args.episodes or sorted(sources)
    episodes = []
    for number in numbers:
        source = sources.get(number)
        original = args.baseline_evidence_dir / f"episode_{number:04d}_ocr.json"
        improved = args.improved_evidence_dir / f"episode_{number:04d}_ocr.json"
        if not (source and original.is_file() and improved.is_file()):
            continue
        baseline_data = json.loads(original.read_text(encoding="utf-8"))
        improved_data = json.loads(improved.read_text(encoding="utf-8"))
        duration = probe_duration_ms(Path(improved_data["source_file"]))
        reference = [cue for cue in read_srt(source) if cue["end_ms"] <= duration + 100]
        episodes.append({"episode": number, "baseline": compare(reference, baseline_data["detections"]),
                         "improved": compare(reference, improved_data["detections"])})
    totals = {}
    for kind in ("baseline", "improved"):
        totals[kind] = {key: sum(row[kind][key] for row in episodes)
                        for key in ("reference_cues", "detections", "visual_cues_found",
                                    "rough_text_matches", "nearby_text_matches")}
        count = totals[kind]["reference_cues"]
        totals[kind]["visual_recall"] = round(totals[kind]["visual_cues_found"] / count, 4) if count else None
        totals[kind]["rough_text_recall"] = round(totals[kind]["rough_text_matches"] / count, 4) if count else None
        totals[kind]["nearby_text_recall"] = round(totals[kind]["nearby_text_matches"] / count, 4) if count else None
    result = {"reference_used_for_inference": False, "episodes": episodes, "totals": totals}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episode_count": len(episodes), "totals": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
