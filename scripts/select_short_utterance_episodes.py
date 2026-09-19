"""Select short-utterance-dense episodes and report their coordinate provenance.

Reads a directory of human-corrected Chinese SRT files plus the matching video
directory, counts short cues (1-2 and 3-4 characters after tag/punctuation
removal) and target seed entries ("你 爸 妈 嗯 哼"), then selects a small,
deterministic benchmark set.  It never rewrites the SRT, video, or any existing
coordinate file; outputs are new JSON documents under --out.

Selection rule (see data/yaoyao_short_utterance_selection/selection.json):
  stage 1: top 4 by (target_seed_entries, short_1_2, short_3_4, episode)
  stage 2: next 1 by (short_1_2, short_3_4, episode)
  stage 3: next 1 by (short_3_4, short_1_2, episode)
"""

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

TIME = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}")
TAGS = re.compile(r"<[^>]+>")
TARGET_SEED = tuple("你爸妈嗯哼")


def read_text(path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def cues(path):
    """Yield (display_text, normalized_text) for each SRT cue; no file rewrites."""
    content = read_text(path)
    for block in re.split(r"\r?\n\s*\r?\n", content):
        lines = block.strip().splitlines()
        at = next((i for i, line in enumerate(lines) if TIME.search(line)), None)
        if at is None:
            continue
        display = "".join(TAGS.sub("", line).strip() for line in lines[at + 1:])
        normalized = "".join(ch for ch in display if ch.isalnum())
        if normalized:
            yield display, normalized


def episode_number(path):
    return int(path.stem) if re.fullmatch(r"\d+", path.stem) else None


def resolve_duplicates(srt_dir):
    """One logical episode per number; newest mtime wins, filename as tiebreak."""
    by_number = {}
    for srt in sorted(srt_dir.glob("*.srt")):
        number = episode_number(srt)
        if number is None:
            continue
        by_number.setdefault(number, []).append(srt)
    resolved, notes = {}, {}
    for number, candidates in by_number.items():
        chosen = max(candidates, key=lambda p: (p.stat().st_mtime, p.name))
        resolved[number] = chosen
        if len(candidates) > 1:
            notes[number] = {
                "candidates": [p.name for p in sorted(candidates, key=lambda p: p.name)],
                "selected": chosen.name,
                "mtime": {p.name: p.stat().st_mtime for p in candidates},
            }
    return resolved, notes


def stats(path, target_seed=TARGET_SEED):
    rows = list(cues(path))
    lengths = [len(text) for _, text in rows]
    short_12 = sum(1 <= n <= 2 for n in lengths)
    short_34 = sum(3 <= n <= 4 for n in lengths)
    seed = sum(1 <= n <= 2 and any(ch in text for ch in target_seed)
               for (_, text), n in zip(rows, lengths))
    contains = {ch: sum(ch in text for _, text in rows) for ch in target_seed}
    occurrences = {ch: sum(text.count(ch) for _, text in rows) for ch in target_seed}
    target_cues = sum(any(ch in text for ch in target_seed) for _, text in rows)
    return {
        "total_cues": len(rows),
        "short_1_2_entries": short_12,
        "short_3_4_entries": short_34,
        "target_seed_entries": seed,
        "target_seed_share_of_1_2": round(seed / short_12, 6) if short_12 else 0,
        "target_cue_entries": target_cues,
        "target_cue_share": round(target_cues / len(rows), 6) if rows else 0,
        "target_total_occurrences": sum(occurrences.values()),
        "target_contains_counts": contains,
        "target_occurrence_counts": occurrences,
    }


def select(rows, count=6):
    remaining = list(rows)
    picks = []

    def take(key):
        if not remaining:
            return None
        chosen = max(remaining, key=key)
        remaining.remove(chosen)
        return chosen

    for _ in range(min(4, count)):
        chosen = take(lambda r: (r["target_seed_entries"], r["short_1_2_entries"],
                                 r["short_3_4_entries"], -r["episode"]))
        if chosen:
            chosen["stage"] = 1
            picks.append(chosen)
    if len(picks) < count:
        chosen = take(lambda r: (r["short_1_2_entries"], r["short_3_4_entries"], -r["episode"]))
        if chosen:
            chosen["stage"] = 2
            picks.append(chosen)
    if len(picks) < count:
        chosen = take(lambda r: (r["short_3_4_entries"], r["short_1_2_entries"], -r["episode"]))
        if chosen:
            chosen["stage"] = 3
            picks.append(chosen)
    return picks


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-count", type=int, default=6)
    parser.add_argument("--target-seed", default="".join(TARGET_SEED))
    args = parser.parse_args()
    target_seed = tuple(dict.fromkeys(args.target_seed))
    if not target_seed:
        raise SystemExit("--target-seed must not be empty")

    resolved, duplicate_notes = resolve_duplicates(args.srt_dir)
    videos = {}
    for video in args.video_dir.glob("*.mp4"):
        number = episode_number(video)
        if number is not None:
            videos.setdefault(number, video)

    rows = []
    for number, srt in sorted(resolved.items()):
        row = {"episode": number, "srt": str(srt), **stats(srt, target_seed)}
        row["video"] = str(videos[number]) if number in videos else None
        rows.append(row)
    eligible = [row for row in rows if row["video"]]
    picks = select(eligible, args.benchmark_count)
    aggregate_occurrences = {
        char: sum(row["target_occurrence_counts"][char] for row in rows)
        for char in target_seed
    }
    aggregate_contains = {
        char: sum(row["target_contains_counts"][char] for row in rows)
        for char in target_seed
    }

    before = {str(srt): sha256(srt) for srt in sorted(resolved.values())}
    content_manifest = hashlib.sha256(
        "".join(f"{k} {v}\n" for k, v in sorted(before.items())).encode()).hexdigest()

    result = {
        "schema_version": "1.0",
        "status": "待人工复核",
        "generated_date": date.today().isoformat(),
        "mode": "audit-selection-only",
        "target_seed_characters": list(target_seed),
        "aggregate": {
            "total_cues": sum(row["total_cues"] for row in rows),
            "target_seed_entries": sum(row["target_seed_entries"] for row in rows),
            "target_cue_entries": sum(row["target_cue_entries"] for row in rows),
            "target_total_occurrences": sum(row["target_total_occurrences"] for row in rows),
            "target_contains_counts": aggregate_contains,
            "target_occurrence_counts": aggregate_occurrences,
            "short_1_2_entries": sum(row["short_1_2_entries"] for row in rows),
            "short_3_4_entries": sum(row["short_3_4_entries"] for row in rows),
        },
        "per_episode_statistics": rows,
        "selection_rule": {
            "benchmark_episode_count": args.benchmark_count,
            "stage_1": "按 target_seed_entries、short_1_2_entries、short_3_4_entries 降序，集号升序取前 4 集。",
            "stage_2": "从剩余集按 short_1_2_entries、short_3_4_entries 降序，集号升序取 1 集。",
            "stage_3": "从剩余集按 short_3_4_entries、short_1_2_entries 降序，集号升序取 1 集。",
        },
        "selected_episodes": picks,
        "duplicate_episode_resolution": duplicate_notes,
        "source_integrity": {
            "srt_sha256_before": before,
            "srt_content_manifest_sha256_before": content_manifest,
        },
        "review_notice": "待人工复核：本文件用于 OCR benchmark 选样，不是对字幕内容或 ROI 的最终批准。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected episodes: {[row['episode'] for row in picks]}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
