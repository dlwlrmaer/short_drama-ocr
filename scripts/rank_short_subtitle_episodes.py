"""Rank local drama episodes by short Chinese subtitle cues (1-4 characters)."""

import argparse
import json
import re
from pathlib import Path

TIME = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")


def cue_texts(path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            content = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    for block in re.split(r"\r?\n\s*\r?\n", content):
        lines = block.strip().splitlines()
        index = next((i for i, line in enumerate(lines) if TIME.search(line)), None)
        if index is not None:
            text = "".join(re.sub(r"<[^>]+>", "", line).strip() for line in lines[index + 1:])
            if text:
                yield text


def ignored(path):
    return any(re.search(r"(^|[^a-z])en($|[^a-z])|英文|english|review|翻译|backup|备份", part, re.I)
               for part in path.parts)


def source_priority(path):
    parts = [part.lower() for part in path.parts]
    if "中文" in parts:
        return 3
    if any("字幕" in part or "srt" in part for part in parts):
        return 2
    return 1


def episode_number(path):
    return int(path.stem) if re.fullmatch(r"\d+", path.stem) else None


def rank(root):
    rows = []
    for project in root.iterdir():
        if not project.is_dir() or project.name.lower().endswith("agent"):
            continue
        videos = {}
        for video in project.rglob("*.mp4"):
            number = episode_number(video)
            if number is not None:
                videos.setdefault(number, []).append(video)
        best = {}
        for srt in project.rglob("*.srt"):
            number = episode_number(srt)
            if number is None or number not in videos or ignored(srt.relative_to(project)):
                continue
            texts = list(cue_texts(srt))
            lengths = [sum(ch.isalnum() for ch in text) for text in texts]
            counts = {"short_1_2": sum(1 <= n <= 2 for n in lengths),
                      "short_3_4": sum(3 <= n <= 4 for n in lengths),
                      "short_1_4": sum(1 <= n <= 4 for n in lengths),
                      "total": len(lengths)}
            video = max(videos[number], key=lambda item: item.stat().st_size)
            row = {"project": project.name, "episode": number,
                   "srt": str(srt), "video": str(video), **counts,
                   "short_ratio": round(counts["short_1_4"] / len(lengths), 3) if lengths else 0}
            if number not in best or (source_priority(srt.relative_to(project)), row["short_1_4"]) > (
                    source_priority(Path(best[number]["srt"]).relative_to(project)),
                    best[number]["short_1_4"]):
                best[number] = row
        rows.extend(best.values())
    return sorted(rows, key=lambda row: (row["short_1_4"], row["short_1_2"],
                                         row["short_ratio"]), reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    rows = rank(args.root)
    for row in rows[:args.top]:
        print(f"{row['project']} #{row['episode']}: <=4={row['short_1_4']} "
              f"(1-2={row['short_1_2']}), total={row['total']}, ratio={row['short_ratio']:.1%}")
        print(f"  {row['video']}")
        print(f"  {row['srt']}")
    print(f"Ranked {len(rows)} episodes with matching video and Chinese SRT")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
