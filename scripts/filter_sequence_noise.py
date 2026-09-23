"""Remove brief patterned-background OCR artifacts from sequence evidence.

This operates only on OCR evidence and never reads reference subtitles.
Rejected detections remain in each JSON file for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.asr_adapter import save_evidence  # noqa: E402
from app.subtitle_sequence import is_background_noise  # noqa: E402
from scripts.ocr_asr_batch import write_srt  # noqa: E402


def clean_root(root: Path) -> dict:
    episodes, removed, kept = 0, 0, 0
    for path in sorted((root / "evidence").glob("episode_*_ocr.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("guidance", {}).get("type") != "visual_sequence_asr_timing":
            continue
        if data.get("noise_filter", {}).get("version") == 1:
            episodes += 1
            kept += len(data["detections"])
            removed += len(data.get("rejected_noise", []))
            continue
        accepted, rejected = [], []
        for row in data["detections"]:
            text = row.get("raw_ocr_text", row["ocr_text"])
            if is_background_noise(text, float(row["confidence"]),
                                   int(row.get("observations", 1))):
                rejected.append({**row, "rejection_reason": "brief_pattern_noise"})
            else:
                accepted.append(row)
        data["detections"] = accepted
        data["rejected_noise"] = rejected
        data["noise_filter"] = {"version": 1, "method": "ocr_character_mix_and_persistence"}
        data["summary"]["detected_segments"] = len(accepted)
        data["summary"]["noise_rejected"] = len(rejected)
        episode = int(path.stem.split("_")[1])
        save_evidence(path, data)
        write_srt(root / "subtitles" / f"{episode:02d}.srt", data["segments"], accepted)
        episodes += 1
        removed += len(rejected)
        kept += len(accepted)
    report = {"episodes": episodes, "kept": kept, "noise_rejected": removed,
              "reference_subtitles_used": False}
    save_evidence(root / "noise_filter_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="清理视觉序列 OCR 中的短暂背景花纹乱码")
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        print(json.dumps({"root": str(root), **clean_root(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
