"""Reproducible Windows benchmark for burned-in Chinese drama subtitles.

Examples:
  python scripts/benchmark_ocr_win.py prepare --video episode.mp4 --srt episode.srt --out data/ocr_benchmark --limit 12
  python scripts/benchmark_ocr_win.py run --frames data/ocr_benchmark --engine rapidocr
  python scripts/benchmark_ocr_win.py run --frames data/ocr_benchmark --engine paddleocr

SRT is a timing/text reference, not a manually verified frame annotation.
"""

import argparse
import base64
import json
import re
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path


TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def milliseconds(parts):
    hour, minute, second, milli = map(int, parts)
    return ((hour * 60 + minute) * 60 + second) * 1000 + milli


def read_srt(path):
    content = path.read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", content):
        lines = block.strip().splitlines()
        match_at = next((i for i, line in enumerate(lines) if TIME.search(line)), None)
        if match_at is None:
            continue
        match = TIME.search(lines[match_at])
        start, end = milliseconds(match.groups()[:4]), milliseconds(match.groups()[4:])
        reference = "".join(re.sub(r"<[^>]+>", "", line).strip() for line in lines[match_at + 1:])
        if end > start and reference:
            cues.append({"start_ms": start, "end_ms": end, "reference": reference})
    return cues


def frame_bytes(video, at_ms):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{at_ms / 1000:.3f}",
               "-i", str(video), "-frames:v", "1", "-y", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    return subprocess.run(command, capture_output=True, check=True, timeout=60).stdout


def locate(args):
    """Use a local VLM on a few full frames, once per video, to find the subtitle band."""
    from io import BytesIO
    from PIL import Image

    cues = read_srt(args.srt)
    if not cues:
        raise SystemExit("No valid SRT cues found")
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected = [cues[(i * len(cues) + len(cues) // 2) // args.samples]
                for i in range(args.samples)]
    boxes = []
    observations = []
    for cue in selected:
        at_ms = (cue["start_ms"] + cue["end_ms"]) // 2
        with Image.open(BytesIO(frame_bytes(args.video, at_ms))) as image:
            image.thumbnail((768, 1280))
            image_width, image_height = image.size
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=88)
        payload = {"model": args.model, "stream": False,
                   "options": {"temperature": 0}, "messages": [{"role": "user",
                   "content": "Locate only the horizontal dialogue subtitle in this drama frame. "
                              "Ignore vertical show titles, logos, credits, and clothing patterns. "
                              "Return its tight bounding box as bbox=[x0,y0,x1,y1] in integer "
                              "coordinates from 0 to 1000 relative to the full image. "
                              "If no dialogue subtitle is visible, set has_subtitle=false. "
                              "Reply with JSON only.",
                   "images": [base64.b64encode(buffer.getvalue()).decode("ascii")]}]}
        box, raw_content, done_reason, output = None, "", "", {}
        for _ in range(2):
            request = urllib.request.Request(args.url, data=json.dumps(payload).encode("utf-8"),
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    output = json.load(response)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                output = {"error": str(exc)}
                continue
            raw_content = output.get("message", {}).get("content", "")
            done_reason = output.get("done_reason", "")
            try:
                match = re.search(r"\{[^{}]*\}", raw_content, re.DOTALL)
                box = json.loads(match.group(0) if match else raw_content)
                if isinstance(box, dict) and isinstance(box.get("bbox"), list) and len(box["bbox"]) == 4:
                    box = {"has_subtitle": box.get("has_subtitle", True),
                           **dict(zip(("x0", "y0", "x1", "y1"), box["bbox"]))}
                break
            except json.JSONDecodeError:
                continue
        valid = (isinstance(box, dict) and box.get("has_subtitle")
                 and all(isinstance(box.get(key), int) for key in ("x0", "y0", "x1", "y1"))
                 and 0 <= box["x0"] < box["x1"] <= 1000
                 and 0 <= box["y0"] < box["y1"] <= 1000
                 and box["x1"] - box["x0"] >= 20 and box["y1"] - box["y0"] >= 8)
        observations.append({"at_ms": at_ms, "reference": cue["reference"],
                             "image_size": [image_width, image_height], "box_pixels": box,
                             "raw_content": raw_content, "done_reason": done_reason,
                             "response_meta": {key: value for key, value in output.items()
                                               if key not in ("message",)},
                             "message": output.get("message"),
                             "accepted": bool(valid)})
        if valid:
            boxes.append({"x0": box["x0"] / 1000, "y0": box["y0"] / 1000,
                          "x1": box["x1"] / 1000, "y1": box["y1"] / 1000})
        print(f"{at_ms} ms: {box} accepted={valid} reason={done_reason} raw={raw_content[:300]!r} "
              f"error={output.get('error')!r}", flush=True)
    if not boxes:
        debug_path = args.out.with_suffix(".debug.json")
        debug_path.write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("VLM found no valid subtitle boxes")
    # Merge the observed positions, then add generous padding for moving text.
    roi = [max(0, min(box["x0"] for box in boxes) - 0.07),
           max(0, min(box["y0"] for box in boxes) - 0.04),
           min(1, max(box["x1"] for box in boxes) + 0.07),
           min(1, max(box["y1"] for box in boxes) + 0.04)]
    # A short sample line must not narrow the crop for later, longer dialogue.
    if roi[2] - roi[0] < 0.75:
        center = (roi[0] + roi[2]) / 2
        roi[0], roi[2] = max(0, center - 0.375), min(1, center + 0.375)
    result = {"model": args.model, "video": str(args.video), "roi": roi,
              "observations": observations}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Suggested normalized ROI: {roi}")


def prepare(args):
    from PIL import Image

    cues = read_srt(args.srt)
    if not cues:
        raise SystemExit("No valid SRT cues found")
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    roi = json.loads(args.roi_file.read_text(encoding="utf-8"))["roi"] if args.roi_file else [
        args.x0, args.y0, args.x1, args.y1]
    if not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise SystemExit("Invalid normalized ROI")
    selected = [cues[(i * len(cues) + len(cues) // 2) // args.limit] for i in range(args.limit)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, cue in enumerate(selected, 1):
        at_ms = (cue["start_ms"] + cue["end_ms"]) // 2
        image_path = args.out / f"frame_{i:03d}.png"
        from io import BytesIO
        with Image.open(BytesIO(frame_bytes(args.video, at_ms))) as image:
            width, height = image.size
            region = image.crop((int(width * roi[0]), int(height * roi[1]),
                                 int(width * roi[2]), int(height * roi[3])))
            region.save(image_path)
        rows.append({"image": image_path.name, "video": str(args.video), "at_ms": at_ms,
                     "reference": cue["reference"]})
    manifest = {"srt": str(args.srt), "roi": roi, "roi_source": str(args.roi_file) if args.roi_file else "manual",
                "samples": rows,
                "note": "SRT text and timing are unverified against extracted frames"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(rows)} subtitle crops to {args.out}")


def normalize(text):
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def make_engine(name):
    if name == "rapidocr":
        from rapidocr import RapidOCR
        engine = RapidOCR()

        def infer(path):
            result = engine(str(path))
            return list(result.txts or [])
        return infer
    if name == "paddleocr":
        from paddleocr import PaddleOCR
        engine = PaddleOCR(ocr_version="PP-OCRv5", lang="ch", device="cpu",
                           enable_mkldnn=False,
                           use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False)

        def infer(path):
            output = engine.predict(str(path))
            texts = []
            for item in output:
                data = item.json
                texts.extend(data.get("res", data).get("rec_texts", []))
            return texts
        return infer
    raise ValueError(name)


def run(args):
    manifest = json.loads((args.frames / "manifest.json").read_text(encoding="utf-8"))
    start = time.perf_counter()
    infer = make_engine(args.engine)
    load_seconds = time.perf_counter() - start
    results = []
    for row in manifest["samples"]:
        start = time.perf_counter()
        texts = infer(args.frames / row["image"])
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        prediction = "".join(texts)
        reference = normalize(row["reference"])
        detected = normalize(prediction)
        distance = edit_distance(reference, detected)
        result = {**row, "detected_lines": texts, "prediction": prediction,
                  "edit_distance": distance, "reference_chars": len(reference),
                  "latency_ms": elapsed_ms}
        results.append(result)
        print(f"{row['image']}: {row['reference']} => {prediction} ({elapsed_ms} ms)", flush=True)
    total_chars = sum(row["reference_chars"] for row in results)
    total_edits = sum(row["edit_distance"] for row in results)
    latencies = sorted(row["latency_ms"] for row in results[1:])
    summary = {"engine": args.engine, "samples": len(results), "load_seconds": round(load_seconds, 2),
               "srt_proxy_cer": round(total_edits / total_chars, 3) if total_chars else None,
               "exact_matches": sum(row["edit_distance"] == 0 for row in results),
               "median_warm_latency_ms": latencies[len(latencies) // 2] if latencies else None,
               "note": manifest["note"]}
    report = {"summary": summary, "results": results}
    (args.frames / f"{args.engine}_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--video", type=Path, required=True)
    prep.add_argument("--srt", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--limit", type=int, default=16)
    prep.add_argument("--roi-file", type=Path)
    prep.add_argument("--x0", type=float, default=0.1)
    prep.add_argument("--y0", type=float, default=0.44)
    prep.add_argument("--x1", type=float, default=0.85)
    prep.add_argument("--y1", type=float, default=0.78)
    runner = commands.add_parser("run")
    runner.add_argument("--frames", type=Path, required=True)
    runner.add_argument("--engine", choices=["rapidocr", "paddleocr"], required=True)
    locator = commands.add_parser("locate")
    locator.add_argument("--video", type=Path, required=True)
    locator.add_argument("--srt", type=Path, required=True)
    locator.add_argument("--out", type=Path, required=True)
    locator.add_argument("--samples", type=int, default=3)
    locator.add_argument("--model", default="qwen3-vl:4b-instruct")
    locator.add_argument("--url", default="http://localhost:11434/api/chat")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        locate(args)


if __name__ == "__main__":
    main()
