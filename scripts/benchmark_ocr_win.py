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
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path


TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")

# Dependency-free fallback for the script variants seen in the benchmark.  This is
# deliberately narrow: it reports known Simplified/Traditional shape differences,
# not arbitrary semantic equivalence.
SCRIPT_VARIANTS = str.maketrans({
    "別": "别", "媽": "妈", "爸": "爸", "妳": "你", "爾": "尔", "嗎": "吗",
    "們": "们", "這": "这", "個": "个", "來": "来", "說": "说", "沒": "没",
    "對": "对", "裡": "里", "後": "后", "讓": "让", "為": "为", "還": "还",
    "滾": "滚", "況": "况",
})


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
    if args.min_chars:
        cues = [cue for cue in cues if len(normalize(cue["reference"])) >= args.min_chars]
    if args.max_chars:
        cues = [cue for cue in cues if 1 <= len(normalize(cue["reference"])) <= args.max_chars]
    if args.include_term:
        cues = [cue for cue in cues if any(term in normalize(cue["reference"])
                                           for term in args.include_term)]
    if not cues:
        raise SystemExit("No valid SRT cues found")
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    roi = read_roi_file(args.roi_file, args.canvas) if args.roi_file else [
        args.x0, args.y0, args.x1, args.y1]
    if not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise SystemExit("Invalid normalized ROI")
    sample_count = min(args.limit, len(cues))
    selected = [cues[(i * len(cues) + len(cues) // 2) // sample_count]
                for i in range(sample_count)]
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
    manifest = {"srt": str(args.srt), "roi": roi,
                "roi_source": str(args.roi_file) if args.roi_file else "manual",
                "roi_canvas": list(args.canvas) if args.canvas else None,
                "samples": rows,
                "note": "SRT text and timing are unverified against extracted frames"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(rows)} subtitle crops to {args.out}")


def _find_roi(value):
    """Find an existing ROI in a benchmark/workspace JSON object.

    Returns (roi, coordinate_space) where coordinate_space is "normalized" for
    [x0,y0,x1,y1] / left/top/right/bottom and "pixels" for x/y/width/height.
    """
    if not isinstance(value, dict):
        return None
    for key in ("reference_roi", "roi"):
        candidate = value.get(key)
        if isinstance(candidate, list) and len(candidate) == 4:
            return candidate, "normalized"
        if isinstance(candidate, dict):
            for names in (("x0", "y0", "x1", "y1"),
                          ("left", "top", "right", "bottom")):
                if all(name in candidate for name in names):
                    return [candidate[name] for name in names], "normalized"
            for x_name, y_name, w_name in (("x", "y", "width"), ("x", "y", "w")):
                if all(name in candidate for name in (x_name, y_name, w_name)):
                    x, y = candidate[x_name], candidate[y_name]
                    width = candidate[w_name]
                    height = candidate.get("height", candidate.get("h"))
                    if height is not None:
                        return [x, y, x + width, y + height], "pixels"
    for child in value.values():
        if isinstance(child, dict):
            found = _find_roi(child)
            if found is not None:
                return found
    return None


def read_roi_file(path, canvas=None):
    """Read an ROI without estimating one; supports manifests and OCR workspaces.

    Pixel-space ROIs (x/y/width/height, as written by short-drama
    ocr_workspace.json files) are converted to normalized [x0,y0,x1,y1] when a
    canvas size is supplied as (width, height).  No ROI is ever invented here.
    """
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    found = _find_roi(data)
    if found is None:
        raise SystemExit(f"No roi/reference_roi found in {path}")
    roi, space = found
    try:
        roi = [float(value) for value in roi]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Non-numeric ROI in {path}") from exc
    if space == "pixels":
        if not canvas:
            raise SystemExit(f"Pixel ROI in {path} needs --canvas WIDTHxHEIGHT to normalize")
        width, height = canvas
        if width <= 0 or height <= 0:
            raise SystemExit("Canvas size must be positive")
        roi = [roi[0] / width, roi[1] / height, roi[2] / width, roi[3] / height]
    return roi


def merge(args):
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.sources:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        samples = manifest["samples"]
        if args.per_source and args.per_source < len(samples):
            samples = [samples[(i * len(samples) + len(samples) // 2) // args.per_source]
                       for i in range(args.per_source)]
        for row in samples:
            image_name = f"{source.name}_{row['image']}"
            shutil.copy2(source / row["image"], args.out / image_name)
            rows.append({**row, "image": image_name, "source": source.name})
    result = {"sources": [str(source) for source in args.sources], "samples": rows,
              "note": "SRT text and timing are unverified against extracted frames"}
    (args.out / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {len(rows)} frames from {len(args.sources)} episodes into {args.out}")


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


def edit_counts(reference, detected):
    """Return Levenshtein insert/delete/substitute counts with stable tie breaks."""
    rows, columns = len(reference) + 1, len(detected) + 1
    table = [[None] * columns for _ in range(rows)]
    table[0][0] = (0, 0, 0, 0)
    for i in range(1, rows):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, columns):
        table[0][j] = (j, j, 0, 0)
    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == detected[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            candidates = []
            cost, ins, delete, sub = table[i][j - 1]
            candidates.append((cost + 1, ins + 1, delete, sub))
            cost, ins, delete, sub = table[i - 1][j]
            candidates.append((cost + 1, ins, delete + 1, sub))
            cost, ins, delete, sub = table[i - 1][j - 1]
            candidates.append((cost + 1, ins, delete, sub + 1))
            table[i][j] = min(candidates)
    _, insertions, deletions, substitutions = table[-1][-1]
    return {"insertions": insertions, "deletions": deletions,
            "substitutions": substitutions}


def length_band(length):
    if 1 <= length <= 2:
        return "1-2"
    if 3 <= length <= 4:
        return "3-4"
    return "other"


def summarize_results(engine, load_seconds, note, results):
    def metrics(rows):
        total_chars = sum(row["reference_chars"] for row in rows)
        total_edits = sum(row["edit_distance"] for row in rows)
        return {
            "samples": len(rows),
            "exact_matches": sum(row["edit_distance"] == 0 for row in rows),
            "exact_match_rate": round(sum(row["edit_distance"] == 0 for row in rows) / len(rows), 3)
            if rows else None,
            "srt_proxy_cer": round(total_edits / total_chars, 3) if total_chars else None,
            "empty_results": sum(not row["detected"] for row in rows),
            "script_variant_matches": sum(row["difference_kind"] == "script_variant" for row in rows),
            "inserted_noise_results": sum(row["difference_kind"] == "contains_inserted_noise"
                                          for row in rows),
            "inserted_noise_chars": sum(row["edit_counts"]["insertions"] for row in rows),
            "difference_kinds": {
                kind: sum(row["difference_kind"] == kind for row in rows)
                for kind in ("exact", "script_variant", "contains_inserted_noise",
                             "other_error", "empty")
            },
        }

    latencies = sorted(row["latency_ms"] for row in results[1:])
    p95_index = max(0, (95 * len(latencies) + 99) // 100 - 1) if latencies else None
    summary = {"engine": engine, **metrics(results), "load_seconds": round(load_seconds, 2),
               "median_warm_latency_ms": latencies[len(latencies) // 2] if latencies else None,
               "p95_warm_latency_ms": latencies[p95_index] if latencies else None,
               "by_reference_length": {
                   band: metrics([row for row in results if row["length_band"] == band])
                   for band in ("1-2", "3-4", "other")
               },
               "note": note}
    return summary


def score_result(row):
    """Recompute proxy OCR metrics from an existing reference/prediction pair."""
    prediction = row.get("prediction", "".join(row.get("detected_lines", [])))
    reference = normalize(row["reference"])
    detected = normalize(prediction)
    distance = edit_distance(reference, detected)
    counts = edit_counts(reference, detected)
    if not detected:
        difference_kind = "empty"
    elif distance == 0:
        difference_kind = "exact"
    elif reference.translate(SCRIPT_VARIANTS) == detected.translate(SCRIPT_VARIANTS):
        difference_kind = "script_variant"
    elif counts["insertions"]:
        difference_kind = "contains_inserted_noise"
    else:
        difference_kind = "other_error"
    return {**row, "prediction": prediction, "detected": detected,
            "edit_distance": distance, "edit_counts": counts,
            "difference_kind": difference_kind, "reference_chars": len(reference),
            "length_band": length_band(len(reference))}


def make_engine(name):
    if name.startswith("rapidocr"):
        from rapidocr import ModelType, OCRVersion, RapidOCR
        variants = {
            "rapidocr": None,
            "rapidocr_v6_medium": (OCRVersion.PPOCRV6, ModelType.MEDIUM),
            "rapidocr_v5_mobile": (OCRVersion.PPOCRV5, ModelType.MOBILE),
            "rapidocr_v5_server": (OCRVersion.PPOCRV5, ModelType.SERVER),
        }
        if name not in variants:
            raise ValueError(name)
        variant = variants[name]
        params = ({"Det.ocr_version": variant[0], "Det.model_type": variant[1],
                   "Rec.ocr_version": variant[0], "Rec.model_type": variant[1]}
                  if variant else None)
        engine = RapidOCR(params=params)

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
    samples = manifest["samples"]
    if args.limit and args.limit < len(samples):
        samples = [samples[(i * len(samples) + len(samples) // 2) // args.limit]
                   for i in range(args.limit)]
    for row in samples:
        start = time.perf_counter()
        texts = infer(args.frames / row["image"])
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        result = score_result({**row, "detected_lines": texts,
                               "prediction": "".join(texts), "latency_ms": elapsed_ms})
        results.append(result)
        print(f"{row['image']}: {row['reference']} => {result['prediction']} "
              f"({elapsed_ms} ms)", flush=True)
    summary = summarize_results(args.engine, load_seconds, manifest["note"], results)
    report = {"summary": summary, "results": results}
    (args.frames / f"{args.engine}_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def rescore(args):
    """Refresh metrics for stored OCR predictions without running inference again."""
    report = json.loads(args.results.read_text(encoding="utf-8"))
    prior = report.get("summary", {})
    rows = [score_result(row) for row in report["results"]]
    summary = summarize_results(prior.get("engine", args.results.stem.removesuffix("_results")),
                                prior.get("load_seconds", 0), prior.get("note", ""), rows)
    output = args.out or args.results
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rescored {len(rows)} stored predictions to {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def canvas_size(value):
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("canvas must be WIDTHxHEIGHT, e.g. 1080x1920")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("canvas dimensions must be positive")
    return (width, height)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--video", type=Path, required=True)
    prep.add_argument("--srt", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--limit", type=int, default=16)
    prep.add_argument("--max-chars", type=int, default=0,
                      help="only sample SRT cues with at most this many alphanumeric characters")
    prep.add_argument("--min-chars", type=int, default=0,
                      help="only sample SRT cues with at least this many alphanumeric characters")
    prep.add_argument("--include-term", action="append", default=[],
                      help="only sample cues containing one of these normalized terms; repeatable")
    prep.add_argument("--roi-file", type=Path)
    prep.add_argument("--canvas", type=canvas_size, default=None,
                      help="WIDTHxHEIGHT reference canvas so a pixel-space "
                           "reference_roi can be normalized (e.g. 1080x1920)")
    prep.add_argument("--x0", type=float, default=0.1)
    prep.add_argument("--y0", type=float, default=0.44)
    prep.add_argument("--x1", type=float, default=0.85)
    prep.add_argument("--y1", type=float, default=0.78)
    merger = commands.add_parser("merge")
    merger.add_argument("--sources", type=Path, nargs="+", required=True)
    merger.add_argument("--out", type=Path, required=True)
    merger.add_argument("--per-source", type=int, default=0)
    runner = commands.add_parser("run")
    runner.add_argument("--frames", type=Path, required=True)
    runner.add_argument("--engine", choices=["rapidocr", "rapidocr_v6_medium",
                                              "rapidocr_v5_mobile", "rapidocr_v5_server",
                                              "paddleocr"], required=True)
    runner.add_argument("--limit", type=int, help="uniformly sample N extracted frames")
    scorer = commands.add_parser("rescore")
    scorer.add_argument("--results", type=Path, required=True)
    scorer.add_argument("--out", type=Path)
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
    elif args.command == "merge":
        merge(args)
    elif args.command == "run":
        run(args)
    elif args.command == "rescore":
        rescore(args)
    else:
        locate(args)


if __name__ == "__main__":
    main()
