"""Compare RapidOCR CUDA and CPU providers on the same prepared frame manifest."""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ocr_engine import RapidOCRProvider  # noqa: E402
from app.settings import Settings  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]", "", text or "").lower()


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left != right)))
        previous = current
    return previous[-1]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def image_path(frames_dir: Path, sample: dict) -> Path:
    value = sample.get("image") or sample.get("path")
    if not value:
        raise ValueError("manifest sample 缺少 image/path")
    path = Path(value)
    return path if path.is_absolute() else frames_dir / path


def run_mode(frames_dir: Path, samples: list[dict], mode: str) -> dict:
    provider = RapidOCRProvider(Settings(execution_mode=mode))
    provider.ensure_ready()
    results = []
    latencies = []
    started_all = time.perf_counter()
    for sample in samples:
        image = np.asarray(Image.open(image_path(frames_dir, sample)).convert("RGB"))
        started = time.perf_counter()
        prediction = provider.text(image)
        latency = (time.perf_counter() - started) * 1000
        reference = normalize(sample.get("reference", ""))
        detected = normalize(prediction)
        results.append({"image": str(image_path(frames_dir, sample)),
                        "reference": sample.get("reference", ""),
                        "prediction": prediction, "latency_ms": round(latency, 3),
                        "exact": reference == detected,
                        "edit_distance": edit_distance(reference, detected)})
        latencies.append(latency)
    wall = time.perf_counter() - started_all
    chars = sum(len(normalize(row["reference"])) for row in results)
    edits = sum(row["edit_distance"] for row in results)
    return {
        "mode": mode,
        "provider": provider.status.to_dict(),
        "samples": len(results),
        "exact_matches": sum(row["exact"] for row in results),
        "cer": round(edits / chars, 4) if chars else None,
        "throughput_frames_per_second": round(len(results) / wall, 3) if wall else None,
        "wall_seconds": round(wall, 3),
        "warm_latency_p50_ms": round(statistics.median(latencies[1:] or latencies), 3),
        "warm_latency_p95_ms": round(percentile(latencies[1:] or latencies, 0.95), 3),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("cuda", "cpu"), default=("cuda", "cpu"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.frames / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][:args.limit] if args.limit else manifest["samples"]
    reports = [run_mode(args.frames, samples, mode) for mode in args.modes]
    payload = {"frames": str(args.frames), "reports": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": payload["frames"], "reports": [
        {key: value for key, value in report.items() if key != "results"} for report in reports
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
