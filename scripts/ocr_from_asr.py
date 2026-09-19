"""Read an ASR transcript and write OCR evidence beside it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.asr_adapter import run_asr_transcript, save_evidence  # noqa: E402
from app.ocr_engine import RapidOCRProvider  # noqa: E402
from app.settings import Settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="直接读取 ASR transcript.json 并生成兼容 ocr-propose 的证据文件",
    )
    parser.add_argument("transcript", type=Path, help="ASR 生成的 transcript.json")
    parser.add_argument("--video", type=Path, help="覆盖 transcript 中的 source_file")
    parser.add_argument("--output", type=Path, help="默认写到 transcript 同目录的 ocr_evidence.json")
    parser.add_argument("--profile", choices=("auto", "win11", "linux", "linux-low-vram"),
                        default="auto")
    parser.add_argument("--mode", choices=("auto", "cuda", "cpu"),
                        help="覆盖 profile 的默认执行模式")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.transcript.parent / "ocr_evidence.json"
    settings = Settings.for_profile(
        args.profile, execution_mode=args.mode, gpu_device_id=args.gpu_device_id,
    )
    provider = RapidOCRProvider(settings)
    try:
        evidence = run_asr_transcript(
            args.transcript, video_path=args.video, provider=provider, settings=settings,
        )
        result = save_evidence(output, evidence)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "output": str(result),
        "media_id": evidence["media_id"],
        "detections": len(evidence["detections"]),
        "profile": provider.status.active_profile,
        "model": provider.status.model,
        "provider": provider.status.actual_provider,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
