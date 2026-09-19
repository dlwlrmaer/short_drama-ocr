"""Initialize RapidOCR and print the actual ONNX Runtime provider as JSON."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ocr_engine import RapidOCRProvider  # noqa: E402
from app.settings import Settings  # noqa: E402


def main() -> None:
    provider = RapidOCRProvider(Settings.from_env())
    try:
        provider.ensure_ready()
    except Exception:
        print(json.dumps(provider.status.to_dict(), ensure_ascii=False, indent=2))
        raise
    print(json.dumps(provider.status.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
