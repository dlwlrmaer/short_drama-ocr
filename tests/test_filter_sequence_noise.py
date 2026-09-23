import json

from scripts.filter_sequence_noise import clean_root


def test_noise_filter_keeps_evidence_for_audit_and_rewrites_srt(tmp_path):
    path = tmp_path / "evidence" / "episode_0020_ocr.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "guidance": {"type": "visual_sequence_asr_timing"},
        "segments": [],
        "summary": {"detected_segments": 2},
        "detections": [
            {"id": "visual-1", "start_ms": 0, "end_ms": 200,
             "ocr_text": "C07C3C496", "confidence": 0.77, "observations": 1},
            {"id": "visual-2", "start_ms": 200, "end_ms": 900,
             "ocr_text": "我来了", "confidence": 0.99, "observations": 2},
        ],
    }), encoding="utf-8")
    report = clean_root(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert report["noise_rejected"] == 1
    assert [row["ocr_text"] for row in saved["detections"]] == ["我来了"]
    assert saved["rejected_noise"][0]["ocr_text"] == "C07C3C496"
    srt = tmp_path / "subtitles" / "20.srt"
    assert "我来了" in srt.read_text(encoding="utf-8")
    srt.write_text("stale", encoding="utf-8")
    assert clean_root(tmp_path)["noise_rejected"] == 1
    assert "我来了" in srt.read_text(encoding="utf-8")
