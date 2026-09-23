from app.subtitle_sequence import normalize_ocr_text, sample_points, sequence_detections
from app.video_pipeline import FrameOCR, SubtitleBox


def frame(ms: int, text: str, score: float = 0.98) -> FrameOCR:
    box = SubtitleBox(text, score, (0.3, 0.67, 0.4, 0.05)) if text else None
    return FrameOCR(ms, ms, True, text, score if text else 0.0,
                    box.bbox if box else None, [box] if box else [], 100, 1)


def test_sampling_covers_speech_and_silent_video():
    points = sample_points(2000, [{"start_ms": 700, "end_ms": 1100}],
                           speech_ms=200, fallback_ms=500, padding_ms=100)
    assert {0, 500, 1000, 1500, 600, 800, 1200} <= set(points)


def test_one_asr_window_can_emit_multiple_cues_and_silent_fallback():
    frames = [
        frame(100, "嫂嫂"), frame(350, "嫂嫂"),
        frame(600, "你去哪儿了"), frame(850, "你去哪儿了"),
        frame(1100, ""), frame(1400, "来了"), frame(1650, "来了"),
    ]
    detections = sequence_detections(
        frames, [{"start_ms": 0, "end_ms": 900}],
        duration_ms=2000, speech_ms=250, fallback_ms=400,
    )
    assert [row["ocr_text"] for row in detections] == ["嫂嫂", "你去哪儿了", "来了"]
    assert [row["guidance"] for row in detections] == [
        "asr_timing", "asr_timing", "visual_fallback",
    ]
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(detections, detections[1:]))


def test_single_short_background_mark_is_rejected():
    detections = sequence_detections([frame(500, "1")], [],
                                     duration_ms=1000, speech_ms=250, fallback_ms=500)
    assert detections == []


def test_requested_censorship_character_correction_keeps_raw_evidence():
    detections = sequence_detections([frame(500, "掐三"), frame(750, "掐三")], [],
                                     duration_ms=1000, speech_ms=250, fallback_ms=500)
    assert normalize_ocr_text("亖三") == "死死"
    assert detections[0]["ocr_text"] == "掐死"
    assert detections[0]["raw_ocr_text"] == "掐三"


def test_implausible_tiny_detection_is_rejected():
    tiny = frame(500, "公")
    tiny.bbox = (0.45, 0.69, 0.02, 0.01)
    assert sequence_detections([tiny], [{"start_ms": 400, "end_ms": 600}],
                               duration_ms=1000, speech_ms=250, fallback_ms=500) == []


def test_repeated_caption_after_visible_blank_is_separate_cue():
    frames = [frame(100, "你放开我"), frame(350, "你放开我"), frame(600, ""),
              frame(850, "你放开我"), frame(1100, "你放开我")]
    detections = sequence_detections(frames, [{"start_ms": 0, "end_ms": 1200}],
                                     duration_ms=1400, speech_ms=250, fallback_ms=450)
    assert [row["ocr_text"] for row in detections] == ["你放开我", "你放开我"]


def test_one_frame_ocr_dropout_does_not_duplicate_caption():
    frames = [frame(100, "你放开我"), frame(350, ""),
              frame(600, "你放开我"), frame(850, "你放开我")]
    detections = sequence_detections(frames, [], duration_ms=1000,
                                     speech_ms=250, fallback_ms=450)
    assert len(detections) == 1
