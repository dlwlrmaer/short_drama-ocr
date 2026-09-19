from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.ocr_engine import OCRBox
from app.settings import Settings
from app.video_pipeline import (
    FrameOCR,
    DecodedFrame,
    SubtitleBox,
    candidate_points,
    cue_window,
    decode_frame,
    decode_frames,
    detection_for_segment,
    ocr_subtitle_frame,
    probe_duration_ms,
    probe_frame_pts,
    process_video,
    select_segment,
)


def sb(text, score=0.99, x=0.4, y=0.74, width=0.2, height=0.04):
    return SubtitleBox(text, score, (x, y, width, height))


def frame(pts, text, boxes, requested=None, guard=True, sharpness=100):
    bbox = boxes[0].bbox if len(boxes) == 1 else ((0.4, 0.74, 0.2, 0.04) if boxes else None)
    return FrameOCR(requested if requested is not None else pts, pts, guard, text,
                    0.9 if text else 0.0, bbox, boxes, sharpness, 10)


def test_candidate_schedule_combines_grid_dense_windows_and_anchors():
    settings = Settings()
    points = candidate_points(3000, [
        {"start_ms": 1100, "end_ms": 1800},
        {"start_ms": 1700, "end_ms": 2100},
    ], settings)
    assert {0, 1000, 2000, 3000} <= set(points)
    assert {600, 850, 1100, 1350, 1600, 1850, 2100, 2300} <= set(points)
    assert points == sorted(set(points))
    assert min(points) == 0 and max(points) == 3000


def test_cue_window_trims_and_collapses_short_segment():
    settings = Settings()
    assert cue_window({"start_ms": 1000, "end_ms": 1600}, settings) == (1100, 1500)
    assert cue_window({"start_ms": 1000, "end_ms": 1150}, settings) == (1075, 1075)


def test_roi_coordinate_mapping_and_subtitle_band_filtering():
    class Provider:
        def recognize(self, _image):
            # ROI is x=100..850, y=440..820 on this 1000x1000 image.
            return [
                OCRBox(((100, 275), (300, 275), (300, 315), (100, 315)), "妈", 0.99),
                OCRBox(((100, 20), (300, 20), (300, 60), (100, 60)), "店招", 0.90),
            ]

    result = ocr_subtitle_frame(Image.new("RGB", (1000, 1000), "white"),
                                1000, 1000, Provider(), Settings())
    assert result.text == "妈"
    assert result.confidence == pytest.approx(0.99)
    assert result.bbox == pytest.approx((0.2, 0.715, 0.2, 0.04))
    assert len(result.boxes) == 1


def test_roi_mapping_clamps_out_of_bounds_boxes_at_other_resolution():
    class Provider:
        def recognize(self, _image):
            return [OCRBox(((-50, 550), (1600, 550), (1600, 650), (-50, 650)), "字幕", 0.9)]

    result = ocr_subtitle_frame(Image.new("RGB", (2000, 2000), "white"),
                                1000, 1000, Provider(), Settings())
    assert result.text == "字幕"
    assert all(0 <= value <= 1 for value in result.bbox)
    assert result.bbox[0] == pytest.approx(0.075)


def test_stable_box_track_wins_over_changing_overlay():
    settings = Settings()
    rows = [
        frame(1100, "LOGOA不是", [sb("LOGOA", .7, .2), sb("不是", .99, .5)]),
        frame(1400, "LOGOB不是", [sb("LOGOB", .7, .2), sb("不是", .98, .5)], sharpness=150),
        frame(1700, "LOGOC不是", [sb("LOGOC", .7, .2), sb("不是", .99, .5)]),
    ]
    selected = select_segment({"start_ms": 1000, "end_ms": 1800}, rows, settings)
    assert selected[1] == "不是"
    assert selected[0].pts_ms == 1400
    assert selected[4]["evidence"] == "confirmed_box_track"
    assert selected[4]["evidence_span_ms"] == 600


def test_single_character_confirmation_and_fallbacks():
    settings = Settings()
    confirmed = [frame(1100, "妈", [sb("妈")]), frame(1250, "妈", [sb("妈")])]
    selected = select_segment({"start_ms": 1000, "end_ms": 1400}, confirmed, settings)
    assert selected[1] == "妈"
    assert selected[4]["evidence"] == "confirmed_box_track"

    only = frame(1150, "妈", [sb("妈")])
    selected = select_segment({"start_ms": 1000, "end_ms": 1300}, [only], settings)
    assert selected[4]["evidence"] == "single_frame_fallback"

    edge = frame(1010, "妈", [sb("妈")])
    selected = select_segment({"start_ms": 1000, "end_ms": 1300}, [edge], settings)
    assert selected[4]["stage"] == "untrimmed_fallback"
    assert select_segment({"start_ms": 1000, "end_ms": 1300}, [], settings) is None


def test_detection_contract_uses_real_pts_and_provider_status():
    provider = SimpleNamespace(status=SimpleNamespace(
        rapidocr_version="3.9.2", model="PP-OCRv6-small",
        actual_provider="CUDAExecutionProvider"))
    rows = [frame(1150, "妈", [sb("妈")], requested=1125)]
    result = detection_for_segment(0, {"start_ms": 1000, "end_ms": 1300},
                                   rows, provider, Settings())
    assert result["id"] == "segment-000001"
    assert result["segment_index"] == 0
    assert result["requested_ms"] == 1125
    assert result["frame_pts_ms"] == 1150
    assert result["asr_text_prompted"] is False
    assert result["engine_version"].endswith("CUDAExecutionProvider")
    assert all(0 <= value <= 1 for value in result["bbox"])


def test_probe_and_decode_errors_are_diagnostic(monkeypatch):
    monkeypatch.setattr("app.video_pipeline.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="bad"))
    with pytest.raises(RuntimeError, match="视频时长"):
        probe_duration_ms(Path("bad.mp4"))
    with pytest.raises(RuntimeError, match="解码"):
        decode_frame(Path("bad.mp4"), 0)


def test_decode_reports_timeout_and_missing_pts(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise __import__("subprocess").TimeoutExpired("ffmpeg", 1)

    monkeypatch.setattr("app.video_pipeline.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match="超时"):
        decode_frame(Path("bad.mp4"), 0)


def test_probe_frame_pts_keeps_original_frame_indexes(monkeypatch):
    payload = {"frames": [
        {"best_effort_timestamp_time": "0.000"},
        {},
        {"best_effort_timestamp_time": "0.080"},
    ]}
    monkeypatch.setattr("app.video_pipeline.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=__import__("json").dumps(payload), stderr=""))
    assert probe_frame_pts(Path("video.mp4")) == [(0, 0), (2, 80)]


def test_decode_frames_starts_ffmpeg_once_and_deduplicates_pts(monkeypatch, tmp_path):
    import app.video_pipeline as module

    monkeypatch.setattr(module, "probe_frame_pts", lambda _path: [(0, 0), (1, 40), (2, 80)])
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        pattern = command[-1]
        for index, color in enumerate(("red", "blue")):
            Image.new("RGB", (4, 3), color).save(pattern.replace("%06d", f"{index:06d}"))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    rows = list(decode_frames(tmp_path / "video.mp4", [10, 20, 70]))

    assert len(commands) == 1
    assert commands[0][0] == "ffmpeg"
    filter_argument = next(argument for argument in commands[0] if argument.startswith("select="))
    assert "eq(n\\,1)" in filter_argument and "eq(n\\,2)" in filter_argument
    assert [(row.requested_ms, row.pts_ms) for row in rows] == [(20, 40), (70, 80)]
    assert all(row.image.size == (4, 3) for row in rows)

    image = Image.new("RGB", (2, 2), "black")
    from io import BytesIO
    payload = BytesIO()
    image.save(payload, format="PNG")
    monkeypatch.setattr("app.video_pipeline.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=payload.getvalue(), stderr=b"no pts"))
    with pytest.raises(RuntimeError, match="真实帧 PTS"):
        decode_frame(Path("bad.mp4"), 0)


def test_process_video_deduplicates_ocr_by_actual_pts(monkeypatch):
    import app.video_pipeline as module

    calls = []

    class Provider:
        status = SimpleNamespace(rapidocr_version="3.9.2", model="x",
                                 actual_provider="CPUExecutionProvider")

        def recognize(self, _image):
            calls.append(1)
            return []

    monkeypatch.setattr(module, "probe_duration_ms", lambda _path: 1000)
    monkeypatch.setattr(module, "candidate_points", lambda *_args: [100, 120])
    monkeypatch.setattr(module, "decode_frames", lambda *_args: iter([
        DecodedFrame(100, 120, Image.new("RGB", (20, 20))),
        DecodedFrame(120, 120, Image.new("RGB", (20, 20))),
    ]))
    assert process_video(Path("video.mp4"), [], Provider(), Settings()) == []
    assert len(calls) == 1


def test_process_video_reports_batch_decode_failure(monkeypatch):
    import app.video_pipeline as module

    provider = SimpleNamespace(
        status=SimpleNamespace(rapidocr_version="3.9.2", model="x",
                               actual_provider="CPUExecutionProvider"),
        recognize=lambda _image: [],
    )
    monkeypatch.setattr(module, "probe_duration_ms", lambda _path: 1000)
    monkeypatch.setattr(module, "candidate_points", lambda *_args: [0, 1000])

    monkeypatch.setattr(module, "decode_frames", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("decode failed")))
    with pytest.raises(RuntimeError, match="批量解码候选帧"):
        process_video(Path("video.mp4"), [], provider, Settings())
