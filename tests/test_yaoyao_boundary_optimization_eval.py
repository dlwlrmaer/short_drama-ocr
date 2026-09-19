import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "yaoyao_boundary_optimization_eval.py"
SPEC = importlib.util.spec_from_file_location("boundary_eval", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def box(text, score=0.99, x=0.5, y=0.76):
    return {"text": text, "score": score, "x_center": x, "y_center": y, "bbox": []}


def row(pts, text, boxes, sharpness=100):
    return {"pts_ms": pts, "filtered_text": text, "boxes": boxes,
            "sharpness": sharpness, "ocr_ms": 10}


def test_cue_window_trims_and_collapses_too_short_cue():
    assert EVAL.cue_window({"start_ms": 1000, "end_ms": 1600}, 100) == (1100, 1500)
    assert EVAL.cue_window({"start_ms": 1000, "end_ms": 1150}, 100) == (1075, 1075)


def test_augmentation_planning_uses_timing_not_text():
    cue = {"episode": 1, "start_ms": 1000, "end_ms": 1800, "cue_id": "x"}
    source = {1: [row(1200, "anything", []), row(1400, "", [])]}
    planned, _ = EVAL.planned_augmentations([cue], source, trims=(100,))
    assert planned[1] == [1660]


def test_quality_selector_removes_unstable_logo_and_keeps_stable_subtitle():
    cue = {"start_ms": 1000, "end_ms": 1800}
    rows = [
        row(1100, "LOGOA不是", [box("LOGOA", .70, .36, .716), box("不是", .999, .5)]),
        row(1400, "LOGOB不是", [box("LOGOB", .75, .36, .717), box("不是", .998, .5)], 150),
        row(1700, "LOGOC不是", [box("LOGOC", .72, .36, .716), box("不是", .999, .5)], 120),
    ]
    text, selected, meta = EVAL.select_quality(cue, rows, rows)
    assert text == "不是"
    assert selected["pts_ms"] == 1400
    assert meta["evidence"] == "confirmed_box_track"
    assert meta["evidence_span_ms"] == 600


def test_single_visible_frame_is_retained_by_fallback():
    cue = {"start_ms": 1000, "end_ms": 1300}
    only = row(1150, "妈", [box("妈")])
    text, selected, meta = EVAL.select_quality(cue, [only], [only])
    assert text == "妈"
    assert selected is only
    assert meta["evidence"] == "single_frame_fallback"


def test_trimmed_empty_uses_untrimmed_visible_fallback():
    cue = {"start_ms": 1000, "end_ms": 1300}
    edge = row(1010, "妈", [box("妈")])
    text, selected, meta = EVAL.select_quality(cue, [], [edge])
    assert text == "妈"
    assert selected is edge
    assert meta["selection_stage"] == "untrimmed_fallback"
