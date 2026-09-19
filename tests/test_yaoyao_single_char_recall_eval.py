import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "yaoyao_single_char_recall_eval.py"
SPEC = importlib.util.spec_from_file_location("single_char_eval", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_dense_schedule_keeps_grid_fallback_and_adds_window_samples():
    baseline, dense = EVAL.candidate_schedules(3000, [(1100, 1800)])
    assert baseline == [0, 1000, 2000, 3000]
    assert set(baseline) <= set(dense)
    assert [1100, 1350, 1600, 1800] == [value for value in dense if 1100 <= value <= 1800]


def test_merge_asr_windows():
    assert EVAL.merge_windows([(100, 500), (400, 900), (1200, 1300)]) == [(100, 900), (1200, 1300)]


def test_stable_selector_prefers_repeated_track_without_using_reference():
    cue = {"start_ms": 1000, "end_ms": 1800, "reference": "妈"}
    rows = [
        {"pts_ms": 1000, "filtered_text": "杂", "sharpness": 100, "boxes": []},
        {"pts_ms": 1250, "filtered_text": "妈", "sharpness": 20, "boxes": []},
        {"pts_ms": 1500, "filtered_text": "妈", "sharpness": 30, "boxes": []},
        {"pts_ms": 1750, "filtered_text": "", "sharpness": 90, "boxes": []},
    ]
    text, selected = EVAL.select_stable(cue, rows)
    assert text == "妈"
    assert selected["pts_ms"] == 1500


def test_metric_blocks_overlap_pure_one_and_one_two():
    base = {"recall": True, "exact": True, "edit_distance": 0, "empty": False}
    rows = [{**base, "text": "妈"}, {**base, "text": "你好"},
            {**base, "text": "怎么回事"}]
    assert EVAL.metric_block([row for row in rows if len(row["text"]) == 1])["cues"] == 1
    assert EVAL.metric_block([row for row in rows if 1 <= len(row["text"]) <= 2])["cues"] == 2
