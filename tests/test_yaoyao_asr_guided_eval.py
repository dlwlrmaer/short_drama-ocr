import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "yaoyao_asr_guided_eval.py"
SPEC = importlib.util.spec_from_file_location("yaoyao_asr_guided_eval", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_build_candidates_uses_frozen_grid_and_asr_windows():
    baseline, guided = EVAL.build_candidates(5000, [(1000, 2000)], 1000)
    assert baseline == [0, 1000, 2000, 3000, 4000, 5000]
    assert guided == [1000, 2000]


def test_crop_box_matches_documented_roi():
    assert EVAL.crop_box([0.1, 0.44, 0.85, 0.78], 1080, 1920) == (108, 844, 810, 653)


def test_assign_and_score_are_deterministic_and_ignore_srt_text_in_selection():
    cues = [
        {"start_ms": 1000, "end_ms": 1600, "reference": "好冷"},
        {"start_ms": 3000, "end_ms": 3500, "reference": "所以"},
    ]
    frames = [
        {"candidate": 0, "pts_ms": 0, "ocr_text": "x", "ocr_ms": 10},
        {"candidate": 1, "pts_ms": 1000, "ocr_text": "好冷", "ocr_ms": 10},
        {"candidate": 2, "pts_ms": 2000, "ocr_text": "好冷", "ocr_ms": 10},
        {"candidate": 3, "pts_ms": 4600, "ocr_text": "", "ocr_ms": 5},
    ]
    assignment = EVAL.assign_frames(frames, cues, 500)
    assert assignment == {1: 0, 2: 0}
    metrics = EVAL.metrics_for_set(cues, frames, assignment)
    assert metrics["candidates"] == 4
    assert metrics["invalid_frames"] == 2
    assert metrics["coverage_cues"] == 1
    assert metrics["exact_matches"] == 1


def test_asr_guided_set_reduces_invalid_frames_on_synthetic_grid():
    baseline, guided = EVAL.build_candidates(10000, [(0, 2000)], 1000)
    guided_ids = {index for index, moment in enumerate(baseline) if moment in set(guided)}
    cues = [{"start_ms": 0, "end_ms": 1500, "reference": "妈"}]
    frames = [{"candidate": i, "pts_ms": m, "ocr_text": "妈" if m <= 1500 else "",
               "ocr_ms": 10} for i, m in enumerate(baseline)]
    all_assignment = EVAL.assign_frames(frames, cues, 500)
    guided_frames = [frame for frame in frames if frame["candidate"] in guided_ids]
    assert EVAL.metrics_for_set(cues, frames, all_assignment)["invalid_frames"] == 8
    assert EVAL.metrics_for_set(cues, guided_frames, all_assignment)["invalid_frames"] == 0


def test_summary_states_asr_is_not_ocr_truth(tmp_path):
    episodes = [{
        "episode": "01", "duration_ms": 3000, "srt": str(tmp_path / "01.srt"), "srt_sha256": "a",
        "tolerances": {"cue_frame_tolerance_ms": 500},
        "baseline": {"candidates": 3}, "asr_guided": {"candidates": 2},
    }]
    (tmp_path / "frames_01.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
    (tmp_path / "01.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n妈\n\n", encoding="utf-8")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": [
        {"segment_id": "s1", "start_ms": 1000, "end_ms": 2000, "final_text": "妈"},
    ]}), encoding="utf-8")
    checks = EVAL.asr_not_ocr_truth_checks(episodes, [transcript], tmp_path)
    assert len(checks) == 1
    assert "human SRT" in checks[0]["note"]
