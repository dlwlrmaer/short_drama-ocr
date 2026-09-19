import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_ocr_win.py"
SPEC = importlib.util.spec_from_file_location("benchmark_ocr_win", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_read_roi_file_accepts_manifest_and_nested_reference_roi(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"roi": [0.1, 0.4, 0.9, 0.8]}), encoding="utf-8")
    assert BENCHMARK.read_roi_file(manifest) == [0.1, 0.4, 0.9, 0.8]

    workspace = tmp_path / "ocr_workspace.json"
    workspace.write_text(json.dumps({"workspace": {"reference_roi": {
        "left": 0.2, "top": 0.5, "right": 0.8, "bottom": 0.75,
    }}}), encoding="utf-8")
    assert BENCHMARK.read_roi_file(workspace) == [0.2, 0.5, 0.8, 0.75]


def test_read_roi_file_normalizes_pixel_reference_roi_with_canvas(tmp_path):
    workspace = tmp_path / "ocr_workspace.json"
    workspace.write_text(json.dumps({"workspace": {"reference_roi": {
        "x": 52, "y": 1060, "width": 922, "height": 406,
    }}}), encoding="utf-8")
    roi = BENCHMARK.read_roi_file(workspace, (1080, 1920))
    assert roi == pytest.approx([52 / 1080, 1060 / 1920, 974 / 1080, 1466 / 1920])


def test_read_roi_file_pixel_roi_requires_canvas(tmp_path):
    workspace = tmp_path / "ocr_workspace.json"
    workspace.write_text(json.dumps({"reference_roi": {
        "x": 0, "y": 0, "w": 100, "h": 50,
    }}), encoding="utf-8")
    with pytest.raises(SystemExit):
        BENCHMARK.read_roi_file(workspace)
    assert BENCHMARK.read_roi_file(workspace, (200, 100)) == [0.0, 0.0, 0.5, 0.5]


def test_canvas_size_parses_and_rejects():
    assert BENCHMARK.canvas_size("1080x1920") == (1080, 1920)
    assert BENCHMARK.canvas_size(" 720 X 1280 ") == (720, 1280)
    with pytest.raises(Exception):
        BENCHMARK.canvas_size("1080by1920")


def test_edit_counts_and_length_band():
    assert BENCHMARK.edit_counts("你", "你啊") == {
        "insertions": 1, "deletions": 0, "substitutions": 0,
    }
    assert BENCHMARK.edit_counts("别走", "別走") == {
        "insertions": 0, "deletions": 0, "substitutions": 1,
    }
    assert BENCHMARK.length_band(2) == "1-2"
    assert BENCHMARK.length_band(4) == "3-4"
    assert BENCHMARK.length_band(5) == "other"
    assert BENCHMARK.score_result({"reference": "滚", "prediction": "滾",
                                   "latency_ms": 10})["difference_kind"] == "script_variant"
    assert BENCHMARK.score_result({"reference": "什么情况", "prediction": "什么情況",
                                   "latency_ms": 10})["difference_kind"] == "script_variant"


def test_summary_has_short_text_empty_variant_and_noise_metrics():
    rows = [
        {"reference_chars": 1, "edit_distance": 0, "detected": "你",
         "difference_kind": "exact", "edit_counts": {"insertions": 0},
         "length_band": "1-2", "latency_ms": 100},
        {"reference_chars": 2, "edit_distance": 1, "detected": "別走",
         "difference_kind": "script_variant", "edit_counts": {"insertions": 0},
         "length_band": "1-2", "latency_ms": 120},
        {"reference_chars": 3, "edit_distance": 3, "detected": "",
         "difference_kind": "empty", "edit_counts": {"insertions": 0},
         "length_band": "3-4", "latency_ms": 140},
        {"reference_chars": 3, "edit_distance": 1, "detected": "你好吗啊",
         "difference_kind": "contains_inserted_noise", "edit_counts": {"insertions": 1},
         "length_band": "3-4", "latency_ms": 160},
    ]
    summary = BENCHMARK.summarize_results("fake", 0.5, "proxy", rows)
    assert summary["samples"] == 4
    assert summary["empty_results"] == 1
    assert summary["script_variant_matches"] == 1
    assert summary["inserted_noise_chars"] == 1
    assert summary["inserted_noise_results"] == 1
    assert summary["difference_kinds"] == {
        "exact": 1, "script_variant": 1, "contains_inserted_noise": 1,
        "other_error": 0, "empty": 1,
    }
    assert summary["p95_warm_latency_ms"] == 160
    assert summary["by_reference_length"]["1-2"]["samples"] == 2
    assert summary["by_reference_length"]["3-4"]["samples"] == 2


def test_rescore_refreshes_stored_classification(tmp_path):
    results = tmp_path / "old_results.json"
    results.write_text(json.dumps({
        "summary": {"engine": "fake", "load_seconds": 0.25, "note": "proxy"},
        "results": [{"reference": "滚", "prediction": "滾", "latency_ms": 10}],
    }, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "nested" / "rescored.json"
    BENCHMARK.rescore(Namespace(results=results, out=output))
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["script_variant_matches"] == 1
    assert report["results"][0]["difference_kind"] == "script_variant"


def test_run_scores_and_writes_result(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "samples": [{"image": "unused.png", "reference": "妈"}],
        "note": "proxy",
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(BENCHMARK, "make_engine", lambda _name: lambda _path: ["妈"])
    BENCHMARK.run(Namespace(frames=tmp_path, engine="fake", limit=None))
    report = json.loads((tmp_path / "fake_results.json").read_text(encoding="utf-8"))
    assert report["summary"]["exact_matches"] == 1
    assert report["results"][0]["prediction"] == "妈"
