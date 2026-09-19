import importlib.util
from pathlib import Path

from app.settings import Settings
from app.video_pipeline import cue_window


SCRIPT = Path(__file__).parents[1] / "scripts" / "yaoyao_boundary_optimization_eval.py"
SPEC = importlib.util.spec_from_file_location("boundary_eval_equivalence", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_production_cue_window_matches_frozen_evaluation():
    settings = Settings(trim_start_ms=100, trim_end_ms=100)
    for cue in (
        {"start_ms": 1000, "end_ms": 1600},
        {"start_ms": 1000, "end_ms": 1150},
        {"start_ms": 0, "end_ms": 300},
    ):
        assert cue_window(cue, settings) == EVAL.cue_window(cue, 100)
