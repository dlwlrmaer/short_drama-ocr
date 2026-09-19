import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_short_utterance_episodes.py"
SPEC = importlib.util.spec_from_file_location("select_short_utterance_episodes", SCRIPT)
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def test_stats_counts_target_occurrences_and_shares(tmp_path):
    srt = tmp_path / "01.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你你\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n嗯？\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\n妈妈来了\n",
        encoding="utf-8",
    )
    result = SELECTOR.stats(srt, tuple("你妈嗯"))
    assert result["total_cues"] == 3
    assert result["short_1_2_entries"] == 2
    assert result["target_seed_entries"] == 2
    assert result["target_seed_share_of_1_2"] == 1
    assert result["target_cue_entries"] == 3
    assert result["target_cue_share"] == 1
    assert result["target_occurrence_counts"] == {"你": 2, "妈": 2, "嗯": 1}
    assert result["target_total_occurrences"] == 5
