from scripts.evaluate_sequence_recall import compare, read_srt


def test_reference_matching_uses_visual_time_and_one_to_one_text(tmp_path):
    path = tmp_path / "01.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:01,400\n去吧\n\n"
                    "2\n00:00:01,500 --> 00:00:01,900\n等等\n", encoding="utf-8")
    reference = read_srt(path)
    result = compare(reference, [
        {"frame_pts_ms": 1200, "ocr_text": "去吧"},
        {"frame_pts_ms": 1700, "ocr_text": "等等"},
    ])
    assert result["visual_cues_found"] == 2
    assert result["rough_text_matches"] == 2
    assert result["visual_recall"] == 1


def test_one_detection_cannot_count_for_two_reference_cues():
    reference = [{"start_ms": 1000, "end_ms": 1500, "text": "你好"},
                 {"start_ms": 1500, "end_ms": 2000, "text": "再见"}]
    result = compare(reference, [{"frame_pts_ms": 1500, "ocr_text": "你好"}])
    assert result["visual_cues_found"] == 1
