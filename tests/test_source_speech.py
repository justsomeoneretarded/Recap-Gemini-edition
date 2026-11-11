import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from shorts_pipeline import source_speech as speech


def segment(**changes):
    value = {"id": 7, "start": 1., "end": 3., "text": " Hold your fire!",
             "avg_logprob": -.1, "no_speech_prob": .02,
             "words": [{"word": word, "start": start, "end": end, "probability": .95}
                       for word, start, end in [(" Hold", 1.1, 1.5),
                                                (" your", 1.5, 2.), (" fire!", 2., 2.8)]]}
    value.update(changes)
    return value


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source one")
    monkeypatch.setattr(speech, "_video_info", lambda path: {"duration": 20., "has_audio": True})
    monkeypatch.setattr(speech, "_transcribe", lambda *args: {
        "language": "en", "language_probability": .99, "segments": [segment()]})
    monkeypatch.delenv("SHORTS_SOURCE_WHISPER_MODEL", raising=False)
    monkeypatch.delenv("SHORTS_MAX_SOURCE_SPEECH_CANDIDATES", raising=False)
    return source, tmp_path / "work"


def test_english_candidate_relative_words_and_complete_evidence(fixture):
    result = speech.analyze_source_speech(*fixture)
    candidate = result["candidates"][0]
    assert candidate["id"] == "D0001"
    assert candidate["text"] == "Hold your fire!"
    assert candidate["words"][0]["start"] == pytest.approx(.1)
    assert candidate["timing_provenance"] == "asr-estimated-source-speech"
    assert "speaker" not in candidate
    saved = json.loads(open(result["transcript_path"]).read())
    assert saved["segments"][0] == segment()
    assert saved["filtering_report"]["accepted_count"] == 1


@pytest.mark.parametrize("changes,reason", [
    ({"start": -1.}, "invalid_segment_bounds"),
    ({"end": 22.}, "invalid_segment_bounds"),
    ({"end": 1.39}, "duration_outside_0.4_to_8_seconds"),
    ({"end": 12.}, "duration_outside_0.4_to_8_seconds"),
    ({"words": []}, "word_count_outside_2_to_28"),
    ({"no_speech_prob": .35}, "high_or_missing_no_speech_probability"),
])
def test_rejection_is_explicit(fixture, monkeypatch, changes, reason):
    monkeypatch.setattr(speech, "_transcribe", lambda *args: {
        "language": "en", "segments": [segment(**changes)]})
    result = speech.analyze_source_speech(*fixture)
    assert result["candidates"] == []
    assert result["filtering_report"]["rejection_counts"] == {reason: 1}


@pytest.mark.parametrize("field,value,reason", [
    ("end", 1.1, "invalid_word_bounds"),
    ("start", float("nan"), "invalid_word_bounds"),
    ("end", 21., "invalid_word_bounds"),
    ("probability", .75, "low_word_confidence"),
])
def test_bad_word_bounds_and_confidence(fixture, monkeypatch, field, value, reason):
    item = segment()
    for word in item["words"]:
        word[field] = value
    monkeypatch.setattr(speech, "_transcribe", lambda *args: {"language": "en", "segments": [item]})
    result = speech.analyze_source_speech(*fixture)
    assert result["filtering_report"]["rejection_counts"] == {reason: 1}


def test_non_english_keeps_transcript_without_translation(fixture, monkeypatch):
    monkeypatch.setattr(speech, "_transcribe", lambda *args: {
        "language": "zh", "segments": [segment(text="原始文字")]})
    result = speech.analyze_source_speech(*fixture)
    assert not result["candidates"]
    assert result["segments"][0]["text"] == "原始文字"
    assert result["filtering_report"]["reason"] == "automatic_source_speech_requires_english"


def test_half_second_two_word_source_beat_preserves_native_timing(fixture, monkeypatch):
    item = segment(end=1.52, text="Thanks Doc", words=[
        {"word": "Thanks", "start": 1., "end": 1.24, "probability": .97},
        {"word": "Doc", "start": 1.26, "end": 1.52, "probability": .93},
    ])
    monkeypatch.setattr(speech, "_transcribe", lambda *args: {"language": "en", "segments": [item]})
    result = speech.analyze_source_speech(*fixture)
    candidate, = result["candidates"]
    assert candidate["text"] == "Thanks Doc"
    assert candidate["end"] - candidate["start"] == pytest.approx(.52)
    assert candidate["words"][1]["start"] == pytest.approx(.26)
    assert candidate["words"][1]["end"] == pytest.approx(.52)
    assert candidate["confidence"] == pytest.approx(.95)
    assert result["filtering_report"]["thresholds"]["min_seconds"] == .4


def test_one_word_still_rejected():
    item = segment(words=segment()["words"][:1])
    candidate, reason = speech._candidate(item, 20.)
    assert candidate is None
    assert reason == "word_count_outside_2_to_28"


def test_mock_asr_without_installed_package_metadata(fixture, monkeypatch):
    def unavailable(name):
        raise speech.PackageNotFoundError(name)
    monkeypatch.setattr(speech, "version", unavailable)
    result = speech.analyze_source_speech(*fixture)
    assert result["asr_library_version"] == "unavailable"
    assert len(result["candidates"]) == 1
    assert speech.analyze_source_speech(*fixture) == result


def test_cache_and_content_invalidation(fixture, monkeypatch):
    first = speech.analyze_source_speech(*fixture)
    def fail(*args):
        raise RuntimeError("not called on cache hit")
    monkeypatch.setattr(speech, "_transcribe", fail)
    assert speech.analyze_source_speech(*fixture) == first
    fixture[0].write_bytes(b"source changed")
    with pytest.raises(RuntimeError, match="not called"):
        speech.analyze_source_speech(*fixture)


def test_no_silent_cap_full_transcript_saved(fixture, monkeypatch):
    monkeypatch.setenv("SHORTS_MAX_SOURCE_SPEECH_CANDIDATES", "1")
    monkeypatch.setattr(speech, "_transcribe", lambda *args: {
        "language": "en", "segments": [segment(), segment(id=8)]})
    with pytest.raises(ValueError, match="narrower source excerpt"):
        speech.analyze_source_speech(*fixture)
    saved = json.loads(next((fixture[1] / "source-speech").glob("*.json")).read_text())
    assert len(saved["candidates"]) == len(saved["segments"]) == 2


def test_lazy_model_configuration_and_release(monkeypatch, tmp_path):
    import sys
    calls = {}
    class Model:
        def __init__(self, name, **options):
            calls.update(name=name, options=options)
        def transcribe(self, path, **options):
            calls["transcribe"] = options
            word = lambda text, start, end: SimpleNamespace(word=text, start=start, end=end, probability=.9)
            item = SimpleNamespace(id=0, start=0., end=2., text="Keep moving now.",
                avg_logprob=-.1, no_speech_prob=.01, compression_ratio=1.,
                words=[word("Keep", 0., .5), word("moving", .5, 1.), word("now.", 1., 2.)])
            return iter([item]), SimpleNamespace(language="en", language_probability=.99)
        def __del__(self):
            calls["released"] = True
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    result = speech._transcribe(tmp_path / "source.mp4", "base", 2)
    assert result["segments"][0]["words"][2]["end"] == 2.
    assert calls["options"] == {"device": "cpu", "compute_type": "int8", "cpu_threads": 2, "num_workers": 1}
    assert calls["transcribe"]["word_timestamps"] is True
    assert calls["transcribe"]["task"] == "transcribe"
    assert calls["released"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_real_tiny_video_probe_with_mock_asr(tmp_path, monkeypatch):
    source = tmp_path / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=16x16:r=10",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "4", "-c:v", "libx264",
                    "-threads", "1", "-c:a", "aac", str(source)], check=True)
    monkeypatch.setattr(speech, "_transcribe", lambda *args: {"language": "en", "segments": [segment()]})
    result = speech.analyze_source_speech(source, tmp_path / "work")
    assert result["duration"] == pytest.approx(4, abs=.1)
    assert len(result["candidates"]) == 1
