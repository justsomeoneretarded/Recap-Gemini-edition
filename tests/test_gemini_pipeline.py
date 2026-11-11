"""Offline end-to-end and resume tests of the multi-stage Gemini pipeline."""
import json
import re
from pathlib import Path

import pytest

from shorts_pipeline.cli import generate
from shorts_pipeline.gemini.client import BudgetExceeded, GeminiTimeout
from gemini_fakes import FakeAPIError, FakeSynth, FakeTransport, make_source, synthetic_events

SPEECH = {"cache_key": "fake", "language": "en",
          "segments": [{"start": 13.0, "end": 14.5, "text": "Hold the line."}],
          "candidates": [{"id": "D0001", "start": 13.0, "end": 14.5, "text": "Hold the line.",
                          "words": [{"word": "Hold", "start": 0.1, "end": 0.4}, {"word": "the", "start": 0.45, "end": 0.6},
                                    {"word": "line.", "start": 0.65, "end": 1.1}],
                          "confidence": 0.93, "timing_provenance": "asr-estimated-source-speech"}]}


def _request(source, **extra):
    return {"title": "Synthetic Gemini run", "source": str(source), "brief": "Follow the stranger.",
            "story_notes": "", "target_seconds": 20, "fps": 30, "preview_width": 160, "planner": "gemini",
            "gemini_preset": "balanced", **extra}


@pytest.fixture
def source(tmp_path):
    return make_source(tmp_path / "source.mp4", 30)


@pytest.fixture(autouse=True)
def fake_speech(monkeypatch):
    from shorts_pipeline import source_speech
    monkeypatch.setattr(source_speech, "analyze_source_speech", lambda path, work: SPEECH)


def test_end_to_end_offline_gemini_run_produces_evidence_linked_short(tmp_path, source):
    transport = FakeTransport(events=synthetic_events(30))
    out = tmp_path / "out"
    result = generate(_request(source), tmp_path / "work", out, gemini_transport=transport, synthesize=FakeSynth())
    assert Path(result["final"]).is_file()
    for name in ("event-inventory.json", "story-map.json", "editorial-plan.json", "script-draft.json",
                 "evidence-map.json", "storyboard.json", "timeline.json", "narration-timing.json", "captions.srt",
                 "captions.ass", "quality-review.json", "correction-history.json", "cost-report.json",
                 "final-verification.json", "gemini-settings.json", "cover-01.jpg", "script.txt"):
        assert (out / name).is_file(), name
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "succeeded" and status["production_ready"] is False
    assert status["review_status"] in ("passed_automated_review", "passed_with_warnings")
    # Every narration sentence is placed only on footage chosen for it, with a reason.
    evidence = json.loads((out / "evidence-map.json").read_text())
    assert evidence["sentences"] and all(s["shots"] and all(sh["reason"] for sh in s["shots"]) for s in evidence["sentences"])
    timeline = json.loads((out / "timeline.json").read_text())
    for beat in timeline["beats"]:
        if beat["kind"] != "narration":
            assert beat["kind"] == "dialogue" and beat["speech_id"] == "D0001"
            continue
        for shot in beat["render_shots"]:
            matched = [s for s in beat["shots"] if s["sentence_index"] == shot["sentence_index"]]
            assert any(m["start"] - 1e-6 <= shot["start"] and shot["end"] <= m["end"] + 1e-6 for m in matched)
    # No source range is used twice.
    ranges = sorted((s["start"], s["end"]) for b in timeline["beats"] for s in b["render_shots"])
    assert all(b[0] >= a[1] - 0.25 for a, b in zip(ranges, ranges[1:]))
    cost = json.loads((out / "cost-report.json").read_text())
    assert {"analysis", "planning:story_map", "planning:editorial", "writing:draft", "matching:clips",
            "writing:picture", "writing:reconcile", "review"} <= set(cost["by_stage"])
    assert cost["estimated_total_usd"] > 0 and cost["by_stage"]["analysis"]["models"] == ["gemini-3.8-flash"]
    assert cost["by_stage"]["writing:draft"]["models"] == ["gemini-3.1-pro-preview"]
    verification = json.loads((out / "final-verification.json").read_text())
    assert verification["human_review_required"] is True and verification["checks"]["container"]["width"] == 160
    assert not [i for i in verification["issues"] if i["severity"] == "critical"], verification["issues"]
    assert transport.uploads == 1  # The analysis proxy is uploaded once and reused by every stage.
    # Hybrid script: picture-first action lines with key words aligned to their clip's action anchor.
    for name in ("clip-plan.json", "picture-script.json", "script-merged.json"):
        assert (out / name).is_file(), name
    sentences = [s for b in timeline["beats"] for s in b.get("sentences", [])]
    assert {s["line_kind"] for s in sentences} == {"action", "context"}
    aligned = [r for b in timeline["beats"] for r in b["render_shots"] if r.get("key_word_aligned")]
    assert aligned
    for beat in timeline["beats"]:
        if beat["kind"] != "narration":
            continue
        from shorts_pipeline.render import sentence_spans
        spans = sentence_spans(beat)
        cursor = beat.get("render_timeline_start", 0)
        position = 0.0
        for shot in beat["render_shots"]:
            if shot.get("key_word_aligned"):
                on_screen = position + (shot["anchor"] - shot["start"])
                spoken = spans[shot["sentence_index"]][0] + shot["key_time"]
                assert abs(on_screen - spoken) <= 1.5 / 30 + 1e-6  # Frame quantization of cut + window.
            position += shot["duration"]
    script = (out / "script.txt").read_text()
    assert "Hold the line." in script


def test_interrupted_run_resumes_without_repeating_finished_stages(tmp_path, source):
    events = synthetic_events(30)
    failing = FakeTransport(events=events, overrides={"ScriptDraft": FakeAPIError("timeout", "deadline exceeded")})
    with pytest.raises(GeminiTimeout):
        generate(_request(source), tmp_path / "work", tmp_path / "out1", gemini_transport=failing,
                 synthesize=FakeSynth())
    status = json.loads((tmp_path / "out1" / "status.json").read_text())
    assert status["state"] == "failed" and "ScriptDraft" not in status["error"]
    resumed = FakeTransport(events=events, overrides={
        name: AssertionError(f"{name} should have been resumed") for name in ("WindowAnalysis", "StoryMap", "EditorialPlan")})
    generate(_request(source), tmp_path / "work", tmp_path / "out2", gemini_transport=resumed, synthesize=FakeSynth())
    schemas = [c["schema"] for c in resumed.calls]
    assert "ScriptDraft" in schemas and "WindowAnalysis" not in schemas
    assert resumed.uploads == 0  # The uploaded file was reused from the registry.


def test_preflight_budget_stops_before_any_upload(tmp_path, source):
    transport = FakeTransport(events=synthetic_events(30))
    with pytest.raises(BudgetExceeded, match="before any upload"):
        generate(_request(source, max_budget_usd=0.0001), tmp_path / "work", tmp_path / "out",
                 gemini_transport=transport, synthesize=FakeSynth())
    assert transport.uploads == 0 and transport.calls == []


def test_narration_overflow_is_shortened_by_targeted_rewrite(tmp_path, source):
    transport = FakeTransport(events=synthetic_events(30))
    slow = FakeSynth(words_per_second=1.6)  # Much slower than the planning rate: the first take overflows.
    out = tmp_path / "out"
    generate(_request(source), tmp_path / "work", out, gemini_transport=transport, synthesize=slow)
    history = json.loads((out / "correction-history.json").read_text())
    assert any(h.get("action") == "shorten_beat" and h["outcome"] == "rewritten" for h in history)
    timeline = json.loads((out / "timeline.json").read_text())
    assert timeline["duration"] <= 20.0 + 1e-6
    assert any(c["schema"] == "BeatRewrite" for c in transport.calls)


def test_unfixable_overflow_fails_with_diagnostics(tmp_path, source):
    transport = FakeTransport(events=synthetic_events(30))
    with pytest.raises(ValueError, match="maximum is 20s|shortening pass"):
        generate(_request(source), tmp_path / "work", tmp_path / "out", gemini_transport=transport,
                 synthesize=FakeSynth(words_per_second=0.4))
    assert json.loads((tmp_path / "out" / "status.json").read_text())["state"] == "failed"


def test_failed_corrections_are_reported_as_needing_human_review(tmp_path, source):
    review = {"verdict": "revise", "strengths": [], "opening_score": 2, "coherence_score": 3, "ending_score": 3,
              "issues": [{"check": "unsupported_statement", "severity": "critical", "beat_id": "N01",
                          "sentence_id": "N01.1", "description": "claims a rifle that is never shown",
                          "fix": "rewrite_beat"}]}
    transport = FakeTransport(events=synthetic_events(30), overrides={
        "QualityReview": review,
        # Every rewrite cites no evidence, so validation rejects it and the fix fails.
        "BeatRewrite": {"beat_id": "N01", "rationale": "x", "sentences": [
            {"text": "Something happens somewhere now.", "event_ids": ["NOPE"], "visual_requirement": "",
             "claim_basis": "observed"}]}})
    out = tmp_path / "out"
    generate(_request(source, gemini_preset="economy", review_passes=2), tmp_path / "work", out,
             gemini_transport=transport, synthesize=FakeSynth())
    quality = json.loads((out / "quality-review.json").read_text())
    assert quality["review_status"] == "needs_human_review"
    history = json.loads((out / "correction-history.json").read_text())
    assert [h for h in history if h.get("stage") == "I" and h["outcome"] == "failed"]
    assert json.loads((out / "status.json").read_text())["review_status"] == "needs_human_review"
    with pytest.raises(RuntimeError, match="critical review issue"):
        generate(_request(source, gemini_preset="economy", review_passes=1, strict_review=True), tmp_path / "work",
                 tmp_path / "strict", gemini_transport=transport, synthesize=FakeSynth())


def test_review_driven_rematch_replaces_a_mismatched_shot(tmp_path, source):
    calls = {"n": 0}

    def review(parts):
        calls["n"] += 1
        issues = [] if calls["n"] > 1 else [{"check": "shot_mismatch", "severity": "major", "beat_id": "N01",
                                             "sentence_id": "N01.1", "description": "shot shows the wrong moment",
                                             "fix": "rematch_shot"}]
        return {"verdict": "revise" if issues else "pass", "issues": issues, "strengths": [],
                "opening_score": 3, "coherence_score": 3, "ending_score": 3}
    transport = FakeTransport(events=synthetic_events(30), overrides={"QualityReview": review})
    out = tmp_path / "out"
    generate(_request(source), tmp_path / "work", out, gemini_transport=transport, synthesize=FakeSynth())
    history = json.loads((out / "correction-history.json").read_text())
    fix = next(h for h in history if h.get("stage") == "I")
    assert fix["outcome"] == "applied" and "re-matched N01.1" in fix["action"]
    matching_calls = [c for c in transport.calls if c["schema"] == "MatchingResult"]
    assert '"N01.1"' in matching_calls[-1]["parts"][-1]["text"]


def test_gemini_failure_uses_optional_local_fallback_writer(tmp_path, source, monkeypatch):
    from shorts_pipeline import planner
    used = {}

    def local(request, path, scenes, work, *, source_speech):
        used["writer"] = request["script_model"]
        return {"beats": [{"id": "N1", "kind": "narration", "text": "A fallback line plays.", "shots": [{"start": 0, "end": 4}]}]}
    monkeypatch.setattr(planner, "make_plan", local)
    transport = FakeTransport(models=["something-else"])
    out = tmp_path / "out"
    generate(_request(source, fallback_writer="gemma4:12b"), tmp_path / "work", out, gemini_transport=transport,
             synthesize=FakeSynth())
    status = json.loads((out / "status.json").read_text())
    assert used["writer"] == "gemma4:12b" and "None of the configured" in status["fallback"]["reason"]
    assert status["state"] == "succeeded"


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    import shorts_pipeline.gemini.client as client
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)


def test_free_tier_end_to_end_run_is_paced_and_reported_as_unbilled(tmp_path, source):
    transport = FakeTransport(events=synthetic_events(30), models=["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    out = tmp_path / "out"
    generate(_request(source, gemini_preset="free"), tmp_path / "work", out, gemini_transport=transport,
             synthesize=FakeSynth())
    cost = json.loads((out / "cost-report.json").read_text())
    assert "not billed" in cost["billing"] and cost["max_budget_usd"] is None
    assert 0 < cost["requests_sent"] <= 150
    assert {c["model"] for c in transport.calls} == {"gemini-2.5-flash"}
    settings = json.loads((out / "gemini-settings.json").read_text())
    assert settings["free_tier"] and settings["preflight_estimate"]["estimated_requests"] <= 150


def test_invalid_merge_falls_back_to_the_story_first_script(tmp_path, source):
    bad = {"title": "t", "description": "d", "merge_notes": [], "beats": [{"beat_id": "m", "kind": "narration", "lines": [
        {"kind": "action", "text": "He runs away from the truck tonight.", "source_line_id": "L99", "clip_ids": ["K001"]}]}]}
    transport = FakeTransport(events=synthetic_events(30), overrides={"ReconciledScript": bad})
    out = tmp_path / "out"
    generate(_request(source), tmp_path / "work", out, gemini_transport=transport, synthesize=FakeSynth())
    history = json.loads((out / "correction-history.json").read_text())
    assert any(h.get("stage") == "D2" and h["outcome"] == "fell_back_to_story_script" for h in history)
    assert any(c["schema"] == "MatchingResult" for c in transport.calls)  # Story-first sentences were matched.
    assert json.loads((out / "status.json").read_text())["state"] == "succeeded"


def test_merge_accepts_a_slightly_short_script(tmp_path, source):
    """Live run 43016875121 rejected a merge for being 2 words under the draft minimum."""
    transport = FakeTransport(events=synthetic_events(30))
    original = transport._ReconciledScript

    def short(parts):
        merged = original(parts)
        for beat in merged["beats"]:
            beat["lines"] = [l for l in beat.get("lines", []) if l["kind"] == "action"] if beat["kind"] == "narration" else []
        return merged
    transport.overrides["ReconciledScript"] = short
    out = tmp_path / "out"
    generate(_request(source), tmp_path / "work", out, gemini_transport=transport, synthesize=FakeSynth())
    history = json.loads((out / "correction-history.json").read_text())
    assert any(h.get("stage") == "D2" and h["outcome"] == "accepted" for h in history)


def test_short_footage_is_extended_from_verified_events_never_rendered_short(tmp_path, source):
    """Live run 43063993344 reached render with a sentence holding 1.47s of footage for 2.90s of speech."""
    transport = FakeTransport(events=synthetic_events(30))
    original = transport._ClipPlan

    def tiny(parts):
        plan = original(parts)
        for event in plan["events"]:
            for clip in event["clips"]:
                clip["end"] = clip["start"] + 2.0  # Less picture than the slow voice needs.
                clip["anchor"] = clip["start"] + 1.0
        return plan
    transport.overrides["ClipPlan"] = tiny
    out = tmp_path / "out"
    generate(_request(source), tmp_path / "work", out, gemini_transport=transport,
             synthesize=FakeSynth(words_per_second=1.7))
    history = json.loads((out / "correction-history.json").read_text())
    assert any(h.get("action") in ("event_footage_fallback", "remove_short_sentence") for h in history)
    timeline = json.loads((out / "timeline.json").read_text())
    for beat in timeline["beats"]:
        if beat["kind"] == "narration":
            assert all(shot["speed"] >= 0.85 for shot in beat["render_shots"])
    assert json.loads((out / "status.json").read_text())["state"] == "succeeded"
