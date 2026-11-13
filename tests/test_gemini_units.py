"""Unit tests of Gemini interfaces and evidence validation (offline; no API calls)."""
import json

import pytest

from shorts_pipeline.gemini import client as client_module
from shorts_pipeline.gemini.analysis import (_fix_relative_times, dialogue_support, plan_windows,
                                             validate_events, verify_events)
from shorts_pipeline.gemini.client import (BudgetExceeded, CostLedger, GeminiAuthError, GeminiClient,
                                           GeminiModelUnavailable, GeminiOutputError, GeminiRateLimited,
                                           GeminiOverloaded, GeminiTimeout, UploadedFile, text_part)
from shorts_pipeline.gemini.matching import plan_shots, validate_choices
from shorts_pipeline.gemini.review import deterministic_checks
from shorts_pipeline.gemini.schemas import (EditorialPlan, QualityReview, ScriptBeat, ScriptSentence, StoryMap)
from shorts_pipeline.gemini.script import validate_beats
from shorts_pipeline.gemini.settings import settings_from_request
from shorts_pipeline.gemini.story import normalize_story_map, validate_editorial
from shorts_pipeline.render import allocate_segmented, allocate_shots, sentence_spans
from shorts_pipeline.config import request_config
from gemini_fakes import FakeAPIError


class Scripted:
    """Minimal transport returning queued results/exceptions for generate()."""

    def __init__(self, results=(), models=("gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite")):
        self.results, self.models, self.calls = list(results), models, 0
        self.uploaded, self.lookups = 0, 0

    def classify(self, exc):
        return getattr(exc, "category", "request"), str(exc)

    def list_models(self):
        if isinstance(self.models, Exception):
            raise self.models
        return self.models

    def generate(self, model, parts, **kwargs):
        self.calls += 1
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"text": item if isinstance(item, str) else json.dumps(item), "finish_reason": "STOP",
                "prompt_tokens": 1000, "cached_tokens": 0, "output_tokens": 100, "thinking_tokens": 0, "audio_tokens": 0}

    def upload(self, path, mime):
        self.uploaded += 1
        return {"name": "files/x", "uri": "u", "mime_type": mime, "state": "ACTIVE", "expires_epoch": 4e9}

    def get_file(self, name):
        self.lookups += 1
        return {"name": name, "uri": "u", "mime_type": "video/mp4", "state": "ACTIVE", "expires_epoch": 4e9}


REVIEW = {"verdict": "pass", "issues": []}


def make_client(tmp_path, transport, budget=5.0, **request):
    settings = settings_from_request({"gemini_preset": "balanced", "max_budget_usd": budget, **request})
    return GeminiClient(settings, CostLedger(budget, tmp_path / "cost.json"), tmp_path / "cache", transport,
                        sleep=lambda s: None)


def test_missing_api_key_explains_how_to_add_the_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiAuthError, match="GEMINI_API_KEY repository secret"):
        make_client(tmp_path, None)


def test_invalid_key_and_unsupported_models(tmp_path):
    with pytest.raises(GeminiAuthError):
        make_client(tmp_path, Scripted(models=FakeAPIError("auth", "API key not valid"))).resolve_models()
    with pytest.raises(GeminiModelUnavailable, match="None of the configured analysis models"):
        make_client(tmp_path, Scripted(models=["gemini-1.0-pro"])).resolve_models()
    client = make_client(tmp_path, Scripted(models=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]))
    resolved = client.resolve_models()
    assert resolved["analysis"] == "gemini-2.5-flash" and resolved["writing"] == "gemini-2.5-pro"
    assert any(e["kind"] == "model_substitution" for e in client.ledger.events)
    with pytest.raises(ValueError, match="gemini_models may override only"):
        settings_from_request({"gemini_models": {"music": "x"}})


def test_api_key_is_redacted_from_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSECRETSECRETSECRET")
    client = make_client(tmp_path, Scripted([FakeAPIError("request", "bad request for key AIzaSECRETSECRETSECRET")]))
    client.transport.classify = lambda exc: ("request", client_module._redact(str(exc)))
    with pytest.raises(client_module.GeminiError) as caught:
        client.generate("review", QualityReview, [text_part("x")], system="s")
    assert "AIzaSECRET" not in str(caught.value) and "[REDACTED]" in str(caught.value)
    assert "AIzaSECRET" not in (tmp_path / "cost.json").read_text()


def test_rate_limits_and_timeouts_retry_then_succeed_or_fail_clearly(tmp_path):
    transport = Scripted([FakeAPIError("rate_limit", "429"), FakeAPIError("server", "503"), REVIEW])
    client = make_client(tmp_path, transport)
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    call = [c for c in client.ledger.calls if not c.get("error")][-1]
    assert len(call["retry_reasons"]) == 2 and call["retry_reasons"][0].startswith("rate_limit")
    client = make_client(tmp_path / "b", Scripted([FakeAPIError("timeout", "deadline")] * 5))
    only = client.settings.model_preferences["review"][-1]
    client.settings.model_preferences["review"] = [only]  # No other model: the full retry budget is used.
    client.settings.overload_wait_seconds = 0  # ...and no waiting to cycle back.
    with pytest.raises(GeminiTimeout, match="after 5 attempt"):
        client.generate("review", QualityReview, [text_part("x")], system="s")
    client = make_client(tmp_path / "c", Scripted([FakeAPIError("rate_limit", "429")] * 5))
    with pytest.raises(GeminiRateLimited):
        client.generate("review", QualityReview, [text_part("x")], system="s")


def test_malformed_structured_output_is_repaired_once_then_rejected(tmp_path):
    client = make_client(tmp_path, Scripted(["{not json", {"verdict": "maybe"}, REVIEW]))
    with pytest.raises(GeminiOutputError, match="malformed QualityReview"):
        client.generate("review", QualityReview, [text_part("x")], system="s")
    assert list((tmp_path / "cache" / "diagnostics").glob("review-*.json"))
    client = make_client(tmp_path / "ok", Scripted(["{not json", REVIEW]))
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"


def test_budget_is_enforced_before_calling_the_api(tmp_path):
    transport = Scripted([REVIEW])
    client = make_client(tmp_path, transport, budget=0.0005)
    with pytest.raises(BudgetExceeded, match="would exceed"):
        client.generate("review", QualityReview, [text_part("x" * 40000)], system="s")
    assert transport.calls == 0


def test_response_cache_and_upload_reuse_avoid_repeat_costs(tmp_path):
    transport = Scripted([REVIEW])
    client = make_client(tmp_path, transport)
    for _ in range(2):
        client.generate("review", QualityReview, [text_part("same")], system="s")
    assert transport.calls == 1 and client.ledger.calls[-1]["response_cache_hit"]
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0" * 100)
    first, second = client.upload(video), client.upload(video)
    assert first.uri == second.uri and transport.uploaded == 1 and transport.lookups == 1


WINDOW = {"index": 0, "start": 0.0, "end": 30.0, "core_start": 0.0, "core_end": 30.0}
SCENES = [{"id": f"S{i}", "start": i * 6.0, "end": i * 6.0 + 6} for i in range(5)]
SEGMENTS = [{"start": 10.0, "end": 12.0, "text": "Drop the gun now."}]


def _event(**extra):
    return {"event_id": "e", "start": 7.0, "end": 11.0, "peak": 9.0, "visible_action": "a man aims a gun",
            "entities": [], "objects": [], "location": "", "expression_or_reaction": "", "dialogue_heard": "",
            "sound": "", "kind": "action", "importance": 4, "confidence": 0.8, "motion": "normal", "focus_x": 0.5,
            "uncertainty": "", **extra}


def test_event_validation_rejects_bad_timestamps_hallucinated_dialogue_and_identities():
    events = [_event(start=40.0, end=44.0, peak=42.0),
              _event(dialogue_heard="Drop the gun now"),
              _event(dialogue_heard="I am your father"),
              _event(entities=[{"label": "man", "name": "Marcus", "name_basis": "story_notes"}]),
              _event(entities=[{"label": "man", "name": "Viktor", "name_basis": "story_notes"}])]
    kept, rejected = validate_events(events, WINDOW, SCENES, SEGMENTS, "Marcus is a retired soldier.", 30.0)
    assert len(rejected) == 1 and "outside analyzed window" in rejected[0]["rejection"]
    assert kept[0]["dialogue_status"] == "asr_supported" and kept[0]["shot_ids"] == ["S1"]
    assert kept[1]["dialogue_heard"] == "" and kept[1]["dialogue_status"] == "unverified_removed"
    assert kept[2]["entities"][0]["name"] == "Marcus"
    assert kept[3]["entities"][0]["name"] == "" and "unsupported_identity_removed" in kept[3]["flags"]
    assert dialogue_support("completely different words", SEGMENTS, 7, 11) < 0.6


def test_clip_relative_timestamps_are_detected_and_shifted():
    window = {"start": 100.0, "end": 160.0, "core_start": 100, "core_end": 160}
    events = [_event(start=5.0, end=9.0, peak=7.0)]
    assert _fix_relative_times(events, window) and events[0]["start"] == 105.0


def test_windows_snap_to_shot_boundaries_and_overlap():
    windows = plan_windows(700.0, [{"start": s} for s in (0, 290, 305, 610)], 300.0)
    assert windows[0]["core_end"] == 305 and windows[1]["start"] == 303
    assert windows[-1]["core_end"] == 700.0


def test_zoom_verification_rejects_events_not_observed(tmp_path):
    transport = Scripted([{"checks": [
        {"event_id": "EV0001", "verdict": "not_observed", "start": 7, "end": 11, "peak": 9, "visible_action": "", "notes": "no gun"},
        {"event_id": "EV0002", "verdict": "revised", "start": 13.5, "end": 15.0, "peak": 14.0,
         "visible_action": "he fires once", "focus_x": 0.3}]}])
    client = make_client(tmp_path, transport)
    events = [{**_event(), "id": "EV0001", "status": "observed", "confidence": 0.5},
              {**_event(start=13.0, end=16.0, peak=14.0, motion="fast"), "id": "EV0002", "status": "observed"}]
    file = UploadedFile("files/x", "u", "video/mp4", "abc")
    report = verify_events(client, file, events, client.settings.analysis, 30.0, log=lambda *_: None)
    assert events[0]["status"] == "rejected" and events[1]["status"] == "verified"
    assert events[1]["start"] == 13.5 and events[1]["visible_action"] == "he fires once"
    assert report["not_observed"] == 1 and report["revised"] == 1


INVENTORY = {"events": [
    {"id": "EV0001", "start": 0.5, "end": 5.5, "visible_action": "he opens the crate", "status": "verified",
     "confidence": 0.9, "importance": 4, "shot_ids": ["S0"], "flags": [], "entities": [], "dialogue_status": "none"},
    {"id": "EV0002", "start": 6.5, "end": 11.5, "visible_action": "he lifts a rifle", "status": "observed",
     "confidence": 0.8, "importance": 5, "shot_ids": ["S1"], "flags": [], "entities": [], "dialogue_status": "asr_supported",
     "dialogue_heard": "Drop the gun now"},
    {"id": "EV0003", "start": 12.5, "end": 17.5, "visible_action": "a truck drives off", "status": "observed",
     "confidence": 0.8, "importance": 3, "shot_ids": ["S2"], "flags": [], "entities": [], "dialogue_status": "none"}]}
POOL = {"D1": {"id": "D1", "start": 20.0, "end": 26.0, "text": "Get in the truck.", "confidence": 0.9,
               "words": [{"word": "Get", "start": 0, "end": 0.3}], "timing_provenance": "asr"}}


def test_story_map_drops_unknown_events_and_uncited_inferences():
    raw = StoryMap.model_validate({"logline": "x", "characters": [
        {"character_id": "C1", "label": "man", "name": "Viktor", "name_basis": "story_notes", "event_ids": ["EV0001"]}],
        "events": [{"event_id": "EV9999", "summary": "invented"},
                   {"event_id": "EV0001", "summary": "opens crate", "causes_event_ids": ["EV0002", "EV7777"],
                    "inferred_context": [{"claim": "he is a thief", "basis": "visual_sequence", "supporting_event_ids": ["EV7"]},
                                         {"claim": "he planned it", "basis": "visual_sequence", "supporting_event_ids": ["EV0002"]}]}]})
    data = normalize_story_map(raw, INVENTORY, notes="", transcript="")
    ids = [e["event_id"] for e in data["events"]]
    assert "EV9999" not in ids and ids == ["EV0001", "EV0002", "EV0003"]
    first = data["events"][0]
    assert first["causes_event_ids"] == ["EV0002"] and [c["claim"] for c in first["inferred_context"]] == ["he planned it"]
    assert data["characters"][0]["name"] == "" and data["validation_findings"]


def _plan(**extra):
    return EditorialPlan.model_validate({"title": "t", "description": "d", "hook_event_id": "EV0001",
                                         "ending_mode": "renewed_threat", "selected": [
                                             {"event_id": i, "role": "escalation", "reason": "r"} for i in ("EV0001", "EV0002", "EV0003")],
                                         **extra})


def test_editorial_validation_catches_invalid_references_order_and_dialogue_excess():
    scenes = [{"id": f"S{i}", "start": i * 6.0, "end": i * 6.0 + 6} for i in range(3)]
    assert validate_editorial(_plan(), INVENTORY, POOL, scenes, 15, 20)["selected_footage_seconds"] == 18
    bad = _plan(selected=[{"event_id": "EV0042", "role": "hook", "reason": "r"}], hook_event_id="EV0042")
    with pytest.raises(ValueError, match="unknown or rejected"):
        validate_editorial(bad, INVENTORY, POOL, scenes, 15, 20)
    order = [{"event_id": i, "role": "hook", "reason": "r"} for i in ("EV0003", "EV0001", "EV0002")]
    with pytest.raises(ValueError, match="chronological"):
        validate_editorial(_plan(selected=order, hook_event_id="EV0003"), INVENTORY, POOL, scenes, 15, 20)
    assert validate_editorial(_plan(selected=order, hook_event_id="EV0003", cold_open=True), INVENTORY, POOL, scenes, 15, 20)
    with pytest.raises(ValueError, match="source dialogue"):
        validate_editorial(_plan(dialogue_speech_ids=["D1"]), INVENTORY, POOL, scenes, 15, 20)


def _beat(*texts, ids=("EV0001",), kind="narration", speech=""):
    return ScriptBeat(beat_id="b", kind=kind, speech_id=speech,
                      sentences=[ScriptSentence(text=t, event_ids=list(ids), visual_requirement="crate") for t in texts])


def test_script_validation_rejects_unsupported_or_invented_content():
    ok = validate_beats([_beat("He pries the crate open with both hands.")], INVENTORY, POOL,
                        allowed_quotes=[], budget_words=None)
    assert ok[0]["sentences"][0]["sentence_id"] == "N01.1"
    cases = [([_beat("He pries the crate open.", ids=("EV404",))], "cites no valid inventory event"),
             ([_beat("He shouts “I will find you” at the crate.")], "quotes words not found"),
             ([_beat("Don't forget to subscribe for more recaps.")], "channel filler"),
             ([_beat("He pries the crate open and.")], "unfinished phrase"),
             ([_beat("", kind="dialogue", speech="D9")], "not an approved dialogue clip"),
             ([_beat("He pries the crate open with both hands.")] * 1, "measured voice fits")]
    for beats, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_beats(beats, INVENTORY, POOL, allowed_quotes=["D1"],
                           budget_words=(20, 30) if message == "measured voice fits" else None)
    quoted = validate_beats([_beat("He warns them, “Drop the gun now.”")], INVENTORY, POOL, allowed_quotes=[],
                            budget_words=None)
    assert quoted[0]["text"].startswith("He warns")


def test_matching_rejects_non_candidate_and_out_of_bounds_shots():
    candidates = [{"shot_id": "S1", "start": 6.0, "end": 12.0, "event_id": "EV0002"}]
    match = {"verdict": "supported", "shots": [
        {"shot_id": "S9", "start": 0, "end": 3, "anchor": 1, "shows": "x", "reason": "x", "relevance": 0.9, "focus_x": 0.5},
        {"shot_id": "S1", "start": 2, "end": 9, "anchor": 7, "shows": "x", "reason": "x", "relevance": 0.9, "focus_x": 0.5},
        {"shot_id": "S1", "start": 7, "end": 10, "anchor": 8, "shows": "x", "reason": "x", "relevance": 0.3, "focus_x": 0.5}]}
    result = validate_choices("N01.1", match, candidates)
    assert result["verdict"] == "unsupported" and len(result["validation_notes"]) == 3


def _choice(shot, a, b, relevance=0.9, anchor=None):
    return {"shot_id": shot, "start": a, "end": b, "anchor": anchor if anchor is not None else (a + b) / 2,
            "focus_x": 0.5, "shows": "s", "reason": "r", "relevance": relevance, "event_id": "EV0001",
            "shot_bounds": [a - 1, b + 1]}


def test_shot_planning_never_reuses_footage_and_reserves_dialogue():
    beats = [{"id": "D01", "kind": "dialogue", "shots": [{"start": 20.0, "end": 26.0}]},
             {"id": "N01", "kind": "narration", "text": "a b c d e f. g h i j k l.",
              "sentences": [{"sentence_id": "N01.1", "text": "One two three four five six.", "event_ids": ["EV0001"]},
                            {"sentence_id": "N01.2", "text": "Seven eight nine ten eleven twelve.", "event_ids": ["EV0002"]}]}]
    matches = {"N01.1": {"verdict": "supported", "shots": [_choice("S1", 1.0, 3.0)]},
               "N01.2": {"verdict": "supported", "shots": [_choice("S1", 1.0, 3.0), _choice("S4", 21.0, 25.0)]}}
    placed, evidence, shortfalls = plan_shots(beats, matches, rate=2.5)
    narration = placed[1]["shots"]
    # Extended within its own shot only as far as the measured need (+20% headroom) requires.
    assert narration[0]["end"] == 4.0 and narration[0]["start"] == pytest.approx(0.94)
    assert all(s["sentence_index"] == 0 for s in narration)  # Reused and dialogue-reserved footage is refused.
    assert [s["sentence_id"] for s in shortfalls] == ["N01.2"] and evidence[1]["verdict"] == "unsupported"


def test_sentence_segmented_allocation_cuts_on_measured_sentence_boundaries():
    beat = {"id": "N01", "duration": 4.0, "sentences": [{"text": "One two."}, {"text": "Three four."}],
            "words": [{"word": "One", "start": 0.1, "end": 0.6}, {"word": "two.", "start": 0.7, "end": 1.2},
                      {"word": "Three", "start": 2.0, "end": 2.5}, {"word": "four.", "start": 2.6, "end": 3.8}],
            "shots": [{"start": 0, "end": 5, "sentence_index": 0, "anchor": 4.0},
                      {"start": 10, "end": 14, "sentence_index": 1}]}
    spans = sentence_spans(beat)
    assert spans[0] == (0.0, pytest.approx(1.88)) and spans[1][1] == 4.0
    shots = allocate_segmented(beat, 20.0, 25)
    assert shots[0]["frames"] == 47 and sum(s["frames"] for s in shots) == 100
    assert shots[0]["start"] <= 4.0 <= shots[0]["end"]  # The anchored action survives trimming.
    centred = allocate_shots([{"start": 0, "end": 5}], 1.0, 20.0, 25)[0]
    assert centred["start"] == pytest.approx(2.0)


def test_deterministic_review_flags_repeats_missing_footage_names_and_overflow():
    beats = [{"id": "N01", "kind": "narration", "duration": 25.0, "text": "x",
              "sentences": [{"sentence_id": "N01.1", "text": "Then Viktor grabs it.", "event_ids": ["EV0001"]},
                            {"sentence_id": "N01.2", "text": "He runs away fast.", "event_ids": ["EV0002"]}],
              "shots": [{"start": 1, "end": 5, "sentence_index": 0, "sentence_id": "N01.1", "relevance": 0.9},
                        {"start": 2, "end": 6, "sentence_index": 0, "sentence_id": "N01.1b", "relevance": 0.5}]}]
    issues = deterministic_checks(beats, inventory=INVENTORY, speech_pool=POOL, source_duration=30, fps=30, target=20,
                                  notes="", transcript="", hook_event_id="EV0001")
    checks = {(i["check"], i["severity"]) for i in issues}
    assert ("repetition", "major") in checks and ("shot_mismatch", "critical") in checks
    assert ("unsupported_identity_or_dialogue", "major") in checks and ("pacing", "critical") in checks


def test_request_config_validates_gemini_options():
    request = request_config({"title": "A", "source": "x"})
    assert request["planner"] == "gemini" and request["gemini_preset"] == "free"
    for bad in [{"gemini_preset": "ultra"}, {"review_passes": 9}, {"max_budget_usd": 0}, {"planner": "gpt"},
                {"analysis_quality": "8k"}, {"fallback_writer": "llama"}]:
        with pytest.raises(ValueError):
            request_config({"title": "A", "source": "x", **bad})
    assert request_config({"title": "A", "source": "x", "planner": "local", "gemini_preset": "ultra"})


def test_forced_alignment_result_is_rejected_when_it_disagrees_with_asr():
    from shorts_pipeline.alignment import _romanize, check_against_estimate
    estimate = [{"word": "a", "start": 0.1, "end": 0.3}, {"word": "b", "start": 0.4, "end": 0.6}]
    assert check_against_estimate([{"word": "a", "start": 0.12, "end": 0.3}, {"word": "b", "start": 0.45, "end": 0.6}], estimate)
    assert not check_against_estimate([{"word": "a", "start": 2.0, "end": 2.3}, {"word": "b", "start": 2.5, "end": 2.6}], estimate)
    assert not check_against_estimate([{"word": "a", "start": 0.5, "end": 0.6}, {"word": "b", "start": 0.1, "end": 0.2}], estimate)
    assert _romanize("Twenty-one,") == "twentyone" and _romanize("$14") == "fourteen"


def test_final_verification_measures_the_actual_file_and_flags_defects(tmp_path):
    import subprocess
    from shorts_pipeline.gemini.verify import verify_render
    video = tmp_path / "final.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x160:rate=30:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    timeline = {"beats": [{"id": "N01", "kind": "narration", "duration": 3.0, "timeline_start": 0,
                           "words": [{"word": "x", "start": 0.1, "end": 0.5}],
                           "render_shots": [{"start": 1.0, "end": 2.0, "sentence_id": "N01.1"},
                                            {"start": 1.2, "end": 2.2, "sentence_id": "N01.2"},
                                            {"start": 28.0, "end": 31.0, "sentence_id": "N01.3"}]}]}
    report = verify_render(video, timeline, tmp_path, width=320, height=320, fps=30, target=20, source_duration=30)
    checks = {(i["check"], i["severity"]) for i in report["issues"]}
    assert ("dimensions", "critical") in checks and ("repeated_footage", "major") in checks
    assert ("source_range", "critical") in checks and report["automated_verdict"] == "failed"
    assert report["production_ready"] is False and report["checks"]["container"]["frames"] == 90
    assert report["checks"]["loudness"]["integrated_lufs"] is not None


def test_free_tier_paces_requests_and_does_not_enforce_a_dollar_budget(tmp_path):
    waits = []
    settings = settings_from_request({"gemini_preset": "free"})
    assert settings.free_tier and settings.requests_per_minute == 8 and not settings.explicit_cache
    assert settings.analysis["zoom_checks"] == 6 and settings.max_retries == 6
    ledger = CostLedger(0.0001, tmp_path / "cost.json", free_tier=True)
    client = GeminiClient(settings, ledger, tmp_path / "cache", Scripted([REVIEW, {"verdict": "fail", "issues": []}]),
                          sleep=waits.append)
    client.generate("review", QualityReview, [text_part("x" * 40000)], system="s")
    client.generate("review", QualityReview, [text_part("y")], system="s")
    assert waits and 7.0 < waits[-1] <= 7.5  # 60 s / 8 requests per minute
    report = json.loads((tmp_path / "cost.json").read_text())
    assert "free tier: not billed" in report["billing"] and report["requests_sent"] == 2


def test_free_tier_request_cap_protects_the_daily_quota(tmp_path):
    from shorts_pipeline.gemini.client import RequestLimitExceeded
    client = make_client(tmp_path, Scripted([REVIEW, REVIEW]), gemini_preset="free", max_requests=1)
    client.generate("review", QualityReview, [text_part("a")], system="s")
    with pytest.raises(RequestLimitExceeded, match="rerun to resume"):
        client.generate("review", QualityReview, [text_part("b")], system="s")


def test_zero_free_quota_falls_back_to_the_next_available_model(tmp_path):
    transport = Scripted([FakeAPIError("rate_limit", "Quota exceeded for metric free_tier_requests, limit: 0, "
                                                     "model: gemini-3.8-flash"), REVIEW],
                         models=["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"])
    client = make_client(tmp_path, transport, gemini_preset="free")
    client.resolve_models()
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    assert client.settings.resolved_models["review"] == "gemini-2.5-flash"
    assert any(e["kind"] == "model_substitution" and e["used"] == "gemini-2.5-flash" for e in client.ledger.events)
    exhausted = make_client(tmp_path / "b", Scripted([FakeAPIError("rate_limit", "limit: 0")] * 3,
                                                     models=["gemini-2.5-flash-lite"]), gemini_preset="free")
    exhausted.resolve_models()
    with pytest.raises(GeminiModelUnavailable, match="No further review model candidates"):
        exhausted.generate("review", QualityReview, [text_part("x")], system="s")


def test_daily_quota_switches_model_and_stops_only_when_all_are_used_up(tmp_path):
    from shorts_pipeline.gemini.client import GeminiQuotaExhausted
    daily = "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    transport = Scripted([FakeAPIError("rate_limit", daily), REVIEW, REVIEW],
                         models=["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash-lite"])
    client = make_client(tmp_path, transport, gemini_preset="free")
    client.resolve_models()
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    assert client.settings.resolved_models["review"] == "gemini-3.5-flash"
    assert transport.calls == 2  # No retries are wasted on an exhausted daily quota.
    # A later run the same Pacific day skips the exhausted model without spending a request on it.
    again = make_client(tmp_path, Scripted([REVIEW], models=transport.models), gemini_preset="free")
    again.resolve_models()
    again.generate("planning", QualityReview, [text_part("y")], system="s")
    assert again.settings.resolved_models["planning"] == "gemini-3.5-flash" and again.transport.calls == 1
    only = make_client(tmp_path / "b", Scripted([FakeAPIError("rate_limit", daily)], models=["gemini-2.5-flash-lite"]),
                       gemini_preset="free")
    only.resolve_models()
    with pytest.raises(GeminiQuotaExhausted, match="rerun after the quota resets"):
        only.generate("review", QualityReview, [text_part("x")], system="s")
    waits = []
    client = make_client(tmp_path / "c", Scripted([FakeAPIError("rate_limit", "Please retry in 37.5s."), REVIEW]))
    client.sleep = waits.append
    client.generate("review", QualityReview, [text_part("x")], system="s")
    assert max(waits) >= 38.5


def test_truncated_output_gets_more_room_instead_of_a_repair_prompt(tmp_path):
    class Truncating(Scripted):
        def generate(self, model, parts, **kwargs):
            self.calls += 1
            self.limits = getattr(self, "limits", []) + [kwargs["max_output_tokens"]]
            finish = "MAX_TOKENS" if self.calls == 1 else "STOP"
            return {"text": "{" if finish == "MAX_TOKENS" else json.dumps(REVIEW), "finish_reason": finish,
                    "prompt_tokens": 10, "cached_tokens": 0, "output_tokens": 8192, "thinking_tokens": 0, "audio_tokens": 0}
    transport = Truncating()
    client = make_client(tmp_path, transport)
    assert client.generate("review", QualityReview, [text_part("x")], system="s", max_output_tokens=8192).verdict == "pass"
    assert transport.limits == [8192, 16384]
    assert not any("Your previous response" in str(p) for p in transport.limits)


CLIPS = [{"clip_id": f"K{i:03d}", "event_id": f"EV000{i}", "seconds": 4.0, "shot_id": f"S{i}", "start": i * 6.0,
          "end": i * 6.0 + 4, "anchor": i * 6.0 + 2, "focus_x": 0.5, "shows": "he lifts a rifle", "role": "action",
          "quality": 0.9, "shot_bounds": [i * 6.0, i * 6.0 + 6]} for i in (1, 2, 3)]


def test_picture_script_validation_enforces_clip_order_capacity_and_key_word():
    from shorts_pipeline.gemini.picture import validate_picture
    from shorts_pipeline.gemini.schemas import PictureScript
    ok = PictureScript(lines=[{"clip_ids": ["K001"], "text": "He lifts the rifle from the crate.", "key_word": "rifle"},
                              {"clip_ids": ["K002"], "text": "Then he aims it at the door.", "key_word": "aims"}])
    lines = validate_picture(ok, CLIPS, rate=2.5, budget_high=20)
    assert [l["line_id"] for l in lines] == ["L01", "L02"] and lines[0]["event_ids"] == ["EV0001"]
    bad = [([{"clip_ids": ["K001", "K003"], "text": "He lifts the rifle.", "key_word": "rifle"}], "not consecutive"),
           ([{"clip_ids": ["K002"], "text": "Then he aims carefully.", "key_word": "aims"},
             {"clip_ids": ["K001"], "text": "He lifts the rifle.", "key_word": "rifle"}], "goes back"),
           ([{"clip_ids": ["K001"], "text": "He slowly lifts the heavy rusted rifle out of the old wooden crate in the dark and silent barn.",
              "key_word": "rifle"}], "fit"),
           ([{"clip_ids": ["K009"], "text": "He lifts the rifle.", "key_word": "rifle"}], "known clip")]
    for lines_, message in bad:
        with pytest.raises(ValueError, match=message):
            validate_picture(PictureScript(lines=lines_), CLIPS, rate=2.5, budget_high=10)


def test_merge_keeps_action_clips_and_limits_context():
    from shorts_pipeline.gemini.picture import validate_reconciled
    from shorts_pipeline.gemini.schemas import ReconciledScript
    a_lines = [{"line_id": "L01", "clip_ids": ["K001"], "text": "He lifts the rifle.", "key_word": "rifle",
                "event_ids": ["EV0001"]},
               {"line_id": "L02", "clip_ids": ["K002"], "text": "He aims at the door.", "key_word": "aims",
                "event_ids": ["EV0002"]}]
    story_map = {"events": [{"event_id": "EV0003", "inferred_context": [{"claim": "x"}]}]}
    kwargs = dict(lines=a_lines, clips=CLIPS, story_map=story_map, inventory=INVENTORY, speech_pool=POOL,
                  allowed_quotes=[], notes="", rate=2.5, budget_words=None)

    def merged(*lines):
        return ReconciledScript(title="t", description="d", beats=[{"beat_id": "b", "kind": "narration", "lines": list(lines)}])
    action1 = {"kind": "action", "text": "He lifts the rifle from the crate.", "source_line_id": "L01", "clip_ids": ["K001"], "key_word": "rifle"}
    action2 = {"kind": "action", "text": "Then he aims at the door.", "source_line_id": "L02", "clip_ids": ["K002"], "key_word": "aims"}
    context = {"kind": "context", "text": "The truck waits outside.", "clip_ids": ["K003"], "event_ids": ["EV0003"],
               "basis": "inferred_context"}
    beats, matches = validate_reconciled(merged(action1, action2, context), **kwargs)
    kinds = [s["line_kind"] for s in beats[0]["sentences"]]
    assert kinds == ["action", "action", "context"]
    assert matches["N01.1"]["shots"][0]["key_clip"] and matches["N01.3"]["shots"][0]["cover"]
    cases = [((dict(action1, clip_ids=["K002"]), action2), "keep its clips"),
             ((action1, dict(context, clip_ids=["K001"]), action2), "unused cover clips"),
             ((action1, dict(context, basis="story_notes"), action2), "no"),
             ((action1, dict(context, event_ids=["EV0001"]), action2), "inferred_context without"),
             ((action1,), "backbone"),
             ((dict(action1, text="He lifts the rifle."), dict(action2, text="Then he aims."),
               dict(context, text="The old truck waits outside the barn tonight.")), "action-led")]
    for lines, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_reconciled(merged(*lines), **kwargs)


def test_key_word_lands_on_the_action_anchor_frame():
    beat = {"id": "N01", "duration": 3.0, "sentences": [{"text": "He grabs the rifle.", "key_word": "rifle"}],
            "words": [{"word": "He", "start": 0.1, "end": 0.3}, {"word": "grabs", "start": 0.35, "end": 0.7},
                      {"word": "the", "start": 0.75, "end": 0.9}, {"word": "rifle.", "start": 1.2, "end": 1.8}],
            "shots": [{"start": 10.0, "end": 16.0, "anchor": 14.0, "sentence_index": 0, "key_clip": True}]}
    from shorts_pipeline.gemini.picture import apply_key_times
    beat["kind"] = "narration"
    assert apply_key_times([beat]) == 1 and beat["shots"][0]["key_time"] == pytest.approx(1.2)
    shot = allocate_segmented(beat, 30.0, 25)[0]
    assert shot["key_word_aligned"] and shot["start"] + 1.2 == pytest.approx(14.0)


def test_overloaded_model_switches_to_the_next_candidate(tmp_path):
    from shorts_pipeline.gemini.client import GeminiOverloaded
    busy = FakeAPIError("server", "503 UNAVAILABLE: This model is currently experiencing high demand.")
    transport = Scripted([busy] * 7 + [REVIEW], models=["gemini-3.8-flash", "gemini-2.5-flash"])
    client = make_client(tmp_path, transport, gemini_preset="free")
    client.resolve_models()
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    assert client.settings.resolved_models["review"] == "gemini-2.5-flash"
    stuck = make_client(tmp_path / "b", Scripted([busy] * 7, models=["gemini-2.5-flash"]), gemini_preset="free")
    stuck.resolve_models()
    stuck.settings.overload_wait_seconds = 0  # Once the overload wait budget is spent, fail clearly.
    with pytest.raises(GeminiOverloaded, match="rerun later"):
        stuck.generate("review", QualityReview, [text_part("x")], system="s")


def test_cached_answer_from_another_stage_candidate_is_reused_after_a_switch(tmp_path):
    first = make_client(tmp_path, Scripted([REVIEW], models=["gemini-3.8-flash", "gemini-2.5-flash"]), gemini_preset="free")
    first.resolve_models()
    first.generate("review", QualityReview, [text_part("same request")], system="s")
    later = make_client(tmp_path, Scripted([], models=["gemini-2.5-flash"]), gemini_preset="free")
    later.resolve_models()  # 3.8 is gone (e.g. exhausted); the stage now resolves to 2.5-flash.
    assert later.settings.resolved_models["review"] == "gemini-2.5-flash"
    assert later.generate("review", QualityReview, [text_part("same request")], system="s").verdict == "pass"
    assert later.transport.calls == 0 and later.ledger.calls[-1]["response_cache_hit"]


def test_clips_cover_their_whole_verified_event_without_overlap():
    from shorts_pipeline.gemini.picture import cover_events
    event = {"id": "E1", "start": 10.0, "end": 26.0, "visible_action": "fight", "peak": 15}
    cands = [{"shot_id": "S1", "start": 8.0, "end": 14.0}, {"shot_id": "S2", "start": 14.0, "end": 22.0},
             {"shot_id": "S3", "start": 22.0, "end": 30.0}]
    clips = [{"shot_id": "S1", "start": 11.0, "end": 12.0, "shot_bounds": [8, 14], "event_id": "E1", "anchor": 11.5},
             {"shot_id": "S2", "start": 16.0, "end": 17.0, "shot_bounds": [14, 22], "event_id": "E1", "anchor": 16.5}]
    notes = []
    out = cover_events(clips, [(event, cands)], notes)
    spans = [(c["start"], c["end"]) for c in out]
    assert spans == [(9.7, 14.0), (14.0, 22.0), (22.0, 26.0)]  # Whole event, padded, no overlaps.
    assert out[-1]["covers_event_remainder"] and "extended clips" in notes[0]
    assert all(c["start"] >= c["shot_bounds"][0] and c["end"] <= c["shot_bounds"][1] for c in out)


def test_factual_slips_from_the_reviewer_are_escalated_to_actionable(tmp_path):
    from shorts_pipeline.gemini.review import gemini_review
    review = {"verdict": "revise", "issues": [
        {"check": "unsupported_statement", "severity": "minor", "beat_id": "N01", "sentence_id": "N01.1",
         "description": "calls the gold coins markers", "fix": "rewrite_beat"},
        {"check": "pacing", "severity": "minor", "description": "a little slow"}]}
    client = make_client(tmp_path, Scripted([review]))
    beats = [{"id": "N01", "kind": "narration", "text": "x", "sentences": [
        {"sentence_id": "N01.1", "text": "He takes the markers.", "event_ids": ["EV0001"], "visual_requirement": ""}],
        "shots": [{"start": 1, "end": 3, "sentence_id": "N01.1"}]}]
    issues, _ = gemini_review(client, UploadedFile("f", "u", "video/mp4", "h"), beats, inventory=INVENTORY,
                              story_map={"events": []}, editorial={"selected": []}, log=lambda *_: None)
    severities = {i["check"]: i["severity"] for i in issues}
    assert severities == {"unsupported_statement": "major", "pacing": "minor"}


def test_a_missing_key_word_disables_alignment_instead_of_rejecting_the_script():
    """Run 43162288647: one line naming 'dog' without saying it discarded the whole footage-first script."""
    from shorts_pipeline.gemini.picture import validate_picture
    from shorts_pipeline.gemini.schemas import PictureScript
    script = PictureScript(lines=[{"clip_ids": ["K001"], "text": "He lifts the rifle from the crate.", "key_word": "dog"}])
    lines = validate_picture(script, CLIPS, rate=2.5, budget_high=10)
    assert lines[0]["key_word"] == "" and lines[0]["clip_ids"] == ["K001"]


def test_overloaded_model_is_abandoned_after_three_tries_for_the_next_candidate(tmp_path):
    # Run 9: 503s were retried 7 times per model and one call took 31 minutes.
    transport = Scripted([FakeAPIError("server", "503 UNAVAILABLE")] * 3 + [REVIEW])
    client = make_client(tmp_path, transport)
    first, second = client.settings.model_preferences["review"][:2]
    client.settings.model_preferences["review"] = [first, second]
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    assert transport.calls == 4 and client.settings.resolved_models["review"] == second
    assert any(n.get("kind") == "model_substitution" for n in client.ledger.events)


def test_footage_fit_fix_adds_event_footage_instead_of_rewriting(tmp_path):
    # Run 9: every beat flagged "Only Xs of footage" was rewritten, cutting 259 words to 159.
    from shorts_pipeline.gemini.correction import _fix

    class StubRun:
        def __init__(self):
            self.inventory = {"events": [{"id": "EV1", "start": 10.0, "end": 16.0, "peak": 12.0,
                                          "visible_action": "he runs"}]}
            self.scenes = [{"id": "S1", "start": 9.0, "end": 12.0}, {"id": "S2", "start": 12.0, "end": 15.0}]
            self.matches = {"N01.1": {"sentence_id": "N01.1", "verdict": "supported",
                                      "shots": [{"shot_id": "S1", "start": 9.0, "end": 12.0}]}}
            self.rewrites, self.rate = [], 2.5

        def rewrite(self, beats, beat_id, reason, **limits):
            self.rewrites.append(limits)
            return beats

    run = StubRun()
    beat = {"id": "N01", "kind": "narration", "text": "He runs through the rain to the hotel door now",
            "sentences": [{"sentence_id": "N01.1", "text": "He runs.", "event_ids": ["EV1"]}]}
    issue = {"check": "narration_footage_fit", "severity": "critical", "fix": "retime", "beat_id": "N01",
             "sentence_id": "", "description": "Only 3.0s of selected footage for 5.0s narration"}
    _, changed, action = _fix(run, [beat], issue)
    assert changed == "N01" and "S2" in action and not run.rewrites
    assert [s["shot_id"] for s in run.matches["N01.1"]["shots"]] == ["S1", "S2"]
    # With no further verified footage, it shortens by at most a quarter.
    _, changed, action = _fix(run, [beat], issue)
    assert run.rewrites == [{"max_words": 8, "min_words": 6}]
    # A content fix keeps the beat's length.
    run.rewrites.clear()
    _fix(run, [beat], {**issue, "check": "unsupported_identity_or_dialogue", "fix": "rewrite_beat",
                       "sentence_id": "N01.1"})
    assert run.rewrites == [{"max_words": 16, "min_words": 8}]


class _RecordingRun:
    def __init__(self, events=(), scenes=()):
        self.history, self.rate = [], 2.5
        self.inventory = {"events": list(events)}
        self.scenes = list(scenes)

    def record(self, **entry):
        self.history.append(entry)


def test_long_event_fallback_takes_only_the_shots_nearest_its_peak():
    # Run 10: one 53 s taxi-and-library event put 23 library shots under a taxi sentence.
    from shorts_pipeline.gemini.pipeline import _event_footage

    scenes = [{"id": f"S{i}", "start": float(i * 2), "end": float(i * 2 + 2)} for i in range(30)]
    run = _RecordingRun([{"id": "EV1", "start": 0.0, "end": 60.0, "peak": 11.0, "visible_action": "taxi"}], scenes)
    sentence = {"sentence_id": "N1.1", "text": "He gets in the taxi.", "event_ids": ["EV1"]}
    assert len(_event_footage(run, sentence)) == 30
    picked = _event_footage(run, sentence, need=3.0, known={(10.0, 12.0)})
    assert [s["shot_id"] for s in picked] == ["S4", "S6"]


def test_a_later_beat_retelling_the_same_event_is_dropped():
    from shorts_pipeline.gemini.pipeline import _drop_repeats

    def beat(ident, *sentences):
        return {"id": ident, "kind": "narration", "text": " ".join(t for t, _ in sentences), "duration": 3.0,
                "sentences": [{"sentence_id": f"{ident}.{i}", "text": t, "event_ids": e}
                              for i, (t, e) in enumerate(sentences, 1)]}

    beats = [beat("N03", ("Seeking a way out, he climbs into a waiting yellow taxi cab with his dog and hands "
                          "the driver a gold coin.", ["EV12"]),
                  ("John speaks urgently through the partition as the car pulls away.", ["EV12"])),
             beat("N04", ("John climbs into the yellow taxi cab with his dog, seeking refuge from the streets.", ["EV12"]),
                  ("He slides a gold coin through the partition to the driver.", ["EV12"]),
                  ("John stays vigilant, watching passing lights on the ride toward an uncertain destination.", ["EV12"]))]
    run = _RecordingRun()
    result = _drop_repeats(run, beats)
    assert [s["sentence_id"] for b in result for s in b["sentences"]] == ["N03.1", "N03.2", "N04.3"]
    assert "duration" not in result[1] and run.history[0]["action"] == "remove_repeated_sentence"


def test_looping_response_is_reasked_then_handed_to_the_next_model(tmp_path):
    # Run 11: gemini-3.1-flash-lite wrote a valid script start, then "1234567890..." until the
    # 65,536-token limit, twice; raising the limit only bought a longer loop.
    from shorts_pipeline.gemini.client import GeminiDegenerateOutput, _looping

    loop = '{"title": "Everyone Wants Him Dead", "beats": [{"text": "John Wick ' + "1234567890" * 200
    assert _looping(loop) and not _looping(json.dumps({"beats": [{"text": "x " * 40, "n": i} for i in range(80)]}, indent=2))

    class Looping(Scripted):
        def generate(self, model, parts, **kwargs):
            self.calls += 1
            self.seen = getattr(self, "seen", []) + [(model, kwargs["max_output_tokens"], kwargs["temperature"])]
            item = self.results.pop(0)
            return {"text": item if isinstance(item, str) else json.dumps(item),
                    "finish_reason": "MAX_TOKENS" if item is loop else "STOP", "prompt_tokens": 10,
                    "cached_tokens": 0, "output_tokens": 10, "thinking_tokens": 0, "audio_tokens": 0}

    transport = Looping([loop, loop, REVIEW])
    client = make_client(tmp_path, transport)
    first, second = client.settings.model_preferences["review"][:2]
    client.settings.model_preferences["review"] = [first, second]
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    (m1, limit1, t1), (m2, limit2, t2), (m3, _, _) = transport.seen
    assert m1 == m2 == first and limit2 == limit1 and t2 > t1 and m3 == second  # no limit doubling
    transport = Looping([loop, loop])
    client = make_client(tmp_path / "b", transport)
    client.settings.model_preferences["review"] = [first]
    with pytest.raises(GeminiDegenerateOutput):
        client.generate("review", QualityReview, [text_part("x")], system="s")


def test_extra_keys_take_over_a_model_whose_daily_quota_is_used_up(tmp_path, monkeypatch):
    from shorts_pipeline.gemini.client import _redact, api_keys, video_part

    monkeypatch.setenv("GEMINI_API_KEY", "primary-key-000000")
    monkeypatch.setenv("GEMINI_API_KEY_2", "second-key-1111111")
    monkeypatch.setenv("GEMINI_API_KEY_3", "")
    assert api_keys() == ["primary-key-000000", "second-key-1111111"]
    assert "second-key" not in _redact("bad key second-key-1111111")

    daily = "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"

    class Keyed(Scripted):
        def __init__(self, label, results):
            super().__init__(results)
            self.label, self.seen = label, []

        def generate(self, model, parts, **kwargs):
            self.seen.append((model, [p.get("file_uri") for p in parts if "file_uri" in p]))
            return super().generate(model, parts, **kwargs)

        def upload(self, path, mime):
            self.uploaded += 1
            return {"name": f"files/{self.label}", "uri": f"uri-{self.label}", "mime_type": mime, "state": "ACTIVE",
                    "expires_epoch": 4e9}

    first, second = Keyed("k1", [FakeAPIError("rate_limit", daily)]), Keyed("k2", [REVIEW, REVIEW])
    settings = settings_from_request({"gemini_preset": "free"})
    client = GeminiClient(settings, CostLedger(5.0, tmp_path / "cost.json", free_tier=True), tmp_path / "cache",
                          first, sleep=lambda s: None, transports=[second])
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0" * 100)
    part = video_part(client.upload(video))
    model = settings.model_preferences["review"][0]
    client.generate("review", QualityReview, [part, text_part("x")], system="s")
    # Same model, second key, with the footage uploaded again under the second key's project.
    assert first.seen[0] == (model, ["uri-k1"]) and second.seen[0] == (model, ["uri-k2"])
    assert client.key_index == 1 and second.uploaded == 1
    assert any(e["kind"] == "api_key_switch" and e["to_key"] == 2 for e in client.ledger.events)
    assert model in client._exhausted_today(0) and model not in client._exhausted_today(1)
    # A later call starts on the second key and reuses its upload.
    client.generate("review", QualityReview, [part, text_part("y")], system="s")
    assert second.uploaded == 1 and second.seen[1] == (model, ["uri-k2"])
    # Only when every key has used up a model does the stage move to the next model.
    client._mark_exhausted(model)
    assert client._key_with_quota(model) is None and client._fallbacks("review", model)


def test_invented_dialogue_is_dropped_when_no_dialogue_clip_is_approved():
    # Run 12: the source has voice-over narration, so retained dialogue is disabled; flash-lite
    # still wrote a dialogue beat "S_ELDER_001" and the draft failed validation twice.
    beats = [_beat("He pries the crate open with both hands."), _beat("", kind="dialogue", speech="S_ELDER_001")]
    result = validate_beats(beats, INVENTORY, {}, allowed_quotes=[], budget_words=None)
    assert [b["kind"] for b in result] == ["narration"]
    with pytest.raises(ValueError, match="not an approved dialogue clip"):
        validate_beats(beats, INVENTORY, POOL, allowed_quotes=["D1"], budget_words=None)


def test_when_every_model_is_overloaded_the_run_waits_and_starts_again_from_the_best(tmp_path):
    # Run 13: 503s on every candidate walked the stage to the last model (a 404) and the run failed.
    waits = []
    busy = [FakeAPIError("server", "503 UNAVAILABLE high demand")]
    client = make_client(tmp_path, Scripted(busy * 3 + busy * 5 + [REVIEW]))
    client.sleep = waits.append
    first, second = client.settings.model_preferences["review"][:2]
    client.settings.model_preferences["review"] = [first, second]
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    cycle = [e for e in client.ledger.events if e["kind"] == "overload_cycle"]
    assert len(cycle) == 1 and cycle[0]["restart_with"] == first and 30.0 in waits
    assert client.settings.resolved_models["review"] == first


def test_a_not_found_model_is_skipped_for_the_rest_of_the_run(tmp_path):
    busy = FakeAPIError("server", "503 UNAVAILABLE high demand")
    gone = FakeAPIError("model", "404 NOT_FOUND no longer available to new users")
    transport = Scripted([gone] + [busy] * 5 + [REVIEW])
    client = make_client(tmp_path, transport)
    seen = []
    original = transport.generate
    transport.generate = lambda model, parts, **kw: (seen.append(model), original(model, parts, **kw))[1]
    first, second = client.settings.model_preferences["review"][:2]
    client.settings.model_preferences["review"] = [first, second]
    client.generate("review", QualityReview, [text_part("x")], system="s")
    assert seen[0] == first and set(seen[1:]) == {second}  # The 404 model is never retried after the wait.
    # With no live model left, the overload still fails clearly instead of waiting forever.
    client = make_client(tmp_path / "b", Scripted([busy] * 20))
    client.settings.model_preferences["review"] = [first]
    client.settings.overload_wait_seconds = 0
    with pytest.raises(GeminiOverloaded):
        client.generate("review", QualityReview, [text_part("x")], system="s")


def test_a_rematch_that_finds_nothing_keeps_the_sentence_and_its_shots():
    # Run 14: re-matching the bounty and Winston sentences while excluding their shots found
    # nothing, so both sentences were deleted and the Short fell to 27 s.
    from shorts_pipeline.gemini.correction import _fix

    class StubRun:
        def __init__(self):
            self.matches = {"N03.1": {"sentence_id": "N03.1", "verdict": "supported",
                                      "shots": [{"shot_id": "S0018", "start": 37.0, "end": 40.5}]}}

        def rematch(self, beats, ids, **kwargs):
            for sid in ids:
                self.matches[sid] = {"sentence_id": sid, "verdict": "unsupported", "shots": []}

    run = StubRun()
    beat = {"id": "N03", "kind": "narration", "text": "A bounty goes live.",
            "sentences": [{"sentence_id": "N03.1", "text": "A bounty goes live.", "event_ids": ["EV5"]}],
            "shots": [{"shot_id": "S0018", "sentence_id": "N03.1"}]}
    issue = {"check": "shot_mismatch", "severity": "major", "fix": "rematch_shot", "beat_id": "N03",
             "sentence_id": "N03.1", "description": "footage shows a fight"}
    _, changed, action = _fix(run, [beat], issue)
    assert changed is None and "kept" in action and run.matches["N03.1"]["shots"][0]["shot_id"] == "S0018"


def test_one_stalled_request_cannot_hold_the_run(tmp_path, monkeypatch):
    # Run 15 ran for hours with no output; one request retried on timeouts can now spend at most
    # max_call_seconds before the stage moves to the next model.
    import shorts_pipeline.gemini.client as client_module

    clock = [0.0]
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])

    class Slow(Scripted):
        def generate(self, model, parts, **kwargs):
            clock[0] += 400.0  # Each attempt times out after a long wait.
            return super().generate(model, parts, **kwargs)

    deadline = FakeAPIError("timeout", "504 DEADLINE_EXCEEDED")
    transport = Slow([deadline, deadline, REVIEW])
    client = make_client(tmp_path, transport)
    first, second = client.settings.model_preferences["review"][:2]
    client.settings.model_preferences["review"] = [first, second]
    assert client.generate("review", QualityReview, [text_part("x")], system="s").verdict == "pass"
    assert transport.calls == 3 and client.settings.resolved_models["review"] == second


def test_a_sentence_never_gets_the_same_shot_twice():
    # Run 15: a sentence's match and its event-footage fallback both named S0028 and S0032,
    # so the same footage played twice back to back.
    beats = [{"id": "N01", "kind": "narration", "text": "x",
              "sentences": [{"sentence_id": "N01.1", "text": "One two three four five six seven eight nine ten.",
                             "event_ids": ["EV1"]}]}]
    matches = {"N01.1": {"verdict": "supported",
                         "shots": [_choice("S28", 62.183, 64.817), {**_choice("S28", 62.183, 64.817), "focus_x": 0.65},
                                   _choice("S29", 64.817, 67.833)]}}
    placed, _, _ = plan_shots(beats, matches, rate=2.5)
    ids = [s["shot_id"] for s in placed[0]["shots"]]
    assert ids.count("S28") == 1 and "S29" in ids


def test_shot_encode_bounds_the_source_audio_and_media_commands_time_out(monkeypatch, tmp_path):
    # Run 15: the shot at 74.283 s never finished because the source audio branch never hit EOF
    # inside amix; the job sat for 78 minutes.
    import subprocess
    from shorts_pipeline import render

    seen = {}
    monkeypatch.setattr(render, "_run", lambda args: seen.setdefault("args", args))
    shot = {"start": 74.283, "end": 75.35, "duration": 1.0666667, "speed": 1.0, "frames": 64, "focus_x": 0.5}
    render._render_shot(tmp_path / "s.mp4", shot, tmp_path / "o.mkv", 1080, 1080, 60, True)
    graph = seen["args"][seen["args"].index("-filter_complex") + 1]
    assert "aformat=channel_layouts=stereo,atrim=duration=1.066666700[source]" in graph

    monkeypatch.undo()
    def stuck(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))
    monkeypatch.setattr(render.subprocess, "run", stuck)
    with pytest.raises(RuntimeError, match="did not finish"):
        render._run(["ffmpeg", "-i", "x"])
