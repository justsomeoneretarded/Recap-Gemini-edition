"""Unit tests of Gemini interfaces and evidence validation (offline; no API calls)."""
import json

import pytest

from shorts_pipeline.gemini import client as client_module
from shorts_pipeline.gemini.analysis import (_fix_relative_times, dialogue_support, plan_windows,
                                             validate_events, verify_events)
from shorts_pipeline.gemini.client import (BudgetExceeded, CostLedger, GeminiAuthError, GeminiClient,
                                           GeminiModelUnavailable, GeminiOutputError, GeminiRateLimited,
                                           GeminiTimeout, UploadedFile, text_part)
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
    assert request["planner"] == "gemini" and request["gemini_preset"] == "balanced"
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
