"""Story-first production for discovered footage: script rules, shot-level cuts, end to end."""
import json
import re
from pathlib import Path

import pytest

from shorts_pipeline.cli import generate
from shorts_pipeline.gemini import storyline
from shorts_pipeline.gemini.storyline import Context, StorylineScript, place_storyline, validate_script
from gemini_fakes import FakeSynth, FakeTransport, make_source

PLAN = {"film": "John Wick: Chapter 3 - Parabellum", "year": 2019,
        "premise": "Hitman John Wick killed a crime lord inside the Continental, so the High Table put a bounty on him.",
        "items": [
            {"item_id": "I1", "title": "Bounty placed", "what_happens": "Winston gives John an hour", "story_context":
             "the countdown starts", "must_have": True, "connects_to": ["I2"], "queries": ["q"]},
            {"item_id": "I2", "title": "Library fight", "what_happens": "John fights a giant with a book",
             "story_context": "the first hunter finds him", "must_have": True, "connects_to": ["I3"], "queries": ["q"]},
            {"item_id": "I3", "title": "The Elder", "what_happens": "John cuts off his finger", "story_context":
             "he buys back his life", "must_have": True, "connects_to": [], "queries": ["q"]},
            {"item_id": "I4", "title": "Unfound", "what_happens": "no footage", "must_have": False, "connects_to": [],
             "queries": ["q"]}]}
POOL = [{"pool_start": 0.0, "pool_end": 10.0, "item_id": "I1", "item_title": "Bounty placed", "shows": "hotel"},
        {"pool_start": 10.0, "pool_end": 20.0, "item_id": "I2", "item_title": "Library fight", "shows": "library"},
        {"pool_start": 20.0, "pool_end": 30.0, "item_id": "I3", "item_title": "The Elder", "shows": "desert tent"}]
# Four 2.5 s shots per story item; the last shot of I1 is filler.
SCENES = [{"id": f"S{n:04d}", "start": n * 2.5, "end": (n + 1) * 2.5} for n in range(12)]


def labelled_catalog() -> dict:
    from shorts_pipeline.gemini.shots import catalog_shots
    shots = catalog_shots(SCENES, storyline.item_ranges(POOL))
    return {s["shot_id"]: {**s, "visible": f"John in {s['shot_id']}", "intensity": 4,
                           "filler": s["shot_id"] == "I1-004", "peak": (s["start"] + s["end"]) / 2,
                           "motion": "normal", "focus_x": 0.4} for s in shots}


SHOTS = labelled_catalog()
WORDS = ("since the bounty keeps growing every killer in the city now wants the reward and the clock "
         "never stops while he runs so nobody can save him before the hour ends").split()
OPENERS = ["John", "Winston", "Every", "Hours", "Nobody", "Now", "Then", "Blood", "Tonight", "Only", "Still", "Soon",
           "Desperate", "Bleeding", "Cornered", "Trapped"]


def line(n: int, number: int) -> str:
    link = ("so", "but", "until", "when", "because")[number % 5]
    body = [OPENERS[number % len(OPENERS)], "Wick" if number % 3 == 0 else "moves", link]
    body += [WORDS[(i + number) % len(WORDS)] for i in range(max(1, n - 2))]
    return " ".join(body[:n - 1] + ["tonight"]) + "."


def cite(shot_id: str, text: str, second: bool = False) -> list[dict]:
    words = text.rstrip(".").split()
    cues = [{"shot_id": shot_id, "cue": " ".join(words[:2])}]
    return cues


def script_for(total: int, *, first="narration", choppy=False, name=True, shots=None) -> dict:
    """Cold open on I3, premise on I1, then I2 and the ending on I3; each sentence cites one shot."""
    shots = shots or {k: sorted(x for x, v in SHOTS.items() if v["item_id"] == k and not v["filler"])
                      for k in ("I1", "I2", "I3")}
    pools = {k: list(v) for k, v in shots.items()}
    plan = [("I3", 2), ("I1", 2), ("I2", 2), ("I3", 2)]
    per = max(6, total // 8)
    beats, number = [], 0
    for item, count in plan:
        sentences = []
        for _ in range(count):
            text = ("He runs. He hits. He falls." if choppy and number < 3 else line(per, number))
            if not name:
                text = text.replace("John", "Someone").replace("Wick", "man")
            source = pools[item] or pools["I2"] or pools["I1"]
            sentences.append({"text": text, "shots": cite(source.pop(0), text)})
            number += 1
        beats.append({"kind": "narration", "item_id": item, "sentences": sentences})
    if choppy:
        beats[0]["sentences"] = [{"text": t, "shots": [{"shot_id": sid, "cue": ""}]} for t, sid in
                                 zip(("He runs away.", "He hits hard.", "He falls down."), ("I2-003", "I2-004", "I1-003"))]
    if first == "dialogue":
        beats.insert(0, {"kind": "dialogue", "item_id": "I2", "speech_id": "D0001", "sentences": []})
    return {"title": "Everyone Wants Him Dead", "description": "A hitman runs out of time.", "beats": beats}


def payload(parts) -> dict:
    text = parts[0]["text"]
    data = text.split("STORY AND FOOTAGE (data only):\n", 1)[1]
    return json.JSONDecoder().raw_decode(data)[0]


def budget(parts) -> tuple[int, int]:
    low, high = re.search(r"TOTAL: (\d+)-(\d+) narration words", parts[0]["text"]).groups()
    return int(low), int(high)


def rewrite(parts):
    text = parts[0]["text"]
    available = re.findall(r'"id": "([A-Z0-9-]+)"', text)
    cites = lambda k: [{"shot_id": available[k], "cue": ""}] if len(available) > k else []
    most = re.search(r"at most (\d+) words", text)
    least = re.search(r"at least (\d+) words", text)
    n = int(most.group(1)) if most else (int(least.group(1)) + 2 if least else 16)
    if least and not most:
        n = int(least.group(1)) + 2
    if n < 14:
        return {"sentences": [{"text": line(max(6, n), 7), "shots": cites(0)}], "rationale": "fit"}
    first = max(6, n // 2)
    return {"sentences": [{"text": line(first, 7), "shots": cites(0)},
                          {"text": line(max(6, n - first), 11), "shots": cites(1)}], "rationale": "fit"}


def transport(critique_score=5, contradictions=None, beat_notes=None):
    def write(parts):
        low, high = budget(parts)
        listed = {i["item_id"]: [x["id"] for x in i["shots"]] for i in payload(parts)["story_items_in_film_order"]}
        return script_for((low + high) // 2, shots=listed)

    def label(parts):
        listing = re.findall(r"'shot_id': '([^']+)', 'start': ([\d.]+), 'end': ([\d.]+)", parts[-1]["text"])
        return {"shots": [{"shot_id": sid, "visible": f"John fights in {sid}", "intensity": 4,
                           "filler": sid == "I1-004", "peak": (float(a) + float(b)) / 2, "motion": "normal",
                           "focus_x": 0.4} for sid, a, b in listing]}

    def check(parts):
        return {"contradictions": contradictions or []}

    scores = {k: critique_score for k in storyline.SCORE_KEYS}
    return FakeTransport(overrides={"StorylineScript": write, "BeatText": rewrite, "VisualCheck": check,
                                    "ShotLabels": label,
                                    "StoryCritique": {**scores, "problems": ["weak hook"], "rewrite_notes": "sharpen",
                                                      "beat_notes": beat_notes or []}})


def context(**extra) -> Context:
    request = {"title": "t", "brief": "", "user_story_notes": "The protagonist is John Wick.", **extra}
    return Context(PLAN, POOL, SHOTS, {}, request, rate=3.0, target=20)


# ------------------------------------------------------------------ rules
def test_script_rules_reject_dialogue_first_frame_by_frame_and_a_missing_hero():
    ctx = context()
    low, high = ctx.budget
    good = validate_script(ctx, StorylineScript.model_validate(script_for((low + high) // 2)))
    assert [b["item_id"] for b in good] == ["I3", "I1", "I2", "I3"] and good[0]["kind"] == "narration"
    assert all(b["source_gain"] == storyline.NARRATION_SOURCE_GAIN for b in good)
    with pytest.raises(ValueError, match="one action at a time"):
        validate_script(ctx, StorylineScript.model_validate(script_for((low + high) // 2, choppy=True)))
    with pytest.raises(ValueError, match="never name the hero"):
        validate_script(ctx, StorylineScript.model_validate(script_for((low + high) // 2, name=False)))
    with pytest.raises(ValueError, match="words"):
        validate_script(ctx, StorylineScript.model_validate(script_for(high * 2)))
    # The unfound item never reaches the writer, and an unknown item is rejected.
    assert [i["item_id"] for i in ctx.items] == ["I1", "I2", "I3"]
    bad = script_for((low + high) // 2)
    bad["beats"][2]["item_id"] = "I4"
    with pytest.raises(ValueError, match="item_id 'I4'"):
        validate_script(ctx, StorylineScript.model_validate(bad))


def test_dialogue_clip_may_not_open_the_short():
    quote = {"id": "D0001", "start": 12.0, "end": 13.5, "text": "Hold the line.", "confidence": 0.9,
             "words": [{"word": "Hold", "start": 0, "end": 0.4}], "timing_provenance": "asr"}
    ctx = Context(PLAN, POOL, SHOTS, {"D0001": quote}, {"title": "t", "user_story_notes": "John Wick"},
                  rate=3.0, target=20)
    low, high = ctx.budget
    with pytest.raises(ValueError, match="cannot open"):
        validate_script(ctx, StorylineScript.model_validate(script_for((low + high) // 2, first="dialogue")))


def test_style_examples_never_use_a_reference_about_the_same_film():
    examples = storyline.style_examples("John Wick: Chapter 3 - Parabellum (2019)", count=8)
    assert examples and not any("wick" in e.lower() for e in examples)
    assert len(examples) >= 2


# ------------------------------------------------------------------ placement
def measured(beat: dict, rate: float = 3.0) -> dict:
    tokens = beat["text"].split()
    step = 1 / rate
    return {**beat, "duration": len(tokens) * step + 0.2,
            "words": [{"word": t, "start": 0.1 + i * step, "end": 0.1 + (i + 0.9) * step} for i, t in enumerate(tokens)]}


def test_every_sentence_cuts_to_its_cited_shots_at_most_every_two_seconds_and_none_is_dropped():
    ctx = context()
    low, high = ctx.budget
    beats = [measured(b) for b in validate_script(ctx, StorylineScript.model_validate(script_for(high)))]
    placed, evidence = place_storyline(ctx, beats)
    assert len(evidence) == sum(len(b["sentences"]) for b in beats)  # Nothing removed.
    windows = []
    for beat in placed:
        for sentence in beat["sentences"]:
            mine = [x for x in beat["shots"] if x["sentence_id"] == sentence["sentence_id"]]
            # The cited shot opens the sentence, inside its own catalog shot.
            first = sentence["shots"][0]["shot_id"]
            assert mine[0]["shot_id"] == first
            assert SHOTS[first]["start"] - 1e-6 <= mine[0]["start"] and mine[0]["end"] <= SHOTS[first]["end"] + 1e-6
        for shot in beat["shots"]:
            assert shot["end"] - shot["start"] <= storyline.MAX_CUT + 0.01  # Fast cutting.
            assert not SHOTS[shot["shot_id"]]["filler"]  # Filler never plays while other shots are left.
            windows.append((shot["start"], shot["end"]))
    windows.sort()
    assert all(b1 <= a2 + 1e-6 for (_, b1), (a2, _) in zip(windows, windows[1:]))  # No footage shown twice.
    for entry in evidence:
        assert abs(entry["footage_seconds"] - entry["estimated_seconds"]) < 0.05


def test_a_cue_cuts_the_picture_on_the_words_it_illustrates():
    from shorts_pipeline.gemini.shots import cue_times
    sentence = {"text": "John grabs a book and smashes the giant with it.",
                "shots": [{"shot_id": "I2-001", "cue": "John grabs a book"}, {"shot_id": "I2-002", "cue": "smashes the giant"}]}
    words = [{"word": w, "start": 0.3 * i, "end": 0.3 * i + 0.25} for i, w in enumerate(sentence["text"].split())]
    assert cue_times(sentence, words, (0.0, 3.0), 0) == [0.0, pytest.approx(1.5)]  # "smashes" is word 5.


def test_slop_is_rejected():
    from shorts_pipeline.gemini.shots import slop
    assert slop(["Little did he know the clock is ticking."])
    assert slop(["A brutal, deadly, ruthless, relentless fight begins."])
    assert not slop(["John drives a pencil through the man's hand because the man came to collect the bounty."])


# ------------------------------------------------------------------ end to end
@pytest.fixture
def discovered(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123456")
    from shorts_pipeline import source_speech, sources
    monkeypatch.setattr(sources, "index_source", lambda path, work: [dict(x) for x in SCENES])
    from shorts_pipeline.gemini import discovery
    monkeypatch.setattr(source_speech, "analyze_source_speech", lambda path, work: {"cache_key": "none",
                                                                                     "segments": [], "candidates": []})
    base = tmp_path / "work" / "discovery" / "k"
    base.mkdir(parents=True)
    pool = make_source(base / "pool.mp4", 30)
    (base / "story-plan.json").write_text(json.dumps(PLAN))
    (base / "pool-map.json").write_text(json.dumps(POOL))
    monkeypatch.setattr(discovery, "discover_footage", lambda *a, **k: (pool, "pool notes"))
    return tmp_path


def _request(**extra):
    return {"title": "Everyone Wants Him Dead", "source": "discover:John Wick: Chapter 3 - Parabellum (2019)",
            "brief": "", "story_notes": "The protagonist is John Wick.", "target_seconds": 20, "fps": 30,
            "preview_width": 160, "planner": "gemini", "gemini_preset": "balanced", **extra}


def test_discovered_footage_is_narrated_as_a_story_over_its_own_footage(discovered):
    fake = transport()
    out = discovered / "out"
    result = generate(_request(), discovered / "work", out, gemini_transport=fake, synthesize=FakeSynth(3.0))
    assert Path(result["final"]).is_file()
    names = [c["schema"] for c in fake.calls]
    assert "StorylineScript" in names and "ScriptDraft" not in names and "MatchingResult" not in names
    assert names.count("StorylineScript") == 1  # A strong critique needs no rewrite.
    timeline = json.loads((out / "timeline.json").read_text())
    assert timeline["planning"]["path"] == "storyline"
    assert [b["item_id"] for b in timeline["beats"] if b["kind"] == "narration"][:2] == ["I3", "I1"]
    history = json.loads((out / "correction-history.json").read_text())
    assert not any("remove" in h.get("action", "") for h in history)
    evidence = json.loads((out / "evidence-map.json").read_text())
    assert len(evidence["sentences"]) == sum(len(b["sentences"]) for b in timeline["beats"] if b["kind"] == "narration")
    quality = json.loads((out / "quality-review.json").read_text())
    assert quality["path"] == "storyline" and quality["story_critique"][0]["hook"] == 5
    assert quality["cuts"]["longest_seconds"] <= storyline.MAX_CUT + 0.01 and quality["cuts"]["count"] >= 8
    catalog = json.loads((out / "shot-catalog.json").read_text())
    assert catalog["I1-004"]["filler"] and not any(x["shot_id"] == "I1-004" for b in timeline["beats"]
                                                   for x in b.get("shots", []) if "shot_id" in x)
    assert "John" in (out / "script.txt").read_text()


def test_a_weak_story_gets_exactly_one_editor_driven_rewrite(discovered):
    fake = transport(critique_score=2)
    generate(_request(), discovered / "work", discovered / "out", gemini_transport=fake, synthesize=FakeSynth(3.0))
    calls = [c for c in fake.calls if c["schema"] == "StorylineScript"]
    assert len(calls) == 2
    assert "EDITOR NOTES" in calls[1]["parts"][0]["text"] and "weak hook" in calls[1]["parts"][0]["text"]
    # The rewritten script is scored again for the report, and a still-weak story is flagged.
    quality = json.loads((discovered / "out" / "quality-review.json").read_text())
    assert len(quality["story_critique"]) == 2 and quality["review_status"] == "passed_with_warnings"
    assert any(i["check"] == "story_quality" for i in quality["issues"])


def test_a_visual_contradiction_rewrites_its_beat_without_deleting_lines(discovered):
    fake = transport(contradictions=[{"sentence_id": "N02.1", "on_screen": "a hotel lobby",
                                      "problem": "names the wrong man"}])
    out = discovered / "out"
    generate(_request(), discovered / "work", out, gemini_transport=fake, synthesize=FakeSynth(3.0))
    history = json.loads((out / "correction-history.json").read_text())
    assert any(h["action"] == "rewrite_contradiction" and h["outcome"] == "rewritten" for h in history)
    timeline = json.loads((out / "timeline.json").read_text())
    beat = next(b for b in timeline["beats"] if b["id"] == "N02")
    assert len(beat["sentences"]) == 2 and beat["shots"]


def test_overlong_narration_is_shortened_by_whole_beat_rewrites(discovered):
    fake = transport()
    out = discovered / "out"
    generate(_request(), discovered / "work", out, gemini_transport=fake, synthesize=FakeSynth(1.8))
    history = json.loads((out / "correction-history.json").read_text())
    assert any(h["action"] == "shorten_beat" and h["outcome"] == "rewritten" for h in history)
    timeline = json.loads((out / "timeline.json").read_text())
    assert timeline["duration"] <= 20.05
    assert not any("remove" in h.get("action", "") for h in history)


class ScriptedClient:
    def __init__(self, data):
        from types import SimpleNamespace
        self.data, self.calls, self.ledger = data, 0, SimpleNamespace(note=lambda **k: None)

    def generate(self, stage, schema, parts, **kwargs):
        self.calls += 1
        return schema.model_validate(self.data)


def test_a_well_formed_script_is_kept_with_an_issue_after_three_rule_failures():
    ctx = context()
    low, high = ctx.budget
    client = ScriptedClient(script_for((low + high) // 2, name=False, first="dialogue"))
    result = storyline.write_storyline(client, ctx)
    assert client.calls == 3 and "never name the hero" in result["accepted_with_issue"]
    assert result["beats"][0]["kind"] == "narration"
    # Malformed output (an unknown story item) still fails loudly.
    bad = script_for((low + high) // 2)
    bad["beats"][0]["item_id"] = "I9"
    with pytest.raises(ValueError, match="failed validation after 3 attempts"):
        storyline.write_storyline(ScriptedClient(bad), ctx)


def test_only_clean_short_film_lines_are_offered_as_dialogue():
    def quote(ident, text, start, end, confidence=0.9):
        return {"id": ident, "start": start, "end": end, "text": text, "confidence": confidence,
                "words": [{"word": w, "start": 0, "end": 0.1} for w in text.split()], "timing_provenance": "asr"}
    pool = {"D1": quote("D1", "little run down there John. Ernest I still have time. He's almost up.", 11, 15),
            "D2": quote("D2", "I still have time.", 12, 13.2),
            "D3": quote("D3", "Who's gonna know?", 14, 15, confidence=0.5)}
    ctx = Context(PLAN, POOL, SHOTS, pool, {"title": "t", "user_story_notes": "John Wick"},
                  rate=3.0, target=20)
    assert [c["speech_id"] for c in ctx.dialogue_clips()] == ["D2"]
    low, high = ctx.budget
    data = script_for((low + high) // 2)
    data["beats"].insert(2, {"kind": "dialogue", "item_id": "I2", "speech_id": "D1", "sentences": []})
    data["beats"].insert(3, {"kind": "dialogue", "item_id": "I2", "speech_id": "D2", "sentences": []})
    beats = validate_script(ctx, StorylineScript.model_validate(data))
    assert [b["speech_id"] for b in beats if b["kind"] == "dialogue"] == ["D2"]


def test_a_beat_may_only_cite_its_own_shots_after_the_opening():
    ctx = context()
    low, high = ctx.budget
    data = script_for((low + high) // 2)
    data["beats"][2]["sentences"][0]["shots"] = [{"shot_id": "I3-004", "cue": ""}]  # Library beat, Elder shot.
    with pytest.raises(ValueError, match="belongs to I3"):
        validate_script(ctx, StorylineScript.model_validate(data))
    data = script_for((low + high) // 2)
    data["beats"][2]["sentences"][0]["shots"] = [{"shot_id": "I1-004", "cue": ""}]  # Filler.
    with pytest.raises(ValueError, match="not a showable shot"):
        validate_script(ctx, StorylineScript.model_validate(data))
    # The premise beat (I1) may show the hero from any moment.
    data = script_for((low + high) // 2)
    data["beats"][1]["sentences"][0]["shots"] = [{"shot_id": "I2-004", "cue": ""}]
    validate_script(ctx, StorylineScript.model_validate(data))


def test_editor_beat_notes_fix_factual_errors_even_in_a_strong_script(discovered):
    fake = transport(beat_notes=[{"beat_id": "N03", "note": "Sofia leads him to Berrada, not the Elder",
                                  "factual_error": True},
                                 {"beat_id": "N04", "note": "could be punchier", "factual_error": False},
                                 {"beat_id": "N99", "note": "unknown beat", "factual_error": True}])
    out = discovered / "out"
    generate(_request(), discovered / "work", out, gemini_transport=fake, synthesize=FakeSynth(3.0))
    history = json.loads((out / "correction-history.json").read_text())
    fixes = [h for h in history if h["action"] == "editor_beat_fix"]
    # Scores are all 5: only the factual error is fixed, the style note and the unknown beat are not.
    assert [(h["beat_id"], h["outcome"]) for h in fixes] == [("N03", "rewritten")]
    rewrites = [c for c in fake.calls if c["schema"] == "BeatText"]
    assert "Berrada" in rewrites[0]["parts"][0]["text"]
    critiques = json.loads((out / "story-critique.json").read_text())
    assert critiques[-1]["after_beat_fixes"] == ["N03"]


def test_placement_never_loops_or_fails_when_footage_runs_out():
    """Run 21: a rounding sliver made the fill loop forever (CI) or fail with 'pool exhausted'."""
    from shorts_pipeline.gemini.shots import HARD_MAX, Footage, place_sentence
    shots = {k: dict(v) for k, v in SHOTS.items() if v["item_id"] == "I2"}
    footage = Footage(shots, ["I2"])
    # Odd cue times that leave sub-millisecond slivers, and far more speech than footage.
    placed, notes = place_sentence(footage, "I2", ["I2-001", "I2-002"], [0.0, 2.2004], 14.3337, 0, "N01.1")
    total = sum(x["end"] - x["start"] for x in placed)
    assert abs(total - 14.3337) < 0.06 and all(x["end"] - x["start"] <= HARD_MAX + 1e-6 for x in placed)
    assert any("repeated" in n for n in notes)  # Reported, not hidden.


class HardToHear(FakeSynth):
    """Fails speech verification for one beat until it is rewritten (or, if stubborn, until it is loose)."""

    def __init__(self, beat_id, stubborn=False):
        super().__init__(3.0)
        self.beat_id, self.stubborn, self.failures = beat_id, stubborn, 0

    def __call__(self, plan, narrator, workdir, model="original", speed=1.0):
        for beat in plan["beats"]:
            if beat["id"] == self.beat_id and not beat.get("loose") and (self.stubborn or not beat.get("speech_rewritten")):
                self.failures += 1
                raise RuntimeError(f"Narration beat {beat['id']} failed complete-speech verification: "
                                   "ASR word counts differ; retry speech or provide reviewed word timings.")
        return super().__call__(plan, narrator, workdir, model, speed)


@pytest.mark.parametrize("stubborn", [False, True])
def test_a_line_the_voice_check_cannot_hear_is_rewritten_then_matched_loosely_never_fatal(discovered, stubborn):
    """Run 22 died on 'Fierce attack dogs maul' (heard 'Fierce the tech dogs, more')."""
    fake = transport()
    out = discovered / "out"
    synth = HardToHear("N02", stubborn=stubborn)
    result = generate(_request(), discovered / "work", out, gemini_transport=fake, synthesize=synth)
    assert Path(result["final"]).is_file()
    history = json.loads((out / "correction-history.json").read_text())
    rewrites = [h for h in history if h["action"] == "speech_rewrite"]
    assert len(rewrites) == 1 and rewrites[0]["beat_id"] == "N02" and rewrites[0]["outcome"] == "rewritten"
    assert any("could not say this line" in c["parts"][0]["text"] for c in fake.calls if c["schema"] == "BeatText")
    loose = [h for h in history if h["action"] == "speech_loose_match"]
    assert len(loose) == (1 if stubborn else 0) and synth.failures == (2 if stubborn else 1)
    assert not any("remove" in h.get("action", "") for h in history)


def test_generic_tacked_on_reasons_are_slop():
    from shorts_pipeline.gemini.shots import slop
    run22 = ["John clashes with armed mercenaries because escape is blocked.",
             "Fierce attack dogs maul tactical guards because lethal force is absolutely required."]
    assert any("tacked-on" in f for f in slop(run22))
    many = [f"John runs to the car because the man {n} called." for n in range(5)] + ["He drives."] * 5
    assert any("'because' in" in f for f in slop(many))
    assert not slop(["Ernest corners John between the stacks, so John grabs the only weapon in reach: a book."])


def test_flash_lite_never_writes_the_script_the_run_stops_resumable_instead(discovered):
    """Run 22: every key's Flash quota was spent, Flash-Lite wrote the script, and it was filler."""
    from gemini_fakes import FakeAPIError
    daily = "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    fake = transport()
    fake.models = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    fake.overrides["StorylineScript"] = lambda parts: FakeAPIError("rate_limit", daily)
    with pytest.raises(Exception, match="rerun after the quota resets"):
        generate(_request(gemini_preset="free"), discovered / "work", discovered / "out", gemini_transport=fake,
                 synthesize=FakeSynth(3.0))
    written = [c["model"] for c in fake.calls if c["schema"] == "StorylineScript"]
    assert written and not any("lite" in m for m in written)
