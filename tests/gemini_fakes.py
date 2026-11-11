"""Scripted offline stand-ins for the Gemini SDK transport and Chatterbox synthesis.

Responses are derived from the prompt evidence so stage wiring, validation and
correction logic run exactly as in production without network access.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


class FakeAPIError(Exception):
    def __init__(self, category: str, message: str = "fake failure"):
        super().__init__(message)
        self.category = category


def evidence(parts: list[dict]) -> dict:
    for part in parts:
        text = part.get("text", "")
        if text.startswith("EVIDENCE (data only):\n"):
            return json.loads(text.split("\n", 1)[1])
    return {}


def last_json(text: str):
    """The trailing JSON payload of an instruction text part."""
    for start in [m.start() for m in re.finditer(r"[\[{]", text)]:
        try:
            return json.loads(text[start:])
        except ValueError:
            continue
    return None


SENTENCE_WORDS = ["The", "stranger", "grabs", "the", "rifle", "from", "the", "crate", "and", "runs",
                  "toward", "the", "waiting", "truck", "before", "anyone", "can", "react"]


def sentence(words: int, number: int) -> str:
    body = [SENTENCE_WORDS[(i + number * 3) % len(SENTENCE_WORDS)].lower() for i in range(words - 1)] + ["tonight"]
    body[0] = body[0].capitalize()
    return " ".join(body) + "."


class FakeTransport:
    """Behaves like SdkTransport. `overrides[schema_name]` may return a dict or raise."""

    def __init__(self, events: list[dict] | None = None, models=("gemini-3.8-flash", "gemini-3.1-pro-preview",
                                                                  "gemini-3.5-flash-lite"), overrides=None):
        self.events = events or []
        self.models = list(models)
        self.overrides = overrides or {}
        self.calls: list[dict] = []
        self.uploads = 0
        self.files: dict[str, dict] = {}

    def classify(self, exc):
        return getattr(exc, "category", "request"), str(exc)

    def list_models(self):
        return self.models

    def upload(self, path, mime_type):
        self.uploads += 1
        name = f"files/fake-{self.uploads}"
        self.files[name] = {"name": name, "uri": f"https://fake/{name}", "mime_type": mime_type, "state": "ACTIVE",
                            "expires_epoch": 4_000_000_000}
        return self.files[name]

    def get_file(self, name):
        # Like the real Files API, uploads persist across client instances until expiry.
        return self.files.get(name, {"name": name, "uri": f"https://fake/{name}", "mime_type": "video/mp4",
                                     "state": "ACTIVE", "expires_epoch": 4_000_000_000})

    def generate(self, model, parts, *, system, schema, **config):
        name = schema.__name__
        self.calls.append({"model": model, "schema": name, "parts": parts, "config": config})
        if name in self.overrides:
            value = self.overrides[name](parts) if callable(self.overrides[name]) else self.overrides[name]
            if isinstance(value, Exception):
                raise value
            if isinstance(value, str):
                return self._result(value)
            return self._result(json.dumps(value))
        return self._result(json.dumps(getattr(self, "_" + name)(parts)))

    @staticmethod
    def _result(text):
        return {"text": text, "finish_reason": "STOP", "prompt_tokens": 1200, "cached_tokens": 200,
                "output_tokens": 300, "thinking_tokens": 100, "audio_tokens": 0}

    # ---- default stage behaviour
    def _WindowAnalysis(self, parts):
        return {"window_summary": "synthetic footage", "events": self.events}

    def _VerificationResult(self, parts):
        claims = last_json(parts[-1]["text"])
        return {"checks": [{"event_id": c["event_id"], "verdict": "confirmed", "start": c["claimed_start"],
                            "end": c["claimed_end"], "peak": (c["claimed_start"] + c["claimed_end"]) / 2,
                            "visible_action": c["claimed_action"], "focus_x": 0.4} for c in claims]}

    def _StoryMap(self, parts):
        events = evidence(parts)["verified_event_inventory"]
        return {"logline": "A stranger takes a rifle and escapes.",
                "characters": [{"character_id": "C1", "label": "stranger", "name": "", "name_basis": "none",
                                "event_ids": [e["id"] for e in events]}],
                "events": [{"event_id": e["id"], "summary": e["visible_action"], "character_ids": ["C1"],
                            "preceding_event_ids": [events[i - 1]["id"]] if i else [],
                            "following_event_ids": [events[i + 1]["id"]] if i + 1 < len(events) else [],
                            "causes_event_ids": [], "story_function": "escalation", "importance": 4,
                            "inferred_context": [{"claim": "He is in a hurry", "basis": "visual_sequence",
                                                  "supporting_event_ids": [e["id"]], "confidence": 0.6}]}
                           for i, e in enumerate(events)], "open_questions": []}

    def _EditorialPlan(self, parts):
        data = evidence(parts)
        events = data["verified_event_inventory"]
        quotes = [q["speech_id"] for q in data["selectable_source_dialogue"]][:1]
        return {"title": "Synthetic", "description": "A synthetic test recap.", "hook_event_id": events[0]["id"],
                "cold_open": False, "ending_mode": "renewed_threat",
                "selected": [{"event_id": e["id"], "role": "hook" if i == 0 else "escalation", "reason": "visible"}
                             for i, e in enumerate(events)], "dialogue_speech_ids": quotes, "dropped": []}

    def _beats(self, parts, low, high):
        data = evidence(parts)
        events = [e["id"] for e in data["verified_event_inventory"]]
        plan = last_json(parts[-1]["text"].split("APPROVED PLAN:\n", 1)[-1].split("\n\nDRAFT:", 1)[0]) or {}
        quotes = (plan.get("editorial_plan") or {}).get("dialogue_speech_ids", [])
        count = min(4, len(events))
        per = max(3, ((low + high) // 2) // count)
        sentences = [{"text": sentence(per, i), "event_ids": [events[i]], "visual_requirement": "the action",
                      "claim_basis": "observed"} for i in range(count)]
        beats = [{"beat_id": "b1", "kind": "narration", "delivery": "dramatic", "sentences": sentences[:2]}]
        beats += [{"beat_id": "q1", "kind": "dialogue", "speech_id": q, "sentences": []} for q in quotes]
        beats += [{"beat_id": "b2", "kind": "narration", "delivery": "normal", "sentences": sentences[2:]}]
        return beats

    def _ScriptDraft(self, parts):
        low, high = map(int, re.search(r"in (\d+)–(\d+) narration words", parts[-1]["text"]).groups())
        return {"title": "Synthetic", "description": "desc", "beats": self._beats(parts, low, high)}

    def _ScriptRevision(self, parts):
        low, high = map(int, re.search(r"within (\d+)–(\d+) words", parts[-1]["text"]).groups())
        return {"problems_found": ["none serious"], "changes_made": [], "beats": self._beats(parts, low, high)}

    def _BeatRewrite(self, parts):
        prompt = next(p["text"] for p in parts if p.get("text", "").startswith("Rewrite only"))
        request = last_json(prompt)
        beat = request["beat"]
        limit = re.search(r"at most (\d+) words", prompt)
        total = int(limit.group(1)) if limit else len(beat["text"].split())
        count = len(beat["sentences"])
        per = max(3, total // count)
        return {"beat_id": beat["id"], "rationale": "shortened",
                "sentences": [{"text": sentence(per, i), "event_ids": s["event_ids"],
                               "visual_requirement": s["visual_requirement"], "claim_basis": "observed"}
                              for i, s in enumerate(beat["sentences"])]}

    def _MatchingResult(self, parts):
        listing = last_json(parts[-1]["text"].split("\n", 1)[1])
        matches = []
        for item in listing:
            cited = [c for c in item["candidate_shots"]
                     if c["event_id"] in item["cited_event_ids"] and c["relation"] == "overlaps_event"]
            shot = max(cited or item["candidate_shots"], key=lambda c: c["end"] - c["start"])
            matches.append({"sentence_id": item["sentence_id"], "verdict": "supported",
                            "shots": [{"shot_id": shot["shot_id"], "start": shot["start"], "end": shot["end"],
                                       "anchor": (shot["start"] + shot["end"]) / 2, "focus_x": 0.45,
                                       "shows": shot["event_observation"], "reason": "shows the cited action",
                                       "relevance": 0.9}]})
        return {"matches": matches}

    def _ClipPlan(self, parts):
        listing = last_json(parts[-1]["text"].split("\n", 1)[1])
        events = []
        for item in listing:
            best = max(item["candidate_shots"], key=lambda c: min(c["end"], item["end"]) - max(c["start"], item["start"]))
            a, b = max(best["start"], item["start"]), min(best["end"], item["end"])
            events.append({"event_id": item["event_id"], "clips": [
                {"shot_id": best["shot_id"], "start": a, "end": b, "anchor": (a + b) / 2, "focus_x": 0.4,
                 "shows": item["observed"], "role": "action", "quality": 0.9}]})
        return {"events": events}

    def _PictureScript(self, parts):
        text = parts[-1]["text"]
        target = int(re.search(r"Aim for about (\d+) words", text).group(1))
        clips = last_json(text.split("\n", 1)[1])["clips"]
        backbone = clips[:-1] if len(clips) > 2 else clips
        per = max(3, round(target * 1.05 / len(backbone)))
        lines = []
        for i, clip in enumerate(backbone):
            words = min(per, max(3, clip["word_capacity"]))
            line = sentence(words, i)
            lines.append({"clip_ids": [clip["clip_id"]], "text": line, "key_word": line.split()[1]})
        return {"lines": lines}

    def _ReconciledScript(self, parts):
        text = parts[-1]["text"]
        low, high = map(int, re.search(r"\((\d+)–(\d+) words", text).groups())
        data = last_json(text.split("\n", 1)[1])
        lines = data["script_a"]
        actions = [{"kind": "action", "text": l["text"], "source_line_id": l["line_id"], "clip_ids": l["clip_ids"],
                    "key_word": l["key_word"], "basis": "observed"} for l in lines]
        total = sum(len(l["text"].split()) for l in lines)
        beats = [{"beat_id": "m1", "kind": "narration", "delivery": "dramatic", "lines": actions[:2]}]
        beats += [{"beat_id": "q", "kind": "dialogue", "speech_id": q} for q in data["approved_dialogue_speech_ids"]]
        rest = actions[2:]
        if data["cover_clips"]:
            cover = data["cover_clips"][0]
            words = min(max(4, low - total + 2), max(4, cover["word_capacity"]))
            rest.append({"kind": "context", "text": sentence(words, 7), "clip_ids": [cover["clip_id"]],
                         "event_ids": [cover["event_id"]], "basis": "inferred_context", "key_word": ""})
        beats.append({"beat_id": "m2", "kind": "narration", "delivery": "normal", "lines": rest})
        return {"title": "Synthetic", "description": "merged", "beats": beats, "merge_notes": []}

    def _QualityReview(self, parts):
        return {"verdict": "pass", "issues": [], "strengths": ["clear action"], "opening_score": 4,
                "coherence_score": 4, "ending_score": 3}


def synthetic_events(duration: float, span: float = 6.0) -> list[dict]:
    events, start, number = [], 0.0, 1
    while start < duration - 1:
        end = min(duration, start + span)
        events.append({"event_id": f"m{number}", "start": start + 0.2, "end": end - 0.2, "peak": (start + end) / 2,
                       "visible_action": f"the stranger moves the crate in part {number}", "entities": [
                           {"label": "stranger", "name": "", "name_basis": "none"}],
                       "objects": ["crate"], "location": "yard", "kind": "action", "importance": 4,
                       "confidence": 0.9, "motion": "normal", "focus_x": 0.5})
        start, number = end, number + 1
    return events


class FakeSynth:
    """Replaces Chatterbox: a tone whose length follows the word count, with even word timings."""

    def __init__(self, words_per_second: float = 2.5):
        self.rate = words_per_second
        self.calls = 0

    def __call__(self, plan, narrator, workdir, model="original"):
        out = dict(plan)
        beats = []
        directory = Path(workdir) / "fake-speech"
        directory.mkdir(parents=True, exist_ok=True)
        for beat in plan["beats"]:
            beat = dict(beat)
            if beat["kind"] == "narration" and not beat.get("words"):
                self.calls += 1
                tokens = beat["text"].split()
                duration = round(len(tokens) / self.rate + 0.2, 3)
                path = directory / f"{abs(hash(beat['text']))}.wav"
                if not path.exists():
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                                    f"sine=frequency=330:sample_rate=48000:duration={duration}", str(path)], check=True)
                step = (duration - 0.2) / len(tokens)
                beat.update(audio_path=str(path), duration=duration, timing_provenance="fake",
                            words=[{"word": t, "start": round(0.1 + i * step, 3), "end": round(0.1 + (i + 0.9) * step, 3)}
                                   for i, t in enumerate(tokens)])
            elif beat["kind"] == "dialogue":
                beat["duration"] = beat["shots"][0]["end"] - beat["shots"][0]["start"]
            beats.append(beat)
        out["beats"] = beats
        return out


def make_source(path: Path, seconds: float = 30) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=24:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path
