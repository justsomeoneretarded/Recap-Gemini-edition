"""Stage H: independent pre-render review of the storyboard and its evidence.

Two independent layers:
1. Deterministic checks that cannot be argued with: evidence references,
   source ranges, shot bounds, reuse, frame allocation of action anchors,
   measured duration, dialogue clip confidence, names absent from evidence.
2. A Gemini reviewer that watches the actual placed footage (and retained
   dialogue clips with audio) beat by beat, with no access to the writer's
   reasoning, and must list concrete problems with severity and evidence.
"""
from __future__ import annotations

import json
import re

from .client import GeminiClient, text_part, video_part
from .schemas import QualityReview
from .sync import dry_run_allocation

REVIEW_SYSTEM = (
    "You are an independent quality reviewer for a movie recap Short. You did not write or edit it, and your "
    "job is to find problems, not to approve. For every narration sentence, look at the footage that plays "
    "under it and decide whether it actually shows what is said at that moment. Check: (1) every statement is "
    "supported by evidence, (2) shots show the narrated event, (3) timestamps are valid, (4) important actions "
    "stay on screen long enough, (5) the story is coherent, (6) important events are missing, (7) identities "
    "or dialogue claims are unsupported, (8) transitions are confusing, (9) the opening engages immediately, "
    "(10) the ending makes narrative sense, (11) narration fits the footage, (12) retained dialogue clips are "
    "accurate and meaningful. Report each problem with severity (critical = false or unsupported claim or "
    "wrong footage; major = clear quality defect; minor = polish), the beat/sentence ID, source seconds and a "
    "fix type. Do not report something as a problem unless you can point to it. Lines of kind 'context' "
    "deliberately play over related cover footage: check only that it is related and does not contradict the line. "
    "For 'action' lines, check that the key_word's action or object is visible in the clip."
)
MIN_ACTION_SECONDS = 1.0
FACTUAL_CHECKS = ("unsupported_statement", "unsupported_identity_or_dialogue", "shot_mismatch")


def _issue(check, severity, description, fix="none", beat_id="", sentence_id="", start=None, end=None, origin="deterministic"):
    return {"check": check, "severity": severity, "beat_id": beat_id, "sentence_id": sentence_id,
            "source_start": start, "source_end": end, "description": description, "fix": fix, "origin": origin}


def _name_candidates(text: str) -> list[str]:
    names = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        tokens = re.findall(r"[A-Za-z][\w'’-]*", sentence)
        names += [t.strip("'’") for t in tokens[1:] if t[0].isupper() and t not in ("I",)]
    return names


def deterministic_checks(beats: list[dict], *, inventory: dict, speech_pool: dict, source_duration: float,
                         fps: int, target: float, notes: str, transcript: str, hook_event_id: str | None) -> list[dict]:
    issues = []
    known = {e["id"]: e for e in inventory["events"]}
    allowed_names = (notes + " " + transcript + " " + " ".join(
        en.get("name", "") for e in inventory["events"] for en in e.get("entities", []))).casefold()
    used: list[tuple[float, float, str]] = []
    for beat in beats:
        for shot in beat.get("shots", []):
            a, b = float(shot["start"]), float(shot["end"])
            owner = shot.get("sentence_id", beat["id"])
            if not 0 <= a < b <= source_duration + 1e-3:
                issues.append(_issue("invalid_timestamp", "critical", f"source range {a:.2f}–{b:.2f}s is outside the "
                                     f"0–{source_duration:.2f}s source", "rematch_shot", beat["id"], shot.get("sentence_id", ""), a, b))
            bounds = shot.get("shot_bounds")
            if bounds and (a < bounds[0] - 1e-3 or b > bounds[1] + 1e-3):
                issues.append(_issue("invalid_timestamp", "major", f"range {a:.2f}–{b:.2f}s crosses its detected shot "
                                     f"{bounds[0]:.2f}–{bounds[1]:.2f}s", "adjust_boundaries", beat["id"], shot.get("sentence_id", ""), a, b))
            for ua, ub, other in used:
                if min(b, ub) - max(a, ua) > 0.25 and other != owner:
                    issues.append(_issue("repetition", "major", f"footage {a:.2f}–{b:.2f}s repeats footage used by {other}",
                                         "rematch_shot", beat["id"], shot.get("sentence_id", ""), a, b))
            used.append((a, b, owner))
        if beat["kind"] == "dialogue":
            quote = speech_pool.get(beat.get("speech_id"))
            if quote is None:
                issues.append(_issue("dialogue_clip_inaccurate", "critical", "dialogue beat has no verified speech ID",
                                     "remove_claim", beat["id"]))
            elif quote["confidence"] < 0.8:
                issues.append(_issue("dialogue_clip_inaccurate", "minor", f"ASR confidence {quote['confidence']:.2f}; "
                                     "audition the quote", "none", beat["id"], start=quote["start"], end=quote["end"]))
            continue
        shots_by_sentence: dict[int, list] = {}
        for shot in beat.get("shots", []):
            shots_by_sentence.setdefault(int(shot.get("sentence_index", -1)), []).append(shot)
        for index, sentence in enumerate(beat["sentences"]):
            sid = sentence["sentence_id"]
            cited = [i for i in sentence["event_ids"] if i in known]
            if not cited:
                issues.append(_issue("unsupported_statement", "critical", f"'{sentence['text']}' cites no verified event",
                                     "rewrite_beat", beat["id"], sid))
            placed = shots_by_sentence.get(index, [])
            if not placed:
                issues.append(_issue("shot_mismatch", "critical", f"'{sentence['text']}' has no supporting footage",
                                     "rematch_shot", beat["id"], sid))
            for shot in placed:
                if shot.get("relevance", 1) < 0.6 and not shot.get("cover"):
                    issues.append(_issue("shot_mismatch", "major", f"{shot.get('shot_id')} only partially supports "
                                         f"'{sentence['text']}' (relevance {shot['relevance']:.2f})", "rematch_shot",
                                         beat["id"], sid, shot["start"], shot["end"]))
            for name in _name_candidates(sentence["text"]):
                if name.casefold() not in allowed_names:
                    issues.append(_issue("unsupported_identity_or_dialogue", "major", f"name '{name}' is not in the "
                                         "story notes, source dialogue or on-screen text evidence", "rewrite_beat", beat["id"], sid))
    allocation = dry_run_allocation(beats, source_duration, fps)
    for beat_id, shots in allocation.items():
        for shot in shots:
            if "error" in shot:
                issues.append(_issue("narration_footage_fit", "critical", shot["error"], "retime", beat_id))
                continue
            if shot.get("speed", 1) < 0.97:
                issues.append(_issue("narration_footage_fit", "minor", f"{shot.get('shot_id')} slowed to "
                                     f"{shot['speed']:.2f}x to cover narration", "rematch_shot", beat_id,
                                     shot.get("sentence_id", ""), shot["start"], shot["end"]))
            anchor = shot.get("anchor")
            if anchor is not None:
                shown = shot["end"] - shot["start"]
                if not shot["start"] - 1e-3 <= anchor <= shot["end"] + 1e-3:
                    issues.append(_issue("action_too_brief", "major", f"{shot.get('shot_id')}: the action at {anchor:.2f}s "
                                         f"is trimmed out (shows {shot['start']:.2f}–{shot['end']:.2f}s)", "adjust_boundaries",
                                         beat_id, shot.get("sentence_id", ""), shot["start"], shot["end"]))
                elif shown < min(MIN_ACTION_SECONDS, shot["source_end"] - shot["source_start"]) - 1e-3:
                    issues.append(_issue("action_too_brief", "major", f"{shot.get('shot_id')} is on screen for only "
                                         f"{shown:.2f}s", "adjust_boundaries", beat_id, shot.get("sentence_id", ""),
                                         shot["start"], shot["end"]))
    total = sum(float(b.get("duration", 0)) for b in beats)
    if total > target + 0.1:
        issues.append(_issue("pacing", "critical", f"measured duration {total:.1f}s exceeds the {target:.0f}s maximum", "retime"))
    elif total and total < 0.7 * target:
        issues.append(_issue("pacing", "minor", f"measured duration {total:.1f}s is well below the {target:.0f}s target"))
    first = next((b for b in beats if b["kind"] == "narration"), None)
    if hook_event_id and first and beats[0] is first and hook_event_id not in first["sentences"][0]["event_ids"]:
        issues.append(_issue("weak_opening", "minor", "the first sentence does not narrate the planned hook event",
                             "revise_opening", first["id"], first["sentences"][0]["sentence_id"]))
    return issues


def gemini_review(client: GeminiClient, file, beats: list[dict], *, inventory: dict, story_map: dict,
                  editorial: dict, only_beats: set[str] | None = None, log=print) -> tuple[list[dict], list[dict]]:
    """Watch the placed footage beat by beat. Returns (issues, review summaries)."""
    fps = client.settings.analysis["review_fps"]
    selected = {s["event_id"] for s in editorial["selected"]}
    context = {"story_logline": story_map.get("logline", ""),
               "selected_events": [{k: e[k] for k in ("event_id", "observed", "start", "end", "inferred_context")}
                                   for e in story_map["events"] if e["event_id"] in selected],
               "planned_ending_mode": editorial.get("ending_mode"), "hook_event_id": editorial.get("hook_event_id")}
    script = [{"beat_id": b["id"], "kind": b["kind"], "text": b["text"],
               "sentences": [{k: s[k] for k in ("sentence_id", "text", "event_ids", "visual_requirement")}
                             for s in b.get("sentences", [])]} for b in beats]
    chunks, current, count = [], [], 0
    for beat in beats:
        if only_beats is not None and beat["id"] not in only_beats:
            continue
        need = len(beat.get("shots", []))
        if current and count + need > 10:
            chunks.append(current)
            current, count = [], 0
        current.append(beat)
        count += need
    if current:
        chunks.append(current)
    issues, summaries = [], []
    ids = {b["id"]: {s["sentence_id"] for s in b.get("sentences", [])} for b in beats}
    for number, chunk in enumerate(chunks, 1):
        parts, clips = [], []
        for beat in chunk:
            sentences = {s["sentence_id"]: s for s in beat.get("sentences", [])}
            for shot in beat.get("shots", [])[:10]:
                parts.append(video_part(file, shot["start"], shot["end"], fps))
                line = sentences.get(shot.get("sentence_id", ""), {})
                clips.append({"clip": len(parts), "beat_id": beat["id"], "sentence_id": shot.get("sentence_id", ""),
                              "line_kind": line.get("line_kind", "action"), "key_word": line.get("key_word", ""),
                              "source_start": round(shot["start"], 2), "source_end": round(shot["end"], 2),
                              "editor_claims_it_shows": shot.get("shows", "")})
        log(f"Gemini review: chunk {number}/{len(chunks)} ({len(chunk)} beat(s), {len(parts)} clip(s))")
        result = client.generate(
            "review", QualityReview,
            parts + [text_part("Clips above, in order: " + json.dumps(clips, ensure_ascii=False)
                               + "\nBeats under review: " + json.dumps([b["id"] for b in chunk])
                               + "\nFull script for context: " + json.dumps(script, ensure_ascii=False)
                               + "\nEvidence: " + json.dumps(context, ensure_ascii=False))],
            system=REVIEW_SYSTEM, temperature=0.0, max_output_tokens=16384, resolution="low", thinking="high")
        summaries.append({"chunk": number, "beats": [b["id"] for b in chunk], "verdict": result.verdict,
                          "scores": {"opening": result.opening_score, "coherence": result.coherence_score,
                                     "ending": result.ending_score}, "strengths": result.strengths})
        for item in result.issues:
            issue = {**item.model_dump(), "origin": "gemini_review"}
            if issue["check"] in FACTUAL_CHECKS and issue["severity"] == "minor":
                # A wrong fact is never cosmetic in a published recap: make it actionable.
                issue["severity"] = "major"
                issue["description"] += " (escalated: factual accuracy)"
            if issue["beat_id"] and issue["beat_id"] not in ids:
                issue["description"] += f" (unknown beat {issue['beat_id']!r} in review output)"
                issue["beat_id"] = ""
            if issue["sentence_id"] and issue["sentence_id"] not in ids.get(issue["beat_id"], set()):
                owner = next((b for b, s in ids.items() if issue["sentence_id"] in s), "")
                issue["beat_id"] = issue["beat_id"] or owner
                if not owner:
                    issue["sentence_id"] = ""
            issues.append(issue)
    return issues, summaries


SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


def summarize(issues: list[dict]) -> dict:
    counts = {s: sum(1 for i in issues if i["severity"] == s) for s in SEVERITY_ORDER}
    return {"counts": counts, "issues": sorted(issues, key=lambda i: (SEVERITY_ORDER[i["severity"]], i["beat_id"]))}

