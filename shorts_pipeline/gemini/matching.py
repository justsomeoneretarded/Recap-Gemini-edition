"""Stages E and F: sentence-to-shot evidence matching and shot planning.

E. For every narration sentence, candidate shots come from the events it
   cites (verified timestamps + PySceneDetect shot boundaries). Gemini watches
   those candidate clips and states, per shot, what is actually visible, the
   best in/out points, the action anchor and the subject position, plus a
   supported/partial/unsupported verdict. Code rejects unknown shots, ranges
   outside shot bounds and low-relevance choices.
F. A deterministic planner picks footage per sentence: relevance first, no
   reuse of source ranges across the Short, retained dialogue ranges reserved,
   chronological order inside a sentence, enough headroom for measured speech,
   expansion within the same shot before borrowing anything else, and no
   long static holds. Every placed shot keeps its reason and evidence.
"""
from __future__ import annotations

import json
import re

from .client import GeminiClient, text_part, video_part
from .schemas import MatchingResult

MATCH_VERSION = 1
MAX_VIDEO_PARTS = 10
MIN_RELEVANCE = 0.45
MIN_SHOT_SECONDS = 0.6
MAX_STATIC_SECONDS = 6.0
FOOTAGE_HEADROOM = 1.2

MATCH_SYSTEM = (
    "You are a picture editor matching narration to footage. For each sentence, watch its candidate clips and "
    "choose only footage that visibly shows what the sentence says (the action, object or consequence), not "
    "merely the same character elsewhere. Pick in/out points that include the whole action and avoid cutting "
    "away before it is visible. Report what each chosen shot actually shows and why it fits. If nothing shows "
    "the narrated action, answer unsupported; never force an unrelated shot. Read times from the burned-in "
    "source timecode."
)


def _tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "of", "to", "in", "on", "his", "her", "their", "is", "with", "at", "as", "he", "she", "they", "it"}
    return {w for w in re.findall(r"[a-z]+", text.casefold()) if len(w) > 2 and w not in stop}


def candidates_for(sentence: dict, inventory: dict, scenes: list[dict], wide: bool = False,
                   exclude: set[str] | None = None) -> list[dict]:
    """Shots overlapping the cited events (plus neighbours when wide), locally scored."""
    events = {e["id"]: e for e in inventory["events"]}
    cited = [events[i] for i in sentence["event_ids"] if i in events]
    margin = 8.0 if wide else 1.0
    want = _tokens(sentence["text"] + " " + sentence.get("visual_requirement", ""))
    result = []
    for shot in scenes:
        a, b = float(shot["start"]), float(shot["end"])
        if exclude and shot["id"] in exclude:
            continue
        best = None
        for event in cited:
            overlap = min(b, event["end"]) - max(a, event["start"])
            near = overlap > 0 or (a < event["end"] + margin and b > event["start"] - margin)
            if not near:
                continue
            words = _tokens(event["visible_action"] + " " + " ".join(event.get("objects", [])))
            lexical = len(want & words) / max(1, len(want))
            score = max(0.0, overlap) / max(0.1, b - a) * 0.7 + lexical * 0.3
            if best is None or score > best[0]:
                best = (score, event)
        if best:
            overlaps = min(b, best[1]["end"]) - max(a, best[1]["start"]) > 0
            result.append({"shot_id": shot["id"], "start": a, "end": b, "local_score": round(best[0], 3),
                           "event_id": best[1]["id"], "relation": "overlaps_event" if overlaps else "adjacent_context",
                           "event_observation": best[1]["visible_action"]})
    result.sort(key=lambda c: -c["local_score"])
    return sorted(result[:12 if wide else 8], key=lambda c: c["start"])


def _merge_ranges(candidates: list[dict], gap: float = 1.0) -> list[list[float]]:
    ranges = []
    for c in sorted(candidates, key=lambda c: c["start"]):
        if ranges and c["start"] <= ranges[-1][1] + gap:
            ranges[-1][1] = max(ranges[-1][1], c["end"])
        else:
            ranges.append([c["start"], c["end"]])
    return ranges


def _batches(items: list[tuple[dict, list[dict]]]) -> list[list[tuple[dict, list[dict]]]]:
    batches, current = [], []
    for item in items:
        trial = current + [item]
        if current and (len(_merge_ranges([c for _, cs in trial for c in cs])) > MAX_VIDEO_PARTS or len(trial) > 6):
            batches.append(current)
            trial = [item]
        current = trial
    if current:
        batches.append(current)
    return batches


def validate_choices(sentence_id: str, match: dict, candidates: list[dict]) -> dict:
    allowed = {c["shot_id"]: c for c in candidates}
    kept, notes = [], []
    for choice in match.get("shots", []):
        shot = allowed.get(choice["shot_id"])
        if shot is None:
            notes.append(f"rejected unknown or non-candidate shot {choice['shot_id']!r}")
            continue
        a, b = choice["start"], choice["end"]
        if a < shot["start"] - 0.3 or b > shot["end"] + 0.3 or b <= a:
            notes.append(f"rejected {shot['shot_id']} range {a:.2f}–{b:.2f}s outside shot {shot['start']:.2f}–{shot['end']:.2f}s")
            continue
        a, b = max(a, shot["start"]), min(b, shot["end"])
        if b - a < min(MIN_SHOT_SECONDS, shot["end"] - shot["start"]) - 1e-6:
            notes.append(f"rejected {shot['shot_id']}: chosen range shorter than {MIN_SHOT_SECONDS}s")
            continue
        if choice["relevance"] < MIN_RELEVANCE:
            notes.append(f"rejected {shot['shot_id']}: relevance {choice['relevance']:.2f} below {MIN_RELEVANCE}")
            continue
        anchor = choice["anchor"]
        if not a - 0.5 <= anchor <= b + 0.5:
            notes.append(f"{shot['shot_id']}: anchor {anchor:.2f}s outside the chosen range; centred")
            anchor = (a + b) / 2
        kept.append({**choice, "start": round(a, 3), "end": round(b, 3), "anchor": round(min(max(anchor, a), b), 3),
                     "event_id": shot["event_id"], "shot_bounds": [shot["start"], shot["end"]]})
    verdict = match.get("verdict", "unsupported")
    if not kept:
        verdict = "unsupported"
    return {"sentence_id": sentence_id, "verdict": verdict, "shots": kept, "problem": match.get("problem", ""),
            "validation_notes": notes}


def match_sentences(client: GeminiClient, file, beats: list[dict], inventory: dict, scenes: list[dict], *,
                    only: set[str] | None = None, wide: bool = False, exclude: dict | None = None,
                    log=print) -> dict:
    """Return {sentence_id: validated match}. `exclude` maps sentence_id -> shot IDs to avoid."""
    fps = client.settings.analysis["match_fps"]
    exclude = exclude or {}
    items = []
    for beat in beats:
        for sentence in beat.get("sentences", []):
            if only is not None and sentence["sentence_id"] not in only:
                continue
            cands = candidates_for(sentence, inventory, scenes, wide=wide, exclude=set(exclude.get(sentence["sentence_id"], ())))
            items.append((sentence, cands))
    results = {}
    for sentence, cands in items:
        if not cands:
            results[sentence["sentence_id"]] = {"sentence_id": sentence["sentence_id"], "verdict": "unsupported",
                                                "shots": [], "problem": "no indexed shot overlaps the cited events",
                                                "validation_notes": []}
    items = [(s, c) for s, c in items if c]
    for number, batch in enumerate(_batches(items), 1):
        ranges = _merge_ranges([c for _, cs in batch for c in cs])
        parts = [video_part(file, a, b, fps) for a, b in ranges]
        listing = [{"sentence_id": s["sentence_id"], "sentence": s["text"],
                    "visual_requirement": s.get("visual_requirement", ""), "cited_event_ids": s["event_ids"],
                    "candidate_shots": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()
                                         if k in ("shot_id", "start", "end", "event_id", "relation", "event_observation")} for c in cs]}
                   for s, cs in batch]
        log(f"Gemini matching: batch {number} ({len(batch)} sentence(s), {len(parts)} clip(s))")
        result = client.generate(
            "matching", MatchingResult,
            parts + [text_part("The clips above cover these source ranges in order: "
                               + json.dumps([[round(a, 2), round(b, 2)] for a, b in ranges])
                               + ". For each sentence choose up to three candidate shots, each with start/end inside "
                               "the shot, the anchor second where the narrated action is clearest, focus_x of the key "
                               "subject, what it shows, the reason it fits, and relevance (1 = shows exactly the "
                               "narrated action). Order shots as they should play.\n"
                               + json.dumps(listing, ensure_ascii=False))],
            system=MATCH_SYSTEM, temperature=0.1, max_output_tokens=8192,
            resolution="low", thinking="medium")
        returned = {m.sentence_id: m.model_dump() for m in result.matches}
        for sentence, cands in batch:
            match = returned.get(sentence["sentence_id"], {"verdict": "unsupported", "shots": [],
                                                           "problem": "matcher returned no answer for this sentence"})
            results[sentence["sentence_id"]] = validate_choices(sentence["sentence_id"], match, cands)
    return results


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def plan_shots(beats: list[dict], matches: dict, rate: float, *, measured: dict | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Place footage for every sentence. Returns (beats, evidence_map, shortfalls).

    `measured` maps sentence_id -> measured spoken seconds (after synthesis);
    before synthesis the measured voice rate estimates sentence length.
    """
    measured = measured or {}
    used: list[tuple[float, float, str]] = []
    for beat in beats:
        if beat["kind"] == "dialogue":
            shot = beat["shots"][0]
            used.append((shot["start"], shot["end"], beat["id"]))
    evidence, shortfalls = [], []
    for beat in beats:
        if beat["kind"] != "narration":
            continue
        shots = []
        for index, sentence in enumerate(beat["sentences"]):
            sid = sentence["sentence_id"]
            match = matches.get(sid, {"verdict": "unsupported", "shots": []})
            words = len(re.findall(r"\b[\w’'-]+\b", sentence["text"]))
            need = measured.get(sid, words / rate + 0.15)
            want = need * FOOTAGE_HEADROOM
            chosen, total, notes = [], 0.0, []
            for choice in sorted(match["shots"], key=lambda c: -c["relevance"]):
                a, b = choice["start"], choice["end"]
                for ua, ub, owner in used:
                    if _overlap((a, b), (ua, ub)) > 0.25:
                        # Keep the larger non-overlapping remainder when it still holds the anchor.
                        left, right = (a, min(b, ua)), (max(a, ub), b)
                        part = max(left, right, key=lambda r: r[1] - r[0])
                        if part[1] - part[0] >= MIN_SHOT_SECONDS and part[0] <= choice["anchor"] <= part[1]:
                            notes.append(f"{choice['shot_id']} trimmed to avoid reusing footage from {owner}")
                            a, b = part
                        else:
                            a = b = None
                            notes.append(f"{choice['shot_id']} skipped: footage already used by {owner}")
                            break
                if a is None:
                    continue
                chosen.append({**choice, "start": a, "end": b})
                total += b - a
                if total >= want:
                    break
            # Expand within the same matched shot before giving up (action context, no new claims).
            for item in sorted(chosen, key=lambda c: -c["relevance"]):
                if total >= want:
                    break
                lo, hi = item["shot_bounds"]
                for side in ("end", "start"):
                    if total >= want:
                        break
                    room = (hi - item["end"]) if side == "end" else (item["start"] - lo)
                    for ua, ub, _ in used:
                        if side == "end" and ua >= item["end"]:
                            room = min(room, ua - item["end"])
                        if side == "start" and ub <= item["start"]:
                            room = min(room, item["start"] - ub)
                    for other in chosen:
                        if other is item:
                            continue
                        if side == "end" and other["start"] >= item["end"]:
                            room = min(room, other["start"] - item["end"])
                        if side == "start" and other["end"] <= item["start"]:
                            room = min(room, item["start"] - other["end"])
                    grow = max(0.0, min(room, want - total))
                    if grow > 0.05:
                        item[side] = item[side] + grow if side == "end" else item[side] - grow
                        total += grow
                        notes.append(f"{item['shot_id']} extended at its {side} by {grow:.2f}s within the same shot")
            for item in chosen:
                if item["end"] - item["start"] > MAX_STATIC_SECONDS and len(chosen) > 1:
                    notes.append(f"{item['shot_id']} is long; renderer trims it around the action anchor")
                used.append((item["start"], item["end"], sid))
            chosen.sort(key=lambda c: c["start"])
            placed = [{"start": round(c["start"], 3), "end": round(c["end"], 3), "focus_x": c["focus_x"],
                       "anchor": c["anchor"], "sentence_index": index, "sentence_id": sid, "shot_id": c["shot_id"],
                       "event_id": c["event_id"], "shows": c["shows"], "reason": c["reason"],
                       "relevance": c["relevance"], "shot_bounds": c["shot_bounds"]} for c in chosen]
            shots += placed
            entry = {"sentence_id": sid, "beat_id": beat["id"], "text": sentence["text"],
                     "event_ids": sentence["event_ids"], "visual_requirement": sentence.get("visual_requirement", ""),
                     "verdict": match["verdict"] if placed else "unsupported", "estimated_seconds": round(need, 2),
                     "footage_seconds": round(total, 2), "shots": placed, "problem": match.get("problem", ""),
                     "notes": match.get("validation_notes", []) + notes}
            evidence.append(entry)
            if not placed or total < need * 0.92:
                shortfalls.append({"sentence_id": sid, "beat_id": beat["id"], "needed": round(need, 2),
                                   "available": round(total, 2), "verdict": entry["verdict"]})
        beat["shots"] = shots
    return beats, evidence, shortfalls
