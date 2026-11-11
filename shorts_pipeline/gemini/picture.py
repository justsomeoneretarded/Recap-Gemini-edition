"""Hybrid scripting: picture-first backbone + story-first context, reconciled.

1. Clip plan (E1): Gemini watches each selected event's candidate shots and
   cuts concrete clips (in/out, action anchor, subject position, role).
2. Script A (picture-first): narration written line by line *for* consecutive
   clips, within each clip group's spoken-word capacity, naming the key word
   that the clip's anchor frame shows.
3. Script B (story-first): the existing evidence-referenced script, which can
   carry motives, stakes and context from the story map and notes.
4. Reconciliation: A is the timing backbone. B contributes context, either
   folded into an action line or as a separate context line placed over an
   unused, related cover clip. Code enforces every rule: action lines keep
   their exact clips, context cites a valid basis, cover clips are used once,
   context stays a minority of the narration, and all script rules still apply.
"""
from __future__ import annotations

import json
import math

from .client import GeminiClient, text_part, video_part
from .matching import _merge_ranges
from .schemas import ClipPlan, PictureScript, ReconciledScript, ScriptBeat, ScriptSentence
from .script import WORD, count_words, validate_beats
from .story import evidence_parts, with_repair

MIN_CLIP_SECONDS = 0.6
MAX_CONTEXT_SHARE = 0.35
MIN_BACKBONE_SHARE = 0.6

CLIP_SYSTEM = (
    "You are a picture editor cutting clips for a movie recap. For each event, watch its candidate shots and "
    "cut the clips that show it best: the complete action (do not cut away before it is visible), plus a strong "
    "reaction or detail shot when one exists. Read times from the burned-in source timecode. Describe exactly "
    "what each clip shows; never describe what is not visible."
)
PICTURE_SYSTEM = (
    "You are a movie recapper writing narration to picture. Each line plays over specific clips, so it must "
    "describe what those clips show, in vivid, concrete, concise recap language, and fit their duration. Name "
    "the key word (action or object) that the clip's anchor frame shows. No speculation, no invented names, no "
    "channel talk."
)
RECONCILE_SYSTEM = (
    "You are the head writer merging two scripts into one gripping recap. Script A is written to picture and "
    "fixes the timing: its action lines keep their exact clips. Script B tells the story with context, motives "
    "and stakes. Keep A's backbone, improve its wording, and bring in B's essential context: fold it into an "
    "action line when it fits the time, or add a short context line over an unused cover clip just before the "
    "action it explains. Context must come from the story notes or cited story-map inferences; never invent. "
    "Keep it action-led: context is a minority of the narration. Keep the hook first and land the ending."
)


def _candidates(event: dict, scenes: list[dict]) -> list[dict]:
    return [{"shot_id": s["id"], "start": float(s["start"]), "end": float(s["end"])} for s in scenes
            if float(s["end"]) > event["start"] - 1.0 and float(s["start"]) < event["end"] + 1.0]


def _validate_clip(choice: dict, shots: dict) -> tuple[dict | None, str | None]:
    shot = shots.get(choice["shot_id"])
    if shot is None:
        return None, f"non-candidate shot {choice['shot_id']!r}"
    a, b = choice["start"], choice["end"]
    if a < shot["start"] - 0.3 or b > shot["end"] + 0.3 or b <= a:
        return None, f"{shot['shot_id']} range {a:.2f}–{b:.2f}s outside the shot"
    a, b = max(a, shot["start"]), min(b, shot["end"])
    if b - a < min(MIN_CLIP_SECONDS, shot["end"] - shot["start"]) - 1e-6:
        return None, f"{shot['shot_id']} clip shorter than {MIN_CLIP_SECONDS}s"
    anchor = min(max(choice["anchor"], a), b)
    return {**choice, "start": round(a, 3), "end": round(b, 3), "anchor": round(anchor, 3),
            "shot_bounds": [shot["start"], shot["end"]]}, None


def plan_clips(client: GeminiClient, file, editorial: dict, inventory: dict, scenes: list[dict], log=print) -> dict:
    """Cut clips for every selected event, in playback order, without overlaps."""
    events = {e["id"]: e for e in inventory["events"]}
    order = [s["event_id"] for s in editorial["selected"] if s["event_id"] in events]
    fps = client.settings.analysis["match_fps"]
    items = [(events[i], _candidates(events[i], scenes)) for i in order]
    batches, current = [], []
    for item in items:
        trial = current + [item]
        if current and (len(_merge_ranges([c for _, cs in trial for c in cs])) > 10 or len(trial) > 6):
            batches.append(current)
            trial = [item]
        current = trial
    if current:
        batches.append(current)
    raw: dict[str, list] = {}
    notes: list[str] = []
    for number, batch in enumerate(batches, 1):
        ranges = _merge_ranges([c for _, cs in batch for c in cs if cs])
        parts = [video_part(file, a, b, fps) for a, b in ranges]
        listing = [{"event_id": e["id"], "observed": e["visible_action"], "start": e["start"], "end": e["end"],
                    "peak": e.get("peak"), "candidate_shots": cs} for e, cs in batch]
        log(f"Gemini clip plan: batch {number}/{len(batches)} ({len(batch)} event(s), {len(parts)} clip range(s))")
        result = client.generate(
            "matching:clips", ClipPlan,
            parts + [text_part("The clips above cover these source ranges in order: "
                               + json.dumps([[round(a, 2), round(b, 2)] for a, b in ranges])
                               + ". For each event cut 1-3 clips (start/end inside one candidate shot) in playback "
                               "order: the full key action first, then an optional reaction/detail. Give anchor, "
                               "focus_x, role, quality and exactly what each clip shows.\n"
                               + json.dumps(listing, ensure_ascii=False))],
            system=CLIP_SYSTEM, temperature=0.1, max_output_tokens=16384, resolution="low", thinking="medium")
        for entry in result.events:
            raw[entry.event_id] = [c.model_dump() for c in entry.clips]
    clips, used = [], []
    for event, cands in items:
        shots = {c["shot_id"]: c for c in cands}
        accepted = []
        for choice in raw.get(event["id"], []):
            clip, problem = _validate_clip(choice, shots)
            if problem:
                notes.append(f"{event['id']}: rejected clip ({problem})")
                continue
            if any(min(clip["end"], b) - max(clip["start"], a) > 0.25 for a, b in used):
                notes.append(f"{event['id']}: rejected clip {clip['start']:.2f}–{clip['end']:.2f}s overlapping another clip")
                continue
            accepted.append(clip)
            used.append((clip["start"], clip["end"]))
        if not accepted:
            # Verified event timestamps still bound a clip; mark it so the reviewer checks it.
            for cand in cands:
                a, b = max(cand["start"], event["start"]), min(cand["end"], event["end"])
                if b - a >= MIN_CLIP_SECONDS and not any(min(b, ub) - max(a, ua) > 0.25 for ua, ub in used):
                    accepted.append({"shot_id": cand["shot_id"], "start": round(a, 3), "end": round(b, 3),
                                     "anchor": round(min(max(event.get("peak", (a + b) / 2), a), b), 3),
                                     "focus_x": event.get("focus_x", 0.5), "shows": event["visible_action"],
                                     "role": "action", "quality": 0.5, "shot_bounds": [cand["start"], cand["end"]],
                                     "fallback_from_event_timestamps": True})
                    used.append((a, b))
                    notes.append(f"{event['id']}: no valid clip from the model; used verified event timestamps")
                    break
        for clip in accepted:
            clips.append({**clip, "event_id": event["id"]})
    for number, clip in enumerate(clips, 1):
        clip["clip_id"] = f"K{number:03d}"
        clip["seconds"] = round(clip["end"] - clip["start"], 3)
    if not clips:
        raise ValueError("No usable clips could be cut from the selected events.")
    return {"clips": clips, "notes": notes}


def _capacity(clip_ids: list[str], clips: dict, rate: float) -> int:
    return math.floor(sum(clips[c]["seconds"] for c in clip_ids) * rate)


def _has_word(text: str, word: str) -> bool:
    target = word.casefold().strip(".,!?;:'\"")
    return bool(target) and any(w.casefold() == target for w in WORD.findall(text))


def validate_picture(result: PictureScript, clips: list[dict], rate: float, budget_high: int) -> list[dict]:
    order = {c["clip_id"]: i for i, c in enumerate(clips)}
    by_id = {c["clip_id"]: c for c in clips}
    lines, used, last = [], set(), -1
    for number, line in enumerate(result.lines, 1):
        ids = line.clip_ids
        if not 1 <= len(ids) <= 3 or any(i not in order for i in ids):
            raise ValueError(f"line {number} must use 1-3 known clip IDs, got {ids}")
        index = [order[i] for i in ids]
        if index != list(range(index[0], index[0] + len(ids))):
            raise ValueError(f"line {number} clips {ids} are not consecutive")
        if used & set(ids):
            raise ValueError(f"line {number} reuses clips {sorted(used & set(ids))}")
        if index[0] < last:
            raise ValueError(f"line {number} goes back in the clip order; keep playback order")
        last = index[-1]
        used |= set(ids)
        words = count_words(line.text)
        limit = math.ceil(_capacity(ids, by_id, rate) * 1.1) + 1
        if not 3 <= words <= limit:
            raise ValueError(f"line {number} has {words} words but its clips fit 3–{limit}")
        if not _has_word(line.text, line.key_word):
            raise ValueError(f"line {number} key_word {line.key_word!r} does not appear in its text")
        lines.append({"line_id": f"L{number:02d}", "clip_ids": ids, "text": " ".join(line.text.split()),
                      "key_word": line.key_word, "event_ids": sorted({by_id[i]["event_id"] for i in ids}),
                      "capacity_words": limit})
    total = sum(count_words(l["text"]) for l in lines)
    if total < 0.4 * budget_high:
        raise ValueError(f"script A has {total} words; use more clips (target at least {math.ceil(0.4 * budget_high)})")
    return lines


def write_picture_script(client: GeminiClient, clips: list[dict], story_map: dict, editorial: dict, rate: float,
                         budget_words: tuple[int, int], inventory: dict, speech_pool: dict, request: dict) -> list[dict]:
    listing = [{"clip_id": c["clip_id"], "event_id": c["event_id"], "seconds": c["seconds"],
                "word_capacity": math.floor(c["seconds"] * rate), "role": c["role"], "shows": c["shows"]}
               for c in clips]
    names = [{k: ch[k] for k in ("label", "name", "name_basis")} for ch in story_map.get("characters", []) if ch.get("name")]
    target = math.floor(budget_words[1] * 0.8)
    prompt = (f"Write script A, line by line, over these clips in order. Each line covers 1-3 consecutive clips and "
              f"must fit their summed word_capacity (spoken at {rate:.2f} words/second). Aim for about {target} words "
              "in total; skip weak clips rather than padding (skipped clips become cover footage). Open on the hook "
              f"({editorial['hook_event_id']}). Use only the names listed.\n"
              + json.dumps({"clips": listing, "named_characters": names}, ensure_ascii=False))
    lines, _ = with_repair(client, "writing:picture", PictureScript, [text_part(prompt)], PICTURE_SYSTEM,
                           lambda r: validate_picture(r, clips, rate, budget_words[1]),
                           temperature=0.6, max_output_tokens=32768, thinking="medium",
                           cached_prefix=evidence_parts(inventory, speech_pool, request))
    return lines


def validate_reconciled(result: ReconciledScript, *, lines: list[dict], clips: list[dict], story_map: dict,
                        inventory: dict, speech_pool: dict, allowed_quotes: list[str], notes: str, rate: float,
                        budget_words: tuple[int, int]) -> tuple[list[dict], dict]:
    a_lines = {l["line_id"]: l for l in lines}
    by_clip = {c["clip_id"]: c for c in clips}
    backbone_clips = {i for l in lines for i in l["clip_ids"]}
    cover_pool = {c["clip_id"] for c in clips} - backbone_clips
    inferred = {e["event_id"] for e in story_map["events"] if e.get("inferred_context")}
    known = {e["id"] for e in inventory["events"]}
    used_lines, used_cover, order_index = set(), set(), {c["clip_id"]: i for i, c in enumerate(clips)}
    beats, details, last_action = [], [], -1
    for number, beat in enumerate(result.beats):
        if beat.kind == "dialogue":
            beats.append(ScriptBeat(beat_id=beat.beat_id, kind="dialogue", speech_id=beat.speech_id))
            details.append(None)
            continue
        if not beat.lines:
            raise ValueError(f"narration beat {beat.beat_id} has no lines")
        sentences, info = [], []
        for line in beat.lines:
            if line.kind == "action":
                source = a_lines.get(line.source_line_id)
                if source is None or line.source_line_id in used_lines:
                    raise ValueError(f"action line uses unknown or repeated source_line_id {line.source_line_id!r}")
                if list(line.clip_ids) != source["clip_ids"]:
                    raise ValueError(f"action line {line.source_line_id} must keep its clips {source['clip_ids']}")
                if number > 0 and order_index[source["clip_ids"][0]] < last_action:
                    raise ValueError(f"action line {line.source_line_id} breaks playback order")
                if number > 0:
                    last_action = order_index[source["clip_ids"][0]]
                used_lines.add(line.source_line_id)
                key = line.key_word or source["key_word"]
                if not _has_word(line.text, key):
                    raise ValueError(f"action line {line.source_line_id} must keep a key word in its text ({key!r})")
                events = source["event_ids"]
            else:
                bad = [c for c in line.clip_ids if c not in cover_pool or c in used_cover]
                if not line.clip_ids or bad:
                    raise ValueError(f"context line must use unused cover clips from {sorted(cover_pool - used_cover)}; got {bad or 'none'}")
                used_cover |= set(line.clip_ids)
                events = [i for i in line.event_ids if i in known] or sorted({by_clip[c]["event_id"] for c in line.clip_ids})
                if line.basis == "story_notes" and not notes.strip():
                    raise ValueError("a context line cites story_notes but there are none")
                if line.basis == "inferred_context" and not set(line.event_ids) & inferred:
                    raise ValueError(f"context line cites inferred_context without an event that has a story-map "
                                     f"inference: {line.event_ids}")
                key = line.key_word
            limit = math.ceil(_capacity(list(line.clip_ids), by_clip, rate) * 1.15) + 2
            if count_words(line.text) > limit:
                raise ValueError(f"{line.kind} line '{line.text[:40]}…' has {count_words(line.text)} words; its clips fit {limit}")
            sentences.append(ScriptSentence(text=line.text, event_ids=events, visual_requirement="",
                                            claim_basis="observed" if line.basis in ("observed", "inferred_context")
                                            else line.basis))
            info.append({"kind": line.kind, "clip_ids": list(line.clip_ids), "key_word": key,
                         "basis": line.basis, "source_line_id": line.source_line_id})
        beats.append(ScriptBeat(beat_id=beat.beat_id, kind="narration", delivery=beat.delivery, sentences=sentences))
        details.append(info)
    if len(used_lines) < MIN_BACKBONE_SHARE * len(lines):
        raise ValueError(f"only {len(used_lines)} of {len(lines)} footage-first lines kept; keep at least "
                         f"{math.ceil(MIN_BACKBONE_SHARE * len(lines))} (the backbone)")
    normalized = validate_beats(beats, inventory, speech_pool, allowed_quotes=allowed_quotes, budget_words=budget_words)
    narration = [b for b in normalized if b["kind"] == "narration"]
    infos = [d for d in details if d is not None]
    context_words = total_words = 0
    for beat, info in zip(narration, infos):
        for sentence, extra in zip(beat["sentences"], info):
            sentence.update(line_kind=extra["kind"], clip_ids=extra["clip_ids"], key_word=extra["key_word"],
                            context_basis=extra["basis"], source_line_id=extra["source_line_id"],
                            visual_requirement=" / ".join(by_clip[c]["shows"] for c in extra["clip_ids"]))
            words = count_words(sentence["text"])
            total_words += words
            context_words += words if extra["kind"] == "context" else 0
    if total_words and context_words / total_words > MAX_CONTEXT_SHARE:
        raise ValueError(f"context is {context_words / total_words:.0%} of the narration; keep it under "
                         f"{MAX_CONTEXT_SHARE:.0%} so the recap stays action-led")
    return normalized, _matches(normalized, by_clip)


def _matches(beats: list[dict], clips: dict) -> dict:
    matches = {}
    for beat in beats:
        for sentence in beat.get("sentences", []):
            ids = sentence["clip_ids"]
            cover = sentence["line_kind"] == "context"
            key_clip = None
            if not cover:
                key_clip = max(ids, key=lambda c: (clips[c]["role"] == "action", clips[c]["quality"]))
            shots = [{"shot_id": clips[c]["shot_id"], "clip_id": c, "start": clips[c]["start"], "end": clips[c]["end"],
                      "anchor": clips[c]["anchor"], "focus_x": clips[c]["focus_x"], "shows": clips[c]["shows"],
                      "reason": ("cover footage for context" if cover else "picture-first line written for this clip"),
                      "relevance": round(max(0.6, clips[c]["quality"]), 3), "event_id": clips[c]["event_id"],
                      "shot_bounds": clips[c]["shot_bounds"], "cover": cover, "key_clip": c == key_clip,
                      "priority": 1 if c == key_clip else 0} for c in ids]
            matches[sentence["sentence_id"]] = {"sentence_id": sentence["sentence_id"], "verdict": "supported",
                                                "shots": shots, "problem": "", "validation_notes": []}
    return matches


def reconcile(client: GeminiClient, *, lines: list[dict], story_script: dict, clips: list[dict], story_map: dict,
              editorial: dict, inventory: dict, speech_pool: dict, request: dict, rate: float,
              budget_words: tuple[int, int]) -> dict:
    backbone = {i for l in lines for i in l["clip_ids"]}
    cover = [{"clip_id": c["clip_id"], "event_id": c["event_id"], "seconds": c["seconds"], "role": c["role"],
              "shows": c["shows"], "word_capacity": math.floor(c["seconds"] * rate)}
             for c in clips if c["clip_id"] not in backbone]
    context = [{"event_id": e["event_id"], "inferences": e["inferred_context"]}
               for e in story_map["events"] if e.get("inferred_context")]
    prompt = (
        f"Merge the scripts into the final narration ({budget_words[0]}–{budget_words[1]} words, dialogue clips "
        f"excluded). Use kind=action for lines based on script A (give source_line_id; keep its clip_ids exactly; you "
        "may sharpen the wording and fold in brief context if it still fits the clips). Use kind=context for "
        "context from script B, each over unused cover clips, with basis story_notes or inferred_context (cite "
        f"event_ids). Keep context under {MAX_CONTEXT_SHARE:.0%} of the words and at least "
        f"{math.ceil(MIN_BACKBONE_SHARE * len(lines))} of script A's lines. Place approved dialogue clips as "
        "kind=dialogue beats. Group lines into beats of 2-4 lines.\n"
        + json.dumps({"script_a": [{k: l[k] for k in ("line_id", "clip_ids", "text", "key_word", "capacity_words")}
                                   for l in lines],
                      "script_b": [{"beat_id": b["id"], "kind": b["kind"], "text": b["text"],
                                    "speech_id": b.get("speech_id", "")} for b in story_script["beats"]],
                      "cover_clips": cover, "story_map_inferences": context,
                      "approved_dialogue_speech_ids": editorial["dialogue_speech_ids"],
                      "ending_mode": editorial["ending_mode"]}, ensure_ascii=False))
    # Running a little short only shortens the Short; running over breaks timing. So the
    # merge keeps the strict maximum but a looser minimum than the first draft.
    merge_budget = (math.floor(budget_words[1] * 0.6), budget_words[1])
    (beats, matches), errors = with_repair(
        client, "writing:reconcile", ReconciledScript, [text_part(prompt)], RECONCILE_SYSTEM,
        lambda r: validate_reconciled(r, lines=lines, clips=clips, story_map=story_map, inventory=inventory,
                                      speech_pool=speech_pool, allowed_quotes=list(editorial["dialogue_speech_ids"]),
                                      notes=request.get("story_notes", ""), rate=rate, budget_words=merge_budget),
        attempts=3, temperature=0.5, max_output_tokens=32768, thinking="high",
        cached_prefix=evidence_parts(inventory, speech_pool, request))
    return {"beats": beats, "matches": matches, "validation_retries": errors}


def build_hybrid(client: GeminiClient, file, *, story_script: dict, editorial: dict, story_map: dict, inventory: dict,
                 scenes: list[dict], speech_pool: dict, request: dict, rate: float, budget_words: tuple[int, int],
                 log=print) -> dict:
    plan = plan_clips(client, file, editorial, inventory, scenes, log=log)
    lines = write_picture_script(client, plan["clips"], story_map, editorial, rate, budget_words, inventory,
                                 speech_pool, request)
    merged = reconcile(client, lines=lines, story_script=story_script, clips=plan["clips"], story_map=story_map,
                       editorial=editorial, inventory=inventory, speech_pool=speech_pool, request=request, rate=rate,
                       budget_words=budget_words)
    return {"clip_plan": plan, "picture_script": lines, **merged}


def key_word_times(beat: dict) -> dict[int, float]:
    """Seconds from each sentence's start to its key word, from measured word timings."""
    from ..render import sentence_spans
    from ..voice import _normal
    if not beat.get("words"):
        return {}
    spans = sentence_spans(beat)
    times, index = {}, 0
    for number, sentence in enumerate(beat["sentences"]):
        tokens = [t for t in sentence["text"].split() if _normal(t)]
        words = beat["words"][index:index + len(tokens)]
        index += len(tokens)
        key = _normal(sentence.get("key_word", ""))
        hit = next((w for w in words if key and _normal(w["word"]) == key), None)
        if hit is not None:
            times[number] = max(0.0, float(hit["start"]) - spans[number][0])
    return times


def apply_key_times(beats: list[dict]) -> int:
    """Attach the measured key-word time to each sentence's key clip. Returns how many were set."""
    count = 0
    for beat in beats:
        if beat["kind"] != "narration":
            continue
        times = key_word_times(beat)
        for shot in beat.get("shots", []):
            shot.pop("key_time", None)
            index = int(shot.get("sentence_index", -1))
            if shot.get("key_clip") and index in times:
                shot["key_time"] = round(times[index], 3)
                count += 1
    return count

