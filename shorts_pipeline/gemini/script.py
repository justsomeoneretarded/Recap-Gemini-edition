"""Stage D: evidence-referenced narration, an independent script review, and
targeted beat rewrites used by the timing and correction loops.

The writer receives only the approved story map/editorial plan. Every sentence
cites inventory events and states what must be visible while it plays; code
rejects uncited sentences, invented quotations, unknown dialogue IDs, channel
filler and word budgets that the measured voice cannot fit.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from ..planner import _DANGLING_SENTENCE_ENDINGS
from .client import GeminiClient, text_part
from .schemas import BeatRewrite, ScriptBeat, ScriptDraft, ScriptRevision
from .story import evidence_parts, with_repair

STYLE_GUIDE = (
    "Style evidence from eight reference recaps (docs/reference-analysis.md): open on a concrete, visible "
    "action or contradiction (a chisel, a cup, a rifle), not a dossier; explain only the minimum cause needed, "
    "delaying backstory until the hook has landed; move in cause-and-effect steps where each beat changes the "
    "situation; let selected original dialogue carry threats or reversals; name specific objects and verbs the "
    "camera can show; end with a story-appropriate payoff, reveal, reflection or renewed threat."
)
WRITER_SYSTEM = (
    "You are a skilled human movie recapper writing narration for a short recap video. Write like a "
    "storyteller, not a summarizer: concrete actions, clear motivations only when supported, concise complete "
    "sentences, natural suspense, varied phrasing, few adjectives. Never invent scenes, identities or dialogue; "
    "never add greetings, channel talk, rankings or calls to subscribe. Every sentence must cite the inventory "
    "event IDs it describes and say what must be visible while it is heard. " + STYLE_GUIDE
)
SCRIPT_REVIEW_SYSTEM = (
    "You are a demanding script editor reviewing a recap script against its evidence. You did not write it. "
    "Find awkward phrasing, repetition, weak opening, gaps in cause and effect, unsupported claims, confusing "
    "jumps, pacing problems and a weak ending, then return the improved complete script. Keep every sentence "
    "tied to evidence event IDs; remove anything unsupported rather than softening it."
)
BANNED = ("subscribe", "like and share", "welcome back", "in this video", "hey guys", "let's dive",
          "smash that", "comment below", "stay tuned", "today we", "this movie")
MAX_SENTENCE_WORDS = 32
WORD = re.compile(r"\b[\w’'-]+\b")


def count_words(text: str) -> int:
    return len(WORD.findall(text))


def _quotes(text: str) -> list[str]:
    return [a or b for a, b in re.findall(r"“([^”]+)”|\"([^\"]+)\"", text)]


def verified_lines(inventory: dict, speech_pool: dict) -> list[str]:
    lines = [q["text"] for q in speech_pool.values()]
    lines += [e["dialogue_heard"] for e in inventory["events"] if e.get("dialogue_status") == "asr_supported"]
    return [l for l in lines if l]


def _quote_supported(quote: str, lines: list[str]) -> bool:
    words = WORD.findall(quote.casefold())
    for line in lines:
        heard = WORD.findall(line.casefold())
        matcher = SequenceMatcher(None, words, heard, autojunk=False)
        if words and sum(b.size for b in matcher.get_matching_blocks()) / len(words) >= 0.8:
            return True
    return False


def validate_beats(beats: list[ScriptBeat], inventory: dict, speech_pool: dict, *, allowed_quotes: list[str],
                   budget_words: tuple[int, int] | None, prefix: str = "") -> list[dict]:
    """Normalize beats to storyboard-ready dicts or raise ValueError with a fix hint."""
    known = {e["id"] for e in inventory["events"]}
    lines = verified_lines(inventory, speech_pool)
    result, used_quotes = [], set()
    narration_number = dialogue_number = 0
    for beat in beats:
        if beat.kind == "dialogue":
            if beat.speech_id not in speech_pool or beat.speech_id not in allowed_quotes:
                raise ValueError(f"dialogue beat {beat.beat_id} uses speech_id {beat.speech_id!r}, which is not "
                                 f"an approved dialogue clip {allowed_quotes}")
            if beat.speech_id in used_quotes:
                raise ValueError(f"speech_id {beat.speech_id} is used twice")
            used_quotes.add(beat.speech_id)
            dialogue_number += 1
            quote = speech_pool[beat.speech_id]
            result.append({"id": f"{prefix}D{dialogue_number:02d}", "kind": "dialogue", "speech_id": quote["id"],
                           "text": quote["text"], "delivery": "normal",
                           "shots": [{"start": quote["start"], "end": quote["end"]}],
                           "words": [dict(w) for w in quote["words"]], "confidence": quote["confidence"],
                           "timing_provenance": quote["timing_provenance"]})
            continue
        if not beat.sentences:
            raise ValueError(f"narration beat {beat.beat_id} has no sentences")
        narration_number += 1
        ident = f"{prefix}N{narration_number:02d}"
        sentences = []
        for index, sentence in enumerate(beat.sentences, 1):
            text = " ".join(sentence.text.split())
            words = count_words(text)
            if not text or not 3 <= words <= MAX_SENTENCE_WORDS:
                raise ValueError(f"beat {beat.beat_id} sentence {index} has {words} words; use 3–{MAX_SENTENCE_WORDS}")
            if not re.search(r"[.!?…]['\"”’]?$", text):
                raise ValueError(f"beat {beat.beat_id} sentence {index} is not a complete punctuated sentence: {text!r}")
            last_word = WORD.findall(text)[-1].casefold()
            if last_word in _DANGLING_SENTENCE_ENDINGS:
                raise ValueError(f"beat {beat.beat_id} sentence {index} ends with an unfinished phrase: {text!r}")
            lowered = text.casefold()
            banned = [b for b in BANNED if b in lowered]
            if banned:
                raise ValueError(f"beat {beat.beat_id} contains channel filler {banned}: {text!r}")
            cited = [i for i in sentence.event_ids if i in known]
            if not cited:
                raise ValueError(f"beat {beat.beat_id} sentence {index} cites no valid inventory event "
                                 f"({sentence.event_ids}); every sentence needs evidence")
            invented = [q for q in _quotes(text) if not _quote_supported(q, lines)]
            if invented:
                raise ValueError(f"beat {beat.beat_id} quotes words not found in verified source dialogue: {invented}")
            sentences.append({"sentence_id": f"{ident}.{index}", "text": text, "event_ids": cited,
                              "visual_requirement": sentence.visual_requirement.strip(),
                              "claim_basis": sentence.claim_basis,
                              "dropped_event_ids": [i for i in sentence.event_ids if i not in known]})
        result.append({"id": ident, "kind": "narration", "delivery": beat.delivery,
                       "text": " ".join(s["text"] for s in sentences), "sentences": sentences})
    if not any(b["kind"] == "narration" for b in result):
        raise ValueError("the script has no narration")
    if budget_words:
        total = sum(count_words(b["text"]) for b in result if b["kind"] == "narration")
        low, high = budget_words
        if not low <= total <= high:
            raise ValueError(f"narration has {total} words; the measured voice fits {low}–{high} words for this "
                             "duration. Add or cut supported story beats rather than padding")
    openers = [" ".join(WORD.findall(s["text"].casefold())[:2]) for b in result for s in b.get("sentences", [])]
    repeated = {o for o in openers if openers.count(o) > 3 and o}
    if repeated:
        raise ValueError(f"repetitive sentence openings {sorted(repeated)}; vary the phrasing")
    return result


def _script_context(editorial: dict, story_map: dict) -> dict:
    selected = {s["event_id"] for s in editorial["selected"]}
    events = [e for e in story_map["events"] if e["event_id"] in selected]
    return {"editorial_plan": {k: editorial[k] for k in ("title", "description", "hook_event_id", "cold_open",
                                                          "ending_mode", "selected", "dialogue_speech_ids",
                                                          "structure_notes")},
            "selected_story_events": events, "characters": story_map.get("characters", [])}


def write_script(client: GeminiClient, inventory: dict, story_map: dict, editorial: dict, speech_pool: dict,
                 request: dict, budget_words: tuple[int, int]) -> dict:
    allowed = list(editorial["dialogue_speech_ids"])
    prefix = evidence_parts(inventory, speech_pool, request)
    instructions = (
        f"Write the narration script for the approved plan in {budget_words[0]}–{budget_words[1]} narration words "
        "(dialogue clips do not count). Group sentences into beats (2–4 sentences per narration beat), and place "
        "the approved dialogue clips as kind=dialogue beats where they land hardest. Begin with the hook event "
        "itself. If cold_open is true, follow the hook with a clear reset (e.g. 'Hours earlier, ...'). Each sentence "
        "must cite event_ids and give a concrete visual_requirement (who/what action/object must be on screen). "
        "Use claim_basis story_notes only for facts present in the notes. Spell numbers as words. Do not quote "
        "dialogue in narration unless it is verified ASR text.\n\nAPPROVED PLAN:\n"
        + json.dumps(_script_context(editorial, story_map), ensure_ascii=False))

    def validate(draft: ScriptDraft):
        return validate_beats(draft.beats, inventory, speech_pool, allowed_quotes=allowed, budget_words=budget_words), draft

    (beats, draft), errors = with_repair(client, "writing:draft", ScriptDraft, [text_part(instructions)],
                                         WRITER_SYSTEM, validate, temperature=0.7, max_output_tokens=32768,
                                         thinking="high", cached_prefix=prefix)
    record = {"title": draft.title, "description": draft.description, "draft_beats": beats,
              "draft_validation_retries": errors}
    # Independent review pass (different stage instruction, planning model).
    review_prompt = (
        "Review and improve this draft. Return problems_found, changes_made and the complete revised beats. "
        f"Keep narration within {budget_words[0]}–{budget_words[1]} words and dialogue beats to approved speech IDs."
        "\n\nAPPROVED PLAN:\n" + json.dumps(_script_context(editorial, story_map), ensure_ascii=False)
        + "\n\nDRAFT:\n" + json.dumps([b.model_dump() for b in draft.beats], ensure_ascii=False))
    try:
        (reviewed, revision), review_errors = with_repair(
            client, "planning:script_review", ScriptRevision, [text_part(review_prompt)], SCRIPT_REVIEW_SYSTEM,
            lambda r: (validate_beats(r.beats, inventory, speech_pool, allowed_quotes=allowed,
                                      budget_words=budget_words), r),
            temperature=0.4, max_output_tokens=32768, thinking="high", cached_prefix=prefix)
        record.update(beats=reviewed, review={"problems_found": revision.problems_found,
                                              "changes_made": revision.changes_made,
                                              "validation_retries": review_errors, "applied": True})
    except ValueError as exc:
        record.update(beats=beats, review={"applied": False, "error": str(exc)})
    return record


def rewrite_beat(client: GeminiClient, beat: dict, reason: str, inventory: dict, speech_pool: dict,
                 request: dict, *, max_words: int | None = None, min_words: int | None = None,
                 allowed_event_ids: list[str] | None = None, context: list[dict] | None = None) -> dict:
    """Rewrite one narration beat only; everything else stays fixed."""
    limits = []
    if max_words:
        limits.append(f"at most {max_words} words in total")
    if min_words:
        limits.append(f"at least {min_words} words")
    prompt = (f"Rewrite only narration beat {beat['id']}. Problem to fix: {reason}. "
              + (f"Keep it to {' and '.join(limits)}. " if limits else "")
              + "Keep its story contribution, cite event_ids for every sentence and give a visual_requirement. "
              "Do not add unsupported facts.\n"
              + json.dumps({"beat": {k: beat[k] for k in ("id", "text", "sentences")},
                            "neighbouring_beats": context or [],
                            "allowed_event_ids": allowed_event_ids}, ensure_ascii=False))

    def validate(result: BeatRewrite):
        fake = ScriptBeat(beat_id=beat["id"], kind="narration", delivery=beat.get("delivery", "normal"),
                          sentences=result.sentences)
        normalized = validate_beats([fake], inventory, speech_pool, allowed_quotes=[], budget_words=None)[0]
        words = count_words(normalized["text"])
        if max_words and words > max_words:
            raise ValueError(f"rewrite has {words} words; maximum is {max_words}")
        if min_words and words < min_words:
            raise ValueError(f"rewrite has {words} words; minimum is {min_words}")
        if allowed_event_ids is not None:
            extra = {i for s in normalized["sentences"] for i in s["event_ids"]} - set(allowed_event_ids)
            if extra:
                raise ValueError(f"rewrite cites events outside the allowed set: {sorted(extra)}")
        return normalized, result.rationale

    (normalized, rationale), _ = with_repair(client, "writing:rewrite", BeatRewrite, [text_part(prompt)],
                                             WRITER_SYSTEM, validate, temperature=0.5, max_output_tokens=16384,
                                             thinking="medium",
                                             cached_prefix=evidence_parts(inventory, speech_pool, request))
    ident = beat["id"]
    for sentence in normalized["sentences"]:
        sentence["sentence_id"] = ident + "." + sentence["sentence_id"].rsplit(".", 1)[1]
    return {**beat, "text": normalized["text"], "sentences": normalized["sentences"], "rewrite_rationale": rationale}
