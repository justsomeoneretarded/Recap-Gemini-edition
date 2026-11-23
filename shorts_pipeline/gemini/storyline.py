"""Story-first production for discovered footage (`source: discover`).

Discovery already knows the story: a premise, story items in film order (what happens,
why it matters) and, for each item, the exact stretches of the footage pool that show it.
This path keeps that structure instead of re-deriving it from anonymous footage:

1. Shot catalog (shots.py): every detected shot of the pool is labelled by Gemini: what is
   visible, intensity, filler or not.
2. Story script: one writer call narrates the items as a story (cold open, premise,
   cause-and-effect chain, ending), with style examples from the reference recaps and a
   per-item word budget that the item's own footage can carry.
3. Story editor: an independent critique scores the script; a weak script is rewritten once.
4. Measured narration: the voice is synthesized at the requested pace; an overlong or
   short Short is fixed by rewriting whole beats, never by deleting sentences.
5. Placement: each sentence cites the shots that show what it says; each cuts in on the
   words it illustrates and stays at most 2.2 s, gaps filled with the item's next best
   shots, so the picture changes every 1.5-2 s like the reference recaps (run 20 held one
   continuous window per sentence: filler on screen, a 17 s stretch with no cut).
6. Visual check: Gemini watches the placed windows and flags contradictions only (a wrong
   person named, a different event on screen); the affected beat is rewritten once.

Run 18 showed why the generic path is wrong here: it matched each sentence to separate
shots and deleted 12 sentences that action footage (a cut every 1-2 s) could not cover
alone, leaving disconnected, frame-by-frame lines and no premise.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..planner import _DANGLING_SENTENCE_ENDINGS
from .client import GeminiOutputError, _write_json, text_part, video_part
from .review import CONNECTIVES, play_by_play
from .script import BANNED, NARRATION_RULE, count_words
from .shots import MAX_CUT, Footage, catalog_shots, cue_times, label_shots, place_sentence, slop
from .story import with_repair

STORYLINE_VERSION = 3
SCRIPT_VERSION = 3       # Bump when the writing rules change: the script is rewritten, the shot catalog is kept.
MAX_DRAFTS = 3          # Full script drafts before the quality bar stops the run.
QUALITY_BAR = 4         # Every story editor score must reach this.
MAX_NOTE_REWRITES = 3    # Beats the story editor names get one targeted rewrite each, at most this many.
FILL = 0.95              # Narration fills this share of the target; dialogue and breaths take the rest.
ITEM_SHARE = 0.85        # An item's narration may use this share of the item's own footage.
NARRATION_SOURCE_GAIN = 0.07  # Film sound (hits, gunshots) stays audible under the voice.
MAX_DIALOGUE = 2
REFERENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "reference-data" / "transcripts"
REFERENCE_ORDER = ("V7", "V8", "V3", "V4", "V2", "V5", "V6", "V1")
GENERIC = frozenset("chapter part movie film the and with from into that this from".split())


# ------------------------------------------------------------------ schemas
class ShotCue(BaseModel):
    shot_id: str = Field(description="A shot id from the item's shot list")
    cue: str = Field("", description="The exact words of this sentence that this shot illustrates")


class StorySentence(BaseModel):
    text: str = Field(description="One complete narration sentence")
    shots: list[ShotCue] = Field(default_factory=list, description="The shots that show what this sentence says, in "
                                                                   "order, about one per 2 seconds of speech")


class StoryBeat(BaseModel):
    kind: Literal["narration", "dialogue"] = "narration"
    item_id: str = Field(description="The story item whose footage plays under this beat")
    speech_id: str = Field("", description="For kind=dialogue only: an approved film dialogue clip id")
    sentences: list[StorySentence] = Field(default_factory=list)


class StorylineScript(BaseModel):
    title: str
    description: str = Field(description="One or two sentences for the video description, no hashtags")
    beats: list[StoryBeat]


class BeatNote(BaseModel):
    beat_id: str = Field(description="The exact beat id, e.g. N03")
    note: str = Field(description="What to change in this beat")
    factual_error: bool = Field(False, description="True if the beat gets the film wrong (person, order, reason)")


class StoryCritique(BaseModel):
    hook: int = Field(ge=1, le=5, description="Does the first sentence make a viewer need to know more?")
    premise: int = Field(ge=1, le=5, description="Within the first two beats: who the hero is, what he did, why he is hunted")
    causality: int = Field(ge=1, le=5, description="Each beat follows from the last (so/but/because), no jumps")
    storytelling: int = Field(ge=1, le=5, description="Story meaning (intent, stakes, consequence), not a description of frames")
    ending: int = Field(ge=1, le=5, description="A payoff or renewed threat that lands")
    clarity: int = Field(ge=1, le=5, description="A newcomer can follow every name and event")
    impact: int = Field(3, ge=1, le=5, description="Concrete, vivid, specific words; no stock phrases or generic "
                                                   "adjectives (AI-sounding narration scores 1-2)")
    problems: list[str] = Field(default_factory=list)
    rewrite_notes: str = Field("", description="Concrete instructions for the rewrite")
    beat_notes: list[BeatNote] = Field(default_factory=list, description="Per-beat fixes, factual errors first")


class BeatText(BaseModel):
    sentences: list[StorySentence]
    rationale: str = ""


class Contradiction(BaseModel):
    sentence_id: str
    on_screen: str = Field(description="What is actually visible in that window")
    problem: str


class VisualCheck(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)


# ------------------------------------------------------------------ prompts
WRITER_SYSTEM = (
    "You are the head writer of a top movie-recap Shorts channel. Your narration is gripping storytelling: it "
    "makes a stranger understand the story and need to know what happens next. You write over real film footage "
    "of each story moment, so the picture already shows the actions; your words give them meaning. "
    + NARRATION_RULE
    + "Never invent events that are not in the film. No greetings, channel talk, rankings or calls to subscribe. "
    "Return JSON matching the schema.")
EDITOR_SYSTEM = (
    "You are a demanding story editor for movie-recap Shorts. You did not write this script. Judge it the way a "
    "viewer who never saw the film experiences it. Be strict: a 5 is publishable as is. Return JSON matching the schema.")
CHECK_SYSTEM = (
    "You check narration against the footage it plays over. Report only clear contradictions: the narration "
    "names the wrong person, or describes an event that is visibly different from what is on screen. Story meaning, "
    "motives, backstory and lines about what happens off screen are NOT contradictions. An event the narration "
    "states that is simply not visible in these shots (it may happen just off screen or a moment later, e.g. a "
    "shooting during a negotiation scene) is NOT a contradiction. Return JSON matching the schema.")

STRUCTURE = (
    "STRUCTURE (this is how the best recap Shorts work):\n"
    "1. COLD OPEN (first beat, 2-3 sentences): start inside the most gripping moment in the footage and give it "
    "its meaning at once, so the viewer needs to know why. Example of the voice: 'He presses his palm flat against "
    "the table and swings the chisel down, severing his own ring finger, because it is the only way left to buy "
    "back his life.'\n"
    "2. PREMISE (second beat): rewind with a clear time reset ('Days earlier,' / 'It all started when'). Say who "
    "the hero is, what he did, and what it costs him now (the threat, the deadline). A newcomer must understand "
    "the situation after this beat.\n"
    "3. CHAIN: then tell the remaining items in story order. Every sentence moves the story: a goal, an obstacle, "
    "a decision or a consequence. Name the characters, give their motives, and link the moments (so, but, "
    "because, until, which means, only to). When the picture is a fight, say what the fight is for and what it "
    "costs, not each blow.\n"
    "4. ENDING: close on the last item with a payoff or the renewed threat in one strong line.\n"
    "OWN FOOTAGE: each beat plays over its own item's shots, so every beat except the premise beat narrates its "
    "own item's moment (what_happens); backstory belongs in the premise beat only. Get the film right: the right "
    "people, the right order, the right reasons.\n"
    "HIGH IMPACT, NO AI SLOP: every sentence names who does what to whom, with concrete nouns and verbs a viewer "
    "can see in the shots, plus the reason or the cost. No stock phrases ('little did he know', 'all hell breaks "
    "loose', 'a deadly game', 'the clock is ticking', 'all-out war', 'unleash'), no piles of adjectives (deadly, "
    "brutal, ruthless, relentless, epic, intense). Specific beats generic: 'a pencil through the hand' beats 'a "
    "brutal fight'. A reason must be a story fact (who wants what, who did what to whom): never tack on a generic "
    "one ('because survival is his only option', 'because escape is blocked', 'because lethal force is "
    "required'). Use 'because' only for real reasons, a few times in the whole script; link the rest with so, "
    "but, until, when, or just state the consequence.\n"
    "STYLE: sentences of 10-24 words with an occasional short punch line; vivid verbs; mostly present tense; spell "
    "numbers as words; no quotation marks; never 'in this movie' or 'this film'.")


# ------------------------------------------------------------------ inputs
def clean_quote(q: dict) -> bool:
    """A film line worth playing: confidently heard, short, one speaker, a complete utterance.
    Run 19 played 'little run down there John. Ernest I still have time.' (two speakers, cut mid-word)."""
    text = " ".join(str(q.get("text", "")).split())
    words = count_words(text)
    seconds = float(q.get("end", 0)) - float(q.get("start", 0))
    return (float(q.get("confidence", 0)) >= 0.85 and 2 <= words <= 14 and 0.6 <= seconds <= 5.0
            and text[:1].isupper() and bool(re.search(r"[.!?]['\"”’]?$", text))
            and len(re.findall(r"[.!?](?=\s)", text)) <= 1)


def load_discovery(directory: Path) -> tuple[dict, list[dict]]:
    directory = Path(directory)
    plan = json.loads((directory / "story-plan.json").read_text(encoding="utf-8"))
    pool_map = json.loads((directory / "pool-map.json").read_text(encoding="utf-8"))
    return plan, pool_map


def item_ranges(pool_map: list[dict]) -> dict[str, list[tuple[float, float]]]:
    ranges: dict[str, list[tuple[float, float]]] = {}
    for p in pool_map:
        ranges.setdefault(p["item_id"], []).append((float(p["pool_start"]), float(p["pool_end"])))
    return {k: sorted(v) for k, v in ranges.items()}


def _film_words(film: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", film.lower()) if len(w) >= 4 and w not in GENERIC}


def style_examples(film: str, count: int = 2, seconds: float = 65.0, directory: Path = REFERENCE_DIR) -> list[str]:
    """Opening excerpts of reference recaps about other films (never the requested one)."""
    avoid = _film_words(film)
    examples = []
    for name in REFERENCE_ORDER:
        path = directory / f"{name}.md"
        if not path.is_file():
            continue
        segments = re.findall(r"\[(\d+(?:\.\d+)?)[–-](\d+(?:\.\d+)?)\]\s*(.+)", path.read_text(encoding="utf-8"))
        text = " ".join(t.strip() for a, _, t in segments if float(a) < seconds)
        if not text or any(re.search(rf"\b{re.escape(w)}\b", text.lower()) for w in avoid):
            continue
        examples.append(text)
        if len(examples) == count:
            break
    return examples


def _protagonist_names(premise: str, notes: str) -> set[str]:
    """Capitalized names that the premise or notes use (not sentence-initial words)."""
    names = set()
    for text in (premise, notes):
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            words = re.findall(r"[A-Za-z][\w'-]*", sentence)
            names |= {w for w in words[1:] if w[0].isupper() and len(w) >= 3 and w.lower() not in GENERIC}
    return names


class Context:
    """Everything the writer, editor and validators need, built once per run."""

    def __init__(self, plan: dict, pool_map: list[dict], shots: dict[str, dict], speech_pool: dict, request: dict,
                 rate: float, target: float):
        self.plan, self.request, self.rate, self.target = plan, request, rate, target
        self.ranges = item_ranges(pool_map)
        self.shots = shots
        # Footage a viewer should see: shots that are not filler.
        self.footage = {k: sum(s["end"] - s["start"] for s in shots.values() if s["item_id"] == k and not s["filler"])
                        for k in self.ranges}
        self.items = [i for i in plan["items"] if self.footage.get(i["item_id"], 0) >= 2.0]
        self.speech_pool = speech_pool
        notes = request.get("user_story_notes", request.get("story_notes", ""))
        self.notes = notes
        self.names = _protagonist_names(plan.get("premise", ""), notes)
        total_footage = sum(self.footage.values())
        seconds = min(target * FILL, total_footage * ITEM_SHARE)
        self.budget = (math.floor(seconds * rate * 0.86), math.floor(seconds * rate))
        self.item_max = {k: max(8, math.floor(v * rate * ITEM_SHARE)) for k, v in self.footage.items()}
        self.examples = style_examples(plan.get("film", ""))

    def item_of(self, t: float) -> str | None:
        for item, spans in self.ranges.items():
            if any(a - 0.5 <= t <= b + 0.5 for a, b in spans):
                return item
        return None

    def shot_list(self, item_id: str) -> list[dict]:
        """The item's showable shots, in film order, for the writer."""
        return [{"id": sid, "seconds": round(s["end"] - s["start"], 1), "visible": s["visible"],
                 "intensity": s["intensity"]}
                for sid, s in sorted(self.shots.items(), key=lambda kv: kv[1]["start"])
                if s["item_id"] == item_id and not s["filler"]]

    def dialogue_clips(self) -> list[dict]:
        clips = []
        for q in self.speech_pool.values():
            if not clean_quote(q):
                continue
            item = self.item_of((q["start"] + q["end"]) / 2)
            if item:
                clips.append({"speech_id": q["id"], "item_id": item, "start": round(q["start"], 2),
                              "end": round(q["end"], 2), "words": q["text"]})
        return clips

    def payload(self) -> dict:
        return {
            "film": self.plan.get("film"), "premise": self.plan.get("premise", ""),
            "user_story_notes_untrusted": self.notes, "brief_untrusted": self.request.get("brief", ""),
            "title": self.request.get("title", ""),
            "story_items_in_film_order": [
                {"item_id": i["item_id"], "title": i["title"], "what_happens": i["what_happens"],
                 "why_it_matters": i.get("story_context", ""),
                 "footage_seconds": round(self.footage[i["item_id"]], 1),
                 "max_narration_words": self.item_max[i["item_id"]],
                 "shots": self.shot_list(i["item_id"])} for i in self.items],
            "dialogue_clips": self.dialogue_clips()}

    def prompt(self, extra: str = "") -> list[dict]:
        examples = "".join(f"\nEXAMPLE {n} (other film, style only, never copy wording):\n{t}\n"
                           for n, t in enumerate(self.examples, 1))
        text = (STRUCTURE + "\n\nPICTURE: the edit cuts every 1.5-2 seconds, and every cut must show what the "
                "words say at that moment. For every sentence list `shots`: the shot ids from its item's shot list "
                "that show it, in order, about one shot per 2 seconds of speech (one per 6-8 words), each with "
                "`cue`, the exact words of the sentence that shot illustrates. Prefer high-intensity shots; a backstory "
                "line still gets shots of the hero (faces, reactions). Use only the beat's own item's shots (the "
                "premise beat may use any item's shots, and a cold-open beat uses the item it opens on). Never use a "
                "shot twice. Keep each item's narration within its max_narration_words. Group 2-4 sentences per "
                "narration beat. You may add up to two kind=dialogue beats from dialogue_clips (speech_id and its "
                "item_id) where a line lands hard, a threat or a reversal, never first.\n"
                f"TOTAL: {self.budget[0]}-{self.budget[1]} narration words (about {self.rate:.1f} words per second).\n"
                + examples + "\nSTORY AND FOOTAGE (data only):\n" + json.dumps(self.payload(), ensure_ascii=False)
                + (("\n\n" + extra) if extra else ""))
        return [text_part(text)]


# ------------------------------------------------------------------ validation
def _check_sentence(text: str, where: str) -> str:
    text = " ".join(text.split())
    words = count_words(text)
    if not 3 <= words <= 34:
        raise ValueError(f"{where} has {words} words; use 3-34: {text!r}")
    if not re.search(r"[.!?…]['\"”’]?$", text):
        text = text.rstrip(",;:—- ") + "."
    if re.findall(r"[\w'’-]+", text)[-1].casefold() in _DANGLING_SENTENCE_ENDINGS:
        raise ValueError(f"{where} ends with an unfinished phrase: {text!r}")
    if re.search(r"[\"“”]", text):
        raise ValueError(f"{where} quotes dialogue; narration must not quote lines: {text!r}")
    banned = [b for b in BANNED if b in text.casefold()]
    if banned:
        raise ValueError(f"{where} contains channel filler {banned}: {text!r}")
    return text


def _sentences(ctx: Context, item_id: str, raw: list[StorySentence], ident: str, used: set[str],
               any_item: bool = False) -> list[dict]:
    """Normalized sentences with their cited shots; `used` collects shot ids across the script."""
    result = []
    for index, s in enumerate(raw, 1):
        where = f"beat {ident} sentence {index}"
        text = _check_sentence(s.text, where)
        cues, problems = [], []
        for cue in s.shots:
            shot = ctx.shots.get(cue.shot_id)
            if shot is None or shot["filler"]:
                problems.append(f"{cue.shot_id} is not a showable shot")
            elif shot["item_id"] != item_id and not any_item:
                problems.append(f"{cue.shot_id} belongs to {shot['item_id']}, not this beat's item {item_id}")
            elif cue.shot_id in used:
                problems.append(f"{cue.shot_id} is already used")
            else:
                used.add(cue.shot_id)
                words = cue.cue.strip()
                cues.append({"shot_id": cue.shot_id,
                             "cue": words if words and words.casefold() in text.casefold() else ""})
        if not cues:
            best = [x["id"] for x in sorted(ctx.shot_list(item_id), key=lambda x: -x["intensity"])
                    if x["id"] not in used][:6]
            raise ValueError(f"{where} cites no usable shot ({'; '.join(problems) or 'none given'}); cite the shots "
                             f"that show it, e.g. from {best}")
        result.append({"sentence_id": f"{ident}.{index}", "text": text, "shots": cues, "item_id": item_id,
                       "event_ids": [], "line_kind": "action"})
    return result


def validate_script(ctx: Context, script: StorylineScript, *, budget: bool = True, story: bool = True) -> list[dict]:
    """Storyboard-ready beats, or ValueError with a concrete fix for the writer."""
    known = {i["item_id"] for i in ctx.items}
    beats, used, shots_used = [], set(), set()
    narration = dialogue = 0
    for raw in script.beats:
        if raw.item_id not in known:
            raise ValueError(f"beat uses item_id {raw.item_id!r}; use one of {sorted(known)} (items with footage)")
        if raw.kind == "dialogue":
            quote = ctx.speech_pool.get(raw.speech_id)
            if not quote or not clean_quote(quote) or raw.speech_id in used or dialogue >= MAX_DIALOGUE:
                continue  # An unknown, repeated or extra dialogue clip is dropped, not fatal.
            used.add(raw.speech_id)
            dialogue += 1
            beats.append({"id": f"D{dialogue:02d}", "kind": "dialogue", "speech_id": quote["id"], "item_id": raw.item_id,
                          "text": quote["text"], "delivery": "normal",
                          "shots": [{"start": quote["start"], "end": quote["end"]}],
                          "words": [dict(w) for w in quote["words"]], "confidence": quote["confidence"],
                          "timing_provenance": quote["timing_provenance"]})
            continue
        if not raw.sentences:
            raise ValueError(f"a narration beat for {raw.item_id} has no sentences")
        narration += 1
        ident = f"N{narration:02d}"
        sentences = _sentences(ctx, raw.item_id, raw.sentences, ident, shots_used, any_item=narration <= 2)
        beats.append({"id": ident, "kind": "narration", "item_id": raw.item_id, "delivery": "normal",
                      "source_gain": NARRATION_SOURCE_GAIN, "sentences": sentences,
                      "text": " ".join(s["text"] for s in sentences)})
    if story:
        check_story(ctx, beats, budget=budget)
        return beats
    first = next((i for i, b in enumerate(beats) if b["kind"] == "narration"), None)
    if first is None:
        raise ValueError("the script has no narration")
    # A leading dialogue clip moves just after the opening narration.
    return [beats[first]] + beats[:first] + beats[first + 1:]


def check_story(ctx: Context, beats: list[dict], *, budget: bool = True) -> None:
    narr = [b for b in beats if b["kind"] == "narration"]
    if not narr:
        raise ValueError("the script has no narration")
    if beats[0]["kind"] != "narration":
        raise ValueError("a dialogue clip cannot open the Short: open with the cold-open narration")
    lines = [s["text"] for b in narr for s in b["sentences"]]
    if play_by_play(lines):
        raise ValueError("three or more short sentences in a row describe the picture one action at a time; "
                         "rewrite them as story with cause and effect (so, but, until, when)")
    short = [t for t in lines if count_words(t) < 6]
    if len(short) > 2:
        raise ValueError(f"too many fragment sentences {short[:4]}; keep at most two short punch lines")
    opening = " ".join(b["text"] for b in narr[:2])
    if ctx.names and not any(re.search(rf"\b{re.escape(n)}\b", opening) for n in ctx.names):
        raise ValueError("the first two beats never name the hero; the premise beat must say who he is "
                         f"(names from the premise/notes: {sorted(ctx.names)[:6]}), what he did and why he is hunted")
    words = [w.strip(".,!?;:'\"").lower() for w in opening.split()]
    if sum(w in CONNECTIVES for w in words) < 2:
        raise ValueError("the cold open and premise do not explain anything; say why it happens and what is at "
                         "stake (so, but, when, until, because)")
    found = slop(lines)
    if found:
        raise ValueError(f"the narration sounds generated: {found}; replace them with concrete, specific words "
                         "about who does what and why")
    openers = [" ".join(re.findall(r"[\w'’-]+", t.casefold())[:2]) for t in lines]
    repeated = sorted({o for o in openers if o and openers.count(o) > 3})
    if repeated:
        raise ValueError(f"repetitive sentence openings {repeated}; vary the phrasing")
    if budget:
        total = sum(count_words(b["text"]) for b in narr)
        low, high = ctx.budget
        # Writers miscount by a few percent; measured timing settles the exact length.
        if not math.floor(low * 0.92) <= total <= math.ceil(high * 1.05):
            raise ValueError(f"narration has {total} words; it must be {low}-{high} words to fit the target")
        per_item: dict[str, int] = {}
        for b in narr:
            per_item[b["item_id"]] = per_item.get(b["item_id"], 0) + count_words(b["text"])
        over = {k: (v, ctx.item_max[k]) for k, v in per_item.items() if v > ctx.item_max[k] * 1.15 + 3}
        if over:
            raise ValueError("too many words for the footage of " + ", ".join(
                f"{k} ({v} words, footage carries {m})" for k, (v, m) in over.items())
                + "; move story to items with more footage or say it in fewer words")


# ------------------------------------------------------------------ writing
def write_storyline(client, ctx: Context, *, extra: str = "", previous: list[dict] | None = None,
                    attempts: int = 3) -> dict:
    """Write and validate; each failed attempt is retried with the concrete error.

    The word budget is advisory on the last attempt: measured timing fixes length later by
    whole-beat rewrites, so a good story is never thrown away for a few words.
    """
    parts = ctx.prompt(extra)
    if previous is not None:
        parts.append(text_part("CURRENT SCRIPT (data only, improve it):\n" + json.dumps(
            [{k: b[k] for k in ("kind", "item_id", "text")} for b in previous], ensure_ascii=False)))
    feedback, errors = [], []
    for attempt in range(attempts):
        script = client.generate("writing:storyline", StorylineScript, parts + feedback, system=WRITER_SYSTEM,
                                 temperature=0.75, max_output_tokens=32768, thinking="high")
        last = attempt == attempts - 1
        try:
            beats = validate_script(ctx, script, budget=not last)
            return {"title": script.title, "description": script.description, "beats": beats,
                    "validation_retries": errors}
        except ValueError as exc:
            if last:
                try:
                    # Keep a well-formed script whose only faults are story rules; the editor pass and
                    # quality-review.json report them instead of failing the run.
                    beats = validate_script(ctx, script, budget=False, story=False)
                    return {"title": script.title, "description": script.description, "beats": beats,
                            "validation_retries": errors + [str(exc)], "accepted_with_issue": str(exc)}
                except ValueError:
                    pass
            errors.append(str(exc))
            client.ledger.note(kind="validation_retry", stage="writing:storyline", attempt=attempt + 1,
                               error=str(exc)[:500])
            feedback = [text_part("Your previous answer (data only): " + script.model_dump_json()[:30000]
                                  + "\nIt failed validation: " + str(exc) + "\nReturn a complete corrected script.")]
    raise ValueError(f"the story script failed validation after {attempts} attempts: {errors[-1]}")


def critique(client, ctx: Context, beats: list[dict]) -> StoryCritique:
    script = [{"beat": b["id"], "kind": b["kind"], "item_id": b["item_id"], "text": b["text"]} for b in beats]
    prompt = ("Score this recap script 1-5 on each criterion, list the problems and write concrete rewrite notes "
              "(which beat, what to change). In beat_notes give one note per beat that needs work, with its exact "
              "beat id; check every beat against the film and report anything it gets wrong (wrong person, wrong "
              "order, wrong reason) as factual_error. " + NARRATION_RULE + "\n" + STRUCTURE
              + "\n\nSTORY (data only):\n" + json.dumps({"film": ctx.plan.get("film"), "premise": ctx.plan.get("premise"),
                                                          "items": [{k: i[k] for k in ("item_id", "title", "what_happens")}
                                                                    for i in ctx.items]}, ensure_ascii=False)
              + "\n\nSCRIPT (data only):\n" + json.dumps(script, ensure_ascii=False))
    return client.generate("planning:story_critique", StoryCritique, [text_part(prompt)], system=EDITOR_SYSTEM,
                           temperature=0.2, max_output_tokens=8192, thinking="medium")


SCORE_KEYS = ("hook", "premise", "causality", "storytelling", "ending", "clarity", "impact")


def _rank(c: StoryCritique) -> tuple[int, int]:
    """Weakest score first, then the total: a draft is only as strong as its weakest part."""
    scores = list(_scores(c).values())
    return min(scores), sum(scores)


def _editor_notes(c: StoryCritique) -> str:
    return ("EDITOR NOTES ON THE CURRENT SCRIPT (fix all of them; keep what works):\n- "
            + "\n- ".join(c.problems + [f"{n.beat_id}: {n.note}" for n in c.beat_notes]) + "\n" + c.rewrite_notes)


def _scores(c: StoryCritique) -> dict:
    return {k: getattr(c, k) for k in SCORE_KEYS}


def beat_role(beats: list[dict], beat_id: str) -> str:
    """What this beat must keep doing, whatever a rewrite fixes (run 25 lost its cold open and its ending)."""
    narration = [b["id"] for b in beats if b["kind"] == "narration"]
    if beat_id == narration[0]:
        return ("ROLE: COLD OPEN. Stay inside the on-screen action and its stakes; no backstory, no setup "
                "(the premise comes in the next beat).")
    if len(narration) > 1 and beat_id == narration[1]:
        return "ROLE: PREMISE. Rewind and say who the hero is, what he did and why he is hunted."
    if beat_id == narration[-1]:
        return ("ROLE: ENDING. Keep the payoff, twist or betrayal this beat ends on; the footage may show only "
                "part of it, which is not a contradiction.")
    return "ROLE: CHAIN. Tell this item's moment as a consequence of the beat before it."


def rewrite_beat(client, ctx: Context, beats: list[dict], beat_id: str, reason: str, *,
                 min_words: int | None = None, max_words: int | None = None,
                 only: list[int] | None = None) -> list[dict]:
    """Rewrite one narration beat; the story around it stays fixed. With ``only`` (0-based sentence indexes),
    just those sentences change and every other sentence must come back word for word."""
    index = next(i for i, b in enumerate(beats) if b["id"] == beat_id)
    beat = beats[index]
    keep = {i: x["text"] for i, x in enumerate(beat.get("sentences", [])) if only is not None and i not in only}
    limits = " and ".join(x for x in (f"at least {min_words} words" if min_words else "",
                                      f"at most {max_words} words" if max_words else "") if x)
    item = next(i for i in ctx.items if i["item_id"] == beat["item_id"])
    narration_ids = [b["id"] for b in beats if b["kind"] == "narration"]
    any_item = beat_id in narration_ids[:2]
    taken = {c["shot_id"] for b in beats if b["kind"] == "narration" and b["id"] != beat_id
             for x in b["sentences"] for c in x.get("shots", [])}
    available = [x for i in (ctx.items if any_item else [item]) for x in ctx.shot_list(i["item_id"])
                 if x["id"] not in taken]
    scope = ("" if only is None else
             f"Rewrite ONLY sentence(s) {', '.join(f'{beat_id}.{i + 1}' for i in only)}; return all "
             f"{len(beat['sentences'])} sentences in order, every other sentence copied word for word. ")
    prompt = (f"Rewrite narration beat {beat_id} only. Problem to fix: {reason}. " + scope + beat_role(beats, beat_id)
              + " " + (f"Keep it to {limits} in total. " if limits else "")
              + "Keep its place in the story and its link to the beats around it. " + NARRATION_RULE
              + "No stock phrases or piled-up adjectives: concrete words about who does what and why. For each "
              "sentence list `shots` (ids from available_shots that show it, about one per 2 seconds of speech, "
              "each with `cue`, the words it illustrates).\n" + json.dumps({
                  "beat": beat["text"], "sentences": {f"{beat_id}.{i + 1}": x["text"]
                                                     for i, x in enumerate(beat.get("sentences", []))},
                  "item": {"item_id": item["item_id"], "title": item["title"],
                                                 "what_happens": item["what_happens"],
                                                 "why_it_matters": item.get("story_context", "")},
                  "available_shots": available,
                  "previous_beat": beats[index - 1]["text"] if index else "",
                  "next_beat": beats[index + 1]["text"] if index + 1 < len(beats) else "",
                  "premise": ctx.plan.get("premise", "")}, ensure_ascii=False))

    def validate(result: BeatText):
        sentences = _sentences(ctx, beat["item_id"], result.sentences, beat_id, set(taken), any_item=any_item)
        if keep:
            changed = [f"{beat_id}.{i + 1}" for i, text in keep.items()
                       if i >= len(sentences) or sentences[i]["text"].strip() != text.strip()]
            if len(sentences) != len(beat["sentences"]) or changed:
                raise ValueError(f"only the flagged sentences may change; keep {', '.join(changed) or 'the count'} "
                                 "exactly as they were")
        words = sum(count_words(s["text"]) for s in sentences)
        if max_words and words > max_words:
            raise ValueError(f"rewrite has {words} words; maximum is {max_words}")
        if min_words and words < min_words:
            raise ValueError(f"rewrite has {words} words; minimum is {min_words}")
        if play_by_play([s["text"] for s in sentences]):
            raise ValueError("the rewrite narrates the picture one action at a time; tell the story instead")
        new = {**{k: v for k, v in beat.items() if k not in ("audio_path", "duration", "words", "timing_provenance",
                                                              "shots")},
               "sentences": sentences, "text": " ".join(s["text"] for s in sentences)}
        candidate = beats[:index] + [new] + beats[index + 1:]
        check_story(ctx, candidate, budget=False)  # A rewrite may not break the opening or the flow.
        return candidate

    opening = [b["id"] for b in beats if b["kind"] == "narration"][:2]
    if beat_id in opening:
        prompt += ("\nThis beat is part of the opening: together with the other opening beat it must name the hero "
                   "and say what he did and why he is in danger.")
    candidate, _ = with_repair(client, "writing:storyline_rewrite", BeatText, [text_part(prompt)], WRITER_SYSTEM,
                               validate, attempts=3, temperature=0.6, max_output_tokens=8192, thinking="medium")
    return candidate


# ------------------------------------------------------------------ placement
def place_storyline(ctx: Context, beats: list[dict]) -> tuple[list[dict], list[dict]]:
    """Cut every sentence to its cited shots, each cut at most MAX_CUT seconds; never drop a sentence.

    Needs measured word timings. Returns (beats with shots, evidence map).
    """
    from ..render import sentence_spans

    order = [i["item_id"] for i in ctx.items]
    footage = Footage(ctx.shots, order)
    for beat in beats:
        if beat["kind"] == "dialogue":
            shot = beat["shots"][0]
            footage.reserve(float(shot["start"]), float(shot["end"]))
    # Cited shots are kept for their own sentences: fill shots come from the rest.
    pending = {c["shot_id"] for b in beats if b["kind"] == "narration" for x in b["sentences"] for c in x["shots"]}
    evidence, placed_beats = [], []
    for beat in beats:
        beat = copy.deepcopy(beat)
        if beat["kind"] != "narration":
            placed_beats.append(beat)
            continue
        spans = sentence_spans(beat)
        shots, offset = [], 0
        for index, (sentence, span) in enumerate(zip(beat["sentences"], spans)):
            cited = [c["shot_id"] for c in sentence["shots"]]
            pending -= set(cited)
            cuts = cue_times(sentence, beat["words"], span, offset)
            offset += sum(1 for t in sentence["text"].split() if _normal(t))
            saved = {sid: footage.free[sid] for sid in pending}
            for sid in pending:  # Hide later sentences' cited shots from the fill search.
                footage.free[sid] = []
            placed, notes = place_sentence(footage, beat["item_id"], cited, cuts, span[1] - span[0],
                                           index, sentence["sentence_id"])
            footage.free.update(saved)
            shots += placed
            evidence.append({"sentence_id": sentence["sentence_id"], "beat_id": beat["id"], "item_id": beat["item_id"],
                             "text": sentence["text"], "estimated_seconds": round(span[1] - span[0], 2),
                             "footage_seconds": round(sum(x["end"] - x["start"] for x in placed), 2),
                             "cited_shots": cited, "longest_cut": round(max(x["end"] - x["start"] for x in placed), 2),
                             "notes": notes, "shots": [{k: x[k] for k in ("shot_id", "start", "end", "shows", "reason")}
                                                       for x in placed]})
        beat["shots"] = shots
        placed_beats.append(beat)
    return placed_beats, evidence


def _normal(token: str) -> str:
    from ..voice import _normal as normal
    return normal(token)


# ------------------------------------------------------------------ visual check
def visual_check(client, file, ctx: Context, beats: list[dict]) -> list[dict]:
    windows, parts = [], []
    for beat in beats:
        if beat["kind"] != "narration":
            continue
        for s in beat["sentences"]:
            shots = [sh for sh in beat["shots"] if sh["sentence_id"] == s["sentence_id"]]
            if shots:
                windows.append({"sentence_id": s["sentence_id"], "text": s["text"],
                                "footage": [[sh["start"], sh["end"]] for sh in shots]})
    if not windows:
        return []
    spans = sorted({(sh["start"], sh["end"]) for b in beats if b["kind"] == "narration" for sh in b["shots"]})
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    parts = [video_part(file, a, b, 0.5) for a, b in merged]
    result = client.generate(
        "review:story_visual", VisualCheck,
        parts + [text_part("The clips above cover these pool ranges in order: " + json.dumps(merged)
                           + ". The hero and characters: " + ctx.plan.get("premise", "")
                           + "\nFor each sentence, compare it with its footage window and report clear "
                           "contradictions only.\n" + json.dumps(windows, ensure_ascii=False))],
        system=CHECK_SYSTEM, temperature=0.1, max_output_tokens=8192, resolution="low", thinking="medium")
    known = {w["sentence_id"] for w in windows}
    return [c.model_dump() for c in result.contradictions if c.sentence_id in known]


# ------------------------------------------------------------------ orchestration
def run_storyline_pipeline(request: dict, source: Path, source_duration: float, workdir: Path, outdir: Path, *,
                           scenes: list[dict], speech: dict | None, transport=None, synthesize=None,
                           log=print, sleep=None):
    from .pipeline import Run, _key
    from .sync import duck_source_dialogue, timing_report

    run = Run(request, source, source_duration, workdir, outdir, scenes=scenes, speech=speech,
              transport=transport, synthesize=synthesize, log=log, sleep=sleep)
    plan, pool_map = load_discovery(Path(request["discovery_dir"]))
    log("Gemini: checking API key and model availability")
    # The script is the product: never let a Flash-Lite fallback write or judge it (run 22's Lite script was
    # filler). When the stronger models' free quota is spent the run stops, cached, and resumes after the reset.
    for stage in ("writing", "planning"):
        strong = [m for m in run.settings.model_preferences[stage] if "lite" not in m]
        if strong:
            run.settings.model_preferences[stage] = strong
    models = run.client.resolve_models()
    _write_json(outdir / "gemini-settings.json", {**run.settings.to_json(), "path": "storyline"})
    source_key = _key(str(source), source_duration, [(s["id"], s["start"], s["end"]) for s in scenes])
    # 1. Shot catalog: every detected shot of the pool, labelled.
    shots = catalog_shots(scenes, item_ranges(pool_map))
    items = {i["item_id"]: i for i in plan["items"]}

    def label():
        return label_shots(run.client, run.ensure_file(), shots, items, log=log)

    catalog = run.checkpoint("A-shot-catalog", [STORYLINE_VERSION, source_key, shots, models["analysis"]], label,
                             "shot-catalog.json")
    run.inventory = {"events": [], "coverage": {"shots": len(catalog),
                                                "filler": sum(1 for x in catalog.values() if x["filler"])}}
    ctx = Context(plan, pool_map, catalog, run.speech_pool, request, run.rate, run.target)
    run.editorial = {"hook_event_id": None, "selected": []}
    # 2-3. Story script and story editor
    inputs = [STORYLINE_VERSION, SCRIPT_VERSION, plan, pool_map, ctx.budget, models["writing"], models["planning"],
              request.get("brief"), request.get("user_story_notes"), run.rate]

    def compose():
        log("Gemini: writing the story script")
        draft = write_storyline(run.client, ctx)
        record = {"draft": draft, "critiques": []}
        beats, title, description = draft["beats"], draft["title"], draft["description"]
        try:
            review = critique(run.client, ctx, beats)
        except (ValueError, GeminiOutputError) as exc:
            record["critique_error"] = str(exc)[:300]
            record.update(beats=beats, title=title, description=description)
            return record
        record["critiques"].append({**_scores(review), "problems": review.problems, "notes": review.rewrite_notes})
        best = (beats, review, title, description)
        # Up to MAX_DRAFTS full drafts; each rewrite gets the editor's notes on the best draft so far.
        for attempt in range(2, MAX_DRAFTS + 1):
            if _rank(best[1])[0] >= QUALITY_BAR:
                break
            log(f"Gemini: story editor scores {_scores(best[1])}; full rewrite {attempt} of {MAX_DRAFTS}")
            try:
                better = write_storyline(run.client, ctx, previous=best[0], extra=(
                    f"DRAFT {attempt} OF {MAX_DRAFTS}: the best draft so far scored {_scores(best[1])}; every "
                    f"score must reach {QUALITY_BAR}.\n" + _editor_notes(best[1])))
                review = critique(run.client, ctx, better["beats"])
            except (ValueError, GeminiOutputError) as exc:
                record.setdefault("rewrite_errors", []).append(str(exc)[:300])
                continue
            record["critiques"].append({**_scores(review), "problems": review.problems,
                                        "notes": review.rewrite_notes, "after_rewrite": attempt})
            if _rank(review) > _rank(best[1]):
                best = (better["beats"], review, better["title"], better["description"])
        beats, review, title, description = best
        # The editor's beat notes become targeted fixes; the fixed script is kept only if it scores no worse.
        fixed_beats, fixed = _apply_beat_notes(run, ctx, beats, review)
        if fixed:
            try:
                final = critique(run.client, ctx, fixed_beats)
                record["critiques"].append({**_scores(final), "problems": final.problems,
                                            "notes": final.rewrite_notes, "after_beat_fixes": fixed})
                if _rank(final) >= _rank(review):
                    beats, review = fixed_beats, final
                else:
                    record["beat_fixes_discarded"] = fixed
            except (ValueError, GeminiOutputError) as exc:
                record["critique_error"] = str(exc)[:300]
        if _rank(review)[0] < QUALITY_BAR:
            # Never render a weak story: stop before the voice and render, with the editor's reasons. The
            # footage and shot labels stay cached, so a rerun only writes new drafts.
            raise ValueError(f"The script is below the quality bar (every editor score must be at least "
                             f"{QUALITY_BAR}) after {MAX_DRAFTS} drafts: best scores {_scores(review)}; problems: "
                             + "; ".join(review.problems)[:600])
        record.update(beats=beats, title=title, description=description, final_scores=_scores(review))
        return record

    script = run.checkpoint("S-storyline", inputs, compose, "script-draft.json")
    _write_json(outdir / "story-critique.json", script.get("critiques", []))
    beats = copy.deepcopy(script["beats"])
    # 4. Measured narration: fix length by rewriting whole beats
    log("Gemini: narration synthesis and timing")
    beats = _fit_length(run, ctx, beats)
    # 5. Placement on continuous story footage
    beats, evidence = place_storyline(ctx, beats)
    # 6. Visual contradictions: rewrite the affected beat once
    issues = []
    try:
        found = visual_check(run.client, run.ensure_file(), ctx, beats)
    except (ValueError, GeminiOutputError) as exc:
        found = []
        run.record(stage="V", action="visual_check", outcome="failed", error=str(exc)[:300])
    for beat_id in sorted({c["sentence_id"].split(".")[0] for c in found}):
        mine = [c for c in found if c["sentence_id"].startswith(beat_id + ".")]
        beat = next(b for b in beats if b["id"] == beat_id)
        words = count_words(beat["text"])
        flagged = sorted({int(c["sentence_id"].split(".")[1]) - 1 for c in mine
                          if c["sentence_id"].split(".")[-1].isdigit()} & set(range(len(beat["sentences"]))))
        try:
            beats = rewrite_beat(run.client, ctx, beats, beat_id, "these lines contradict the footage they play over: "
                                 + "; ".join(f"{c['problem']} (on screen: {c['on_screen']})" for c in mine),
                                 min_words=max(4, math.floor(words * 0.85)), max_words=words + 3,
                                 only=flagged or None)
            run.record(stage="V", action="rewrite_contradiction", beat_id=beat_id, outcome="rewritten",
                       contradictions=mine)
        except (ValueError, GeminiOutputError) as exc:
            issues += [{"check": "visual_contradiction", "severity": "major", "beat_id": beat_id,
                        "sentence_id": c["sentence_id"], "description": c["problem"], "origin": "gemini_review"}
                       for c in mine]
            run.record(stage="V", action="rewrite_contradiction", beat_id=beat_id, outcome="failed",
                       error=str(exc)[:300])
    if found and any(b["kind"] == "narration" and not b.get("words") for b in beats):
        beats = _fit_length(run, ctx, beats)
        beats, evidence = place_storyline(ctx, beats)
    # 7. The bar again, on the words that will actually be heard: fixes after the gate are scored too.
    final_scores = script.get("final_scores")
    gate_text = {b["id"]: b["text"] for b in script["beats"] if b["kind"] == "narration"}
    changed = [b["id"] for b in beats if b["kind"] == "narration" and b["text"] != gate_text.get(b["id"])]
    if changed and final_scores:
        final_scores = _final_bar(run, ctx, beats)
        if final_scores is None or min(final_scores.values()) < QUALITY_BAR:
            # Restore the gate-passed wording; a contradiction it had is listed instead of fixed.
            log(f"Gemini: rewrites after the quality bar scored {final_scores}; restoring the approved wording "
                f"of {', '.join(changed)}")
            approved = {b["id"]: b for b in script["beats"]}
            beats = [copy.deepcopy(approved[b["id"]]) if b["id"] in changed else b for b in beats]
            issues += [{"check": "visual_contradiction", "severity": "minor", "beat_id": c["sentence_id"].split(".")[0],
                        "sentence_id": c["sentence_id"], "description": c["problem"], "origin": "gemini_review"}
                       for c in found if c["sentence_id"].split(".")[0] in changed]
            run.record(stage="Q", action="restore_approved_wording", beats=changed, scores=final_scores)
            beats = _fit_length(run, ctx, beats)
            beats, evidence = place_storyline(ctx, beats)
            now = [b["id"] for b in beats if b["kind"] == "narration" and b["text"] != gate_text.get(b["id"])]
            if now:
                final_scores = _final_bar(run, ctx, beats)
                if final_scores is not None and min(final_scores.values()) < QUALITY_BAR:
                    raise ValueError(f"The final script fell below the quality bar after length fixes: "
                                     f"{final_scores}. Nothing was rendered; a rerun writes new drafts.")
            else:
                final_scores = script.get("final_scores")
    # Quality record
    lines = [s["text"] for b in beats if b["kind"] == "narration" for s in b["sentences"]]
    critiques = script.get("critiques", [])
    if critiques and min(v for k, v in critiques[-1].items() if k in SCORE_KEYS) <= 2:
        issues.append({"check": "story_quality", "severity": "major", "origin": "gemini_review",
                       "description": f"story editor scores {critiques[-1]}"})
    if script.get("draft", {}).get("accepted_with_issue"):
        issues.append({"check": "story_rules", "severity": "major", "origin": "deterministic",
                       "description": script["draft"]["accepted_with_issue"]})
    if play_by_play(lines):
        issues.append({"check": "pacing", "severity": "major", "description": "frame-by-frame narration remains",
                       "origin": "deterministic"})
    cuts = [x["end"] - x["start"] for b in beats if b["kind"] == "narration" for x in b["shots"]]
    long_cuts = sum(1 for c in cuts if c > MAX_CUT + 0.05)
    if long_cuts:
        issues.append({"check": "cut_length", "severity": "minor", "origin": "deterministic",
                       "description": f"{long_cuts} cut(s) longer than {MAX_CUT + 0.05:.2f}s"})
    borrowed = [e["sentence_id"] for e in evidence if any(x["reason"] == "neighbouring story moment"
                                                          for x in e["shots"])]
    total = sum(b["duration"] for b in beats)
    if total < run.target * 0.8:
        issues.append({"check": "duration", "severity": "major", "origin": "deterministic",
                       "description": f"the Short is {total:.1f}s against a {run.target:.0f}s target"})
    review_status = "passed_with_warnings" if issues else "passed_automated_review"
    _write_json(outdir / "quality-review.json", {
        "review_status": review_status, "path": "storyline", "story_critique": critiques, "issues": issues,
        "final_scores": final_scores,
        "sentences_borrowing_neighbour_footage": borrowed,
        "cuts": {"count": len(cuts), "median_seconds": round(sorted(cuts)[len(cuts) // 2], 2) if cuts else 0,
                 "longest_seconds": round(max(cuts), 2) if cuts else 0}, "duration_seconds": round(total, 2),
        "note": "Automated review does not replace human playback review."})
    _write_json(outdir / "evidence-map.json", {"sentences": evidence, "policy": (
        "Every narration sentence cuts to the shots it cites, each on the words it illustrates, at most "
        f"{MAX_CUT}s per cut, gaps filled with the same moment's next best shots; filler only when nothing else is "
        "left; no sentence is removed for lack of footage.")})
    _write_json(outdir / "correction-history.json", run.history)
    ducking = duck_source_dialogue(beats, run.segments)
    _write_json(outdir / "narration-timing.json", {**timing_report(beats), "source_ducking": ducking})
    result = {"title": request["title"], "description": script.get("description") or request.get("brief", ""),
              "beats": beats, "planning_mode": "gemini-multistage",
              "planning": {"version": STORYLINE_VERSION, "path": "storyline", "preset": run.settings.preset,
                           "models": models, "review_status": review_status,
                           "planning_words_per_second": round(run.rate, 3), "planning_rate_source": run.rate_source,
                           "coverage": run.inventory.get("coverage", {})}}
    run.ledger.save()
    return result, run


def _final_bar(run, ctx: Context, beats: list[dict]) -> dict | None:
    """Scores of the final words, or None when the editor could not be reached (then the approved wording wins)."""
    try:
        review = critique(run.client, ctx, beats)
    except (ValueError, GeminiOutputError) as exc:
        run.record(stage="Q", action="final_critique", outcome="failed", error=str(exc)[:300])
        return None
    scores = _scores(review)
    run.record(stage="Q", action="final_critique", scores=scores, problems=review.problems[:4])
    return scores


def _apply_beat_notes(run, ctx: Context, beats: list[dict], review: StoryCritique) -> tuple[list[dict], list[str]]:
    known = {b["id"] for b in beats if b["kind"] == "narration"}
    notes = sorted((n for n in review.beat_notes if n.beat_id in known), key=lambda n: not n.factual_error)
    fixed: list[str] = []
    for note in notes:
        if len(fixed) >= MAX_NOTE_REWRITES or note.beat_id in fixed:
            continue
        if not note.factual_error and min(_scores(review).values()) >= 4:
            break  # A strong script keeps its wording; only factual errors are still fixed.
        beat = next(b for b in beats if b["id"] == note.beat_id)
        words = count_words(beat["text"])
        try:
            beats = rewrite_beat(run.client, ctx, beats, note.beat_id,
                                 ("it gets the film wrong: " if note.factual_error else "story editor: ") + note.note,
                                 min_words=max(6, math.floor(words * 0.85)), max_words=math.ceil(words * 1.15) + 2)
            fixed.append(note.beat_id)
            run.record(stage="S", action="editor_beat_fix", beat_id=note.beat_id, factual=note.factual_error,
                       note=note.note[:300], outcome="rewritten")
        except (ValueError, GeminiOutputError) as exc:
            run.record(stage="S", action="editor_beat_fix", beat_id=note.beat_id, outcome="failed",
                       error=str(exc)[:300])
    return beats, fixed


SPEECH_FAILURE = re.compile(r"Narration beat (\S+) failed complete-speech verification")
MAX_SPEECH_REWRITES = 3
SPEECH_NOTE = ("the voice engine could not say this line so that speech recognition hears every word. Keep the same "
               "meaning and the same shots, in plainer spoken words: spell out numbers, avoid rare names, "
               "abbreviations, hyphenated or invented words and tongue-twisters, and keep every sentence complete")


def _synthesize_checked(run, ctx: Context, beats: list[dict]) -> list[dict]:
    """Synthesize; a beat whose speech fails verification is rewritten in plainer words, then (last resort)
    verified loosely, so one hard-to-hear line never ends the run. Run 22 died on 'Fierce attack dogs maul',
    heard as 'Fierce the tech dogs, more'."""
    rewrites = getattr(run, "speech_rewrites", 0)
    while True:
        try:
            return run.synthesize(beats)
        except RuntimeError as exc:
            match = SPEECH_FAILURE.search(str(exc))
            beat = next((b for b in beats if match and b["id"] == match.group(1) and b["kind"] == "narration"), None)
            if beat is None or beat.get("loose"):
                raise
            if rewrites < MAX_SPEECH_REWRITES and not beat.get("speech_rewritten"):
                rewrites += 1
                run.speech_rewrites = rewrites
                words = count_words(beat["text"])
                try:
                    beats = rewrite_beat(run.client, ctx, beats, beat["id"], SPEECH_NOTE,
                                         min_words=max(6, math.floor(words * 0.85)), max_words=words + 3)
                    beats = [{**b, "speech_rewritten": True} if b["id"] == beat["id"] else b for b in beats]
                    run.record(stage="G", action="speech_rewrite", beat_id=beat["id"], outcome="rewritten",
                               error=str(exc)[:300])
                    continue
                except (ValueError, GeminiOutputError) as rewrite_error:
                    run.record(stage="G", action="speech_rewrite", beat_id=beat["id"], outcome="failed",
                               error=str(rewrite_error)[:300])
            beats = [{**b, "loose": True} if b["id"] == beat["id"] else b for b in beats]
            run.record(stage="G", action="speech_loose_match", beat_id=beat["id"], outcome="retried",
                       error=str(exc)[:300])


def _fit_length(run, ctx: Context, beats: list[dict]) -> list[dict]:
    """Synthesize, then bring the measured length inside [0.85, 1.0] x target by whole-beat rewrites."""
    beats = _synthesize_checked(run, ctx, beats)
    for attempt in range(run.settings.max_timing_repairs + 1):
        total = sum(b["duration"] for b in beats)
        narr = [b for b in beats if b["kind"] == "narration"]
        rate = getattr(run, "measured_rate", run.rate)
        changed = False
        if total > run.target + 0.05:
            excess = total - run.target * 0.97
            for beat in sorted(narr, key=lambda b: -b["duration"]):
                if excess <= 0:
                    break
                words = count_words(beat["text"])
                cut = min(math.ceil(words * 0.35), math.ceil(excess * rate * 1.15))
                try:
                    beats = rewrite_beat(run.client, ctx, beats, beat["id"],
                                         f"the Short is {total:.1f}s but must fit {run.target:.0f}s; say the same story "
                                         "in fewer words", max_words=max(8, words - cut))
                    excess -= cut / max(rate, 0.5)
                    changed = True
                    run.record(stage="G", action="shorten_beat", beat_id=beat["id"], outcome="rewritten",
                               measured_total=round(total, 2))
                except (ValueError, GeminiOutputError) as exc:
                    run.record(stage="G", action="shorten_beat", beat_id=beat["id"], outcome="failed",
                               error=str(exc)[:300])
            if not changed and attempt == run.settings.max_timing_repairs:
                raise ValueError(f"Measured narration is {total:.1f}s; the maximum is {run.target:.0f}s and no "
                                 "beat could be shortened. See correction-history.json.")
        elif total < run.target * 0.85:
            room = run.target * 0.95 - total
            spoken: dict[str, float] = {}
            for b in narr:
                spoken[b["item_id"]] = spoken.get(b["item_id"], 0.0) + b["duration"]
            spare = {k: ctx.footage[k] * 0.92 - spoken.get(k, 0.0) for k in ctx.footage}
            for beat in sorted(narr, key=lambda b: -spare.get(b["item_id"], 0.0)):
                if room <= 1.5:
                    break
                extra_seconds = min(room, spare.get(beat["item_id"], 0.0))
                extra = math.floor(extra_seconds * rate * 0.9)
                if extra < 5:
                    continue
                words = count_words(beat["text"])
                try:
                    beats = rewrite_beat(run.client, ctx, beats, beat["id"],
                                         f"the Short is {total:.0f}s against a {run.target:.0f}s target; deepen this "
                                         "beat with story meaning: why it happens, what is at stake, what it leads to "
                                         "(no new events, no description of the picture)",
                                         min_words=words + max(3, extra // 2), max_words=words + extra)
                    spare[beat["item_id"]] -= extra_seconds
                    room -= extra_seconds
                    changed = True
                    run.record(stage="G", action="expand_beat", beat_id=beat["id"], outcome="rewritten",
                               measured_total=round(total, 2), added_up_to_words=extra)
                except (ValueError, GeminiOutputError) as exc:
                    run.record(stage="G", action="expand_beat", beat_id=beat["id"], outcome="failed",
                               error=str(exc)[:300])
        if not changed:
            break
        beats = _synthesize_checked(run, ctx, beats)
    total = sum(b["duration"] for b in beats)
    if total > run.target + 0.05:
        raise ValueError(f"Measured narration is {total:.1f}s; the maximum is {run.target:.0f}s after "
                         f"{run.settings.max_timing_repairs} shortening pass(es). See correction-history.json.")
    return beats
