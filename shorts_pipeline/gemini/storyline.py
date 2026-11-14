"""Story-first production for discovered footage (`source: discover`).

Discovery already knows the story: a premise, story items in film order (what happens,
why it matters) and, for each item, the exact stretches of the footage pool that show it.
This path keeps that structure instead of re-deriving it from anonymous footage:

1. Moment log: Stage A (analysis.py) timestamps what is visible in the pool.
2. Story script: one writer call narrates the items as a story (cold open, premise,
   cause-and-effect chain, ending), with style examples from the reference recaps and a
   per-item word budget that the item's own footage can carry.
3. Story editor: an independent critique scores the script; a weak script is rewritten once.
4. Measured narration: the voice is synthesized at the requested pace; an overlong or
   short Short is fixed by rewriting whole beats, never by deleting sentences.
5. Placement: each sentence plays over a continuous window of its own item's footage,
   in order, with the moment it narrates landing on its key word.
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
from .story import with_repair

STORYLINE_VERSION = 1
FILL = 0.95              # Narration fills this share of the target; dialogue and breaths take the rest.
ITEM_SHARE = 0.85        # An item's narration may use this share of the item's own footage.
WINDOW_HEADROOM = 1.08   # Each sentence gets a little more footage than it speaks; the renderer trims.
MIN_PIECE = 0.6          # Shorter leftovers are skipped: they would flash on screen.
NARRATION_SOURCE_GAIN = 0.07  # Film sound (hits, gunshots) stays audible under the voice.
MAX_DIALOGUE = 2
REFERENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "reference-data" / "transcripts"
REFERENCE_ORDER = ("V7", "V8", "V3", "V4", "V2", "V5", "V6", "V1")
GENERIC = frozenset("chapter part movie film the and with from into that this from".split())


# ------------------------------------------------------------------ schemas
class StorySentence(BaseModel):
    text: str = Field(description="One complete narration sentence")
    at: float | None = Field(None, description="Pool second where the moment this sentence narrates is visible; "
                                               "must be inside its item's footage; null for a backstory/motive line")
    key_word: str = Field("", description="One word of this sentence that should land on that moment")


class StoryBeat(BaseModel):
    kind: Literal["narration", "dialogue"] = "narration"
    item_id: str = Field(description="The story item whose footage plays under this beat")
    speech_id: str = Field("", description="For kind=dialogue only: an approved film dialogue clip id")
    sentences: list[StorySentence] = Field(default_factory=list)


class StorylineScript(BaseModel):
    title: str
    description: str = Field(description="One or two sentences for the video description, no hashtags")
    beats: list[StoryBeat]


class StoryCritique(BaseModel):
    hook: int = Field(ge=1, le=5, description="Does the first sentence make a viewer need to know more?")
    premise: int = Field(ge=1, le=5, description="Within the first two beats: who the hero is, what he did, why he is hunted")
    causality: int = Field(ge=1, le=5, description="Each beat follows from the last (so/but/because), no jumps")
    storytelling: int = Field(ge=1, le=5, description="Story meaning (intent, stakes, consequence), not a description of frames")
    ending: int = Field(ge=1, le=5, description="A payoff or renewed threat that lands")
    clarity: int = Field(ge=1, le=5, description="A newcomer can follow every name and event")
    problems: list[str] = Field(default_factory=list)
    rewrite_notes: str = Field("", description="Concrete instructions for the rewrite")


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
    "motives, backstory and lines about what happens off screen are NOT contradictions. Return JSON matching the schema.")

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
    "STYLE: flowing sentences of 12-28 words with an occasional short punch line; vivid verbs; mostly present "
    "tense; spell numbers as words; no quotation marks; never 'in this movie' or 'this film'.")


# ------------------------------------------------------------------ inputs
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

    def __init__(self, plan: dict, pool_map: list[dict], inventory: dict, speech_pool: dict, request: dict,
                 rate: float, target: float):
        self.plan, self.request, self.rate, self.target = plan, request, rate, target
        self.ranges = item_ranges(pool_map)
        self.items = [i for i in plan["items"] if i["item_id"] in self.ranges]
        self.footage = {k: sum(b - a for a, b in v) for k, v in self.ranges.items()}
        self.events = sorted(inventory.get("events", []), key=lambda e: e["start"])
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

    def moments(self, item_id: str) -> list[dict]:
        spans = self.ranges[item_id]
        found = [{"at": round(float(e.get("peak", (e["start"] + e["end"]) / 2)), 1),
                  "from": round(e["start"], 1), "to": round(e["end"], 1), "visible": e["visible_action"]}
                 for e in self.events if any(e["start"] < b and e["end"] > a for a, b in spans)]
        return found

    def dialogue_clips(self) -> list[dict]:
        clips = []
        for q in self.speech_pool.values():
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
                 "footage": [[round(a, 1), round(b, 1)] for a, b in self.ranges[i["item_id"]]],
                 "footage_seconds": round(self.footage[i["item_id"]], 1),
                 "max_narration_words": self.item_max[i["item_id"]],
                 "visible_moments": self.moments(i["item_id"])} for i in self.items],
            "dialogue_clips": self.dialogue_clips()}

    def prompt(self, extra: str = "") -> list[dict]:
        examples = "".join(f"\nEXAMPLE {n} (other film, style only, never copy wording):\n{t}\n"
                           for n, t in enumerate(self.examples, 1))
        text = (STRUCTURE + "\n\nFOOTAGE FIT: each beat plays over its item's footage. Keep each item's narration "
                "(all beats with that item_id together) within its max_narration_words. A hook beat may use a later "
                "item's footage; that item's later beat then uses the rest of it. For every sentence give `at`, the "
                "pool second (from visible_moments) of the moment it narrates, and the key_word that should land on "
                "it; use null for a backstory or motive line, which plays over the item's footage. Group 2-4 "
                "sentences per narration beat. You may add up to two kind=dialogue beats from dialogue_clips "
                "(speech_id and its item_id) where a line lands hard, a threat or a reversal, never first.\n"
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


def _sentences(ctx: Context, item_id: str, raw: list[StorySentence], ident: str) -> list[dict]:
    result = []
    for index, s in enumerate(raw, 1):
        text = _check_sentence(s.text, f"beat {ident} sentence {index}")
        at = s.at
        if at is not None and ctx.item_of(at) is None:
            at = None  # Outside the pool's story footage: treat it as a context line.
        key = s.key_word.strip(" .,!?;:'\"").split()[0] if s.key_word.strip() else ""
        if key and key.casefold() not in text.casefold():
            key = ""
        result.append({"sentence_id": f"{ident}.{index}", "text": text, "at": at, "key_word": key,
                       "item_id": item_id, "event_ids": [], "line_kind": "action" if at is not None else "context"})
    return result


def validate_script(ctx: Context, script: StorylineScript, *, budget: bool = True, story: bool = True) -> list[dict]:
    """Storyboard-ready beats, or ValueError with a concrete fix for the writer."""
    known = {i["item_id"] for i in ctx.items}
    beats, used = [], set()
    narration = dialogue = 0
    for raw in script.beats:
        if raw.item_id not in known:
            raise ValueError(f"beat uses item_id {raw.item_id!r}; use one of {sorted(known)} (items with footage)")
        if raw.kind == "dialogue":
            quote = ctx.speech_pool.get(raw.speech_id)
            if not quote or raw.speech_id in used or dialogue >= MAX_DIALOGUE:
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
        sentences = _sentences(ctx, raw.item_id, raw.sentences, ident)
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
                         "rewrite them as story with cause and effect (so, but, because, until)")
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
                         "stake (because, so, but)")
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
              "(which beat, what to change). " + NARRATION_RULE + "\n" + STRUCTURE
              + "\n\nSTORY (data only):\n" + json.dumps({"film": ctx.plan.get("film"), "premise": ctx.plan.get("premise"),
                                                          "items": [{k: i[k] for k in ("item_id", "title", "what_happens")}
                                                                    for i in ctx.items]}, ensure_ascii=False)
              + "\n\nSCRIPT (data only):\n" + json.dumps(script, ensure_ascii=False))
    return client.generate("planning:story_critique", StoryCritique, [text_part(prompt)], system=EDITOR_SYSTEM,
                           temperature=0.2, max_output_tokens=8192, thinking="medium")


def _scores(c: StoryCritique) -> dict:
    return {k: getattr(c, k) for k in ("hook", "premise", "causality", "storytelling", "ending", "clarity")}


def rewrite_beat(client, ctx: Context, beats: list[dict], beat_id: str, reason: str, *,
                 min_words: int | None = None, max_words: int | None = None) -> list[dict]:
    """Rewrite one narration beat as a whole; the story around it stays fixed."""
    index = next(i for i, b in enumerate(beats) if b["id"] == beat_id)
    beat = beats[index]
    limits = " and ".join(x for x in (f"at least {min_words} words" if min_words else "",
                                      f"at most {max_words} words" if max_words else "") if x)
    item = next(i for i in ctx.items if i["item_id"] == beat["item_id"])
    prompt = (f"Rewrite narration beat {beat_id} only. Problem to fix: {reason}. "
              + (f"Keep it to {limits} in total. " if limits else "")
              + "Keep its place in the story and its link to the beats around it. " + NARRATION_RULE
              + "Give `at` (a pool second inside this item's footage, or null for a context line) and key_word for "
              "each sentence.\n" + json.dumps({
                  "beat": beat["text"], "item": {"item_id": item["item_id"], "title": item["title"],
                                                 "why_it_matters": item.get("story_context", ""),
                                                 "footage": ctx.ranges[item["item_id"]],
                                                 "visible_moments": ctx.moments(item["item_id"])},
                  "previous_beat": beats[index - 1]["text"] if index else "",
                  "next_beat": beats[index + 1]["text"] if index + 1 < len(beats) else "",
                  "premise": ctx.plan.get("premise", "")}, ensure_ascii=False))

    def validate(result: BeatText):
        sentences = _sentences(ctx, beat["item_id"], result.sentences, beat_id)
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
                   "and say what he did and why he is in danger (because, so, but).")
    candidate, _ = with_repair(client, "writing:storyline_rewrite", BeatText, [text_part(prompt)], WRITER_SYSTEM,
                               validate, attempts=3, temperature=0.6, max_output_tokens=8192, thinking="medium")
    return candidate


# ------------------------------------------------------------------ placement
def _free(spans: list[tuple[float, float]], used: list[tuple[float, float]]) -> list[tuple[float, float]]:
    free = []
    for a, b in spans:
        pieces = [(a, b)]
        for ua, ub in sorted(used):
            nxt = []
            for pa, pb in pieces:
                if ub <= pa or ua >= pb:
                    nxt.append((pa, pb))
                    continue
                if ua > pa:
                    nxt.append((pa, ua))
                if ub < pb:
                    nxt.append((ub, pb))
            pieces = nxt
        free += [(pa, pb) for pa, pb in pieces if pb - pa >= MIN_PIECE]
    return sorted(free)


def _take(free: list[tuple[float, float]], start: float, need: float) -> list[tuple[float, float]]:
    """Consecutive pieces from `start` onward (then from the earliest free footage) totalling `need`."""
    ordered = [(max(a, start), b) for a, b in free if b - max(a, start) >= MIN_PIECE] + \
              [(a, min(b, start)) for a, b in free if min(b, start) - a >= MIN_PIECE]
    taken, total = [], 0.0
    for a, b in ordered:
        if total >= need:
            break
        length = min(b - a, need - total)
        if length < MIN_PIECE and taken:
            length = min(b - a, MIN_PIECE)
        taken.append((a, a + length))
        total += length
    return taken


def _event_at(ctx: Context, a: float, b: float) -> dict | None:
    best, overlap = None, 0.0
    for e in ctx.events:
        o = min(b, e["end"]) - max(a, e["start"])
        if o > overlap:
            best, overlap = e, o
    return best


def sentence_needs(beat: dict, rate: float) -> list[tuple[float, float]]:
    """(seconds spoken, seconds from sentence start to its key word) per sentence."""
    if beat.get("words"):
        from ..render import sentence_spans
        from .picture import key_word_times
        spans = sentence_spans(beat)
        keys = key_word_times(beat)
        return [(b - a, keys.get(i, 0.3 * (b - a))) for i, (a, b) in enumerate(spans)]
    out = []
    for s in beat["sentences"]:
        seconds = count_words(s["text"]) / max(rate, 0.5) + 0.15
        out.append((seconds, 0.3 * seconds))
    return out


def place_storyline(ctx: Context, beats: list[dict]) -> tuple[list[dict], list[dict]]:
    """Give every sentence continuous footage of its own item; never drop a sentence.

    Returns (beats with shots, evidence map). Raises only if the whole pool is exhausted.
    """
    used: list[tuple[float, float]] = []
    for beat in beats:
        if beat["kind"] == "dialogue":
            shot = beat["shots"][0]
            used.append((shot["start"], shot["end"]))
    order = [i["item_id"] for i in ctx.items]
    cursor: dict[str, float] = {}
    evidence = []
    placed_beats = []
    for beat in beats:
        beat = copy.deepcopy(beat)
        if beat["kind"] != "narration":
            placed_beats.append(beat)
            continue
        item = beat["item_id"]
        shots = []
        for index, (sentence, (need, key_offset)) in enumerate(zip(beat["sentences"], sentence_needs(beat, ctx.rate))):
            want = need * WINDOW_HEADROOM
            at = sentence.get("at")
            own = ctx.ranges.get(item, [])
            start = cursor.get(item, own[0][0] if own else 0.0)
            if at is not None and ctx.item_of(at) == item:
                start = max(start, at - key_offset - 0.15) if at - key_offset - 0.15 >= start else start
            pieces = _take(_free(own, used), start, want)
            borrowed = []
            total = sum(b - a for a, b in pieces)
            if total < want * 0.98:
                # This item's footage is spent: continue with the neighbouring items' unused footage.
                position = order.index(item) if item in order else 0
                for other in sorted((o for o in order if o != item), key=lambda o: abs(order.index(o) - position)):
                    extra = _take(_free(ctx.ranges[other], used + pieces), ctx.ranges[other][0][0], want - total)
                    if extra:
                        borrowed.append(other)
                        pieces += extra
                        total += sum(b - a for a, b in extra)
                    if total >= want * 0.98:
                        break
            if total < need * 0.86:
                raise ValueError(f"the footage pool is exhausted at {sentence['sentence_id']} ({total:.1f}s for "
                                 f"{need:.1f}s of narration); discovery needs more footage")
            used += pieces
            if pieces and not borrowed:
                cursor[item] = pieces[-1][1]
            anchor_piece = None
            if at is not None:
                anchor_piece = next((p for p in pieces if p[0] <= at <= p[1]), None)
            for n, (a, b) in enumerate(pieces):
                event = _event_at(ctx, a, b)
                own_piece = ctx.item_of((a + b) / 2) == item
                anchor = at if (a, b) == anchor_piece else (a + b) / 2
                shot = {"start": round(a, 3), "end": round(b, 3), "anchor": round(anchor, 3),
                        "focus_x": float((event or {}).get("focus_x", 0.5)), "sentence_index": index,
                        "sentence_id": sentence["sentence_id"], "shot_id": f"{sentence['sentence_id']}#{n + 1}",
                        "event_id": (event or {}).get("id", ""),
                        "shows": (event or {}).get("visible_action", ""),
                        "reason": "story item footage" if own_piece else "neighbouring story footage (item exhausted)",
                        "relevance": 1.0 if own_piece else 0.6, "shot_bounds": [round(a, 3), round(b, 3)]}
                if (a, b) == anchor_piece and sentence.get("key_word"):
                    shot.update(key_clip=True, key_time=round(key_offset, 3))
                shots.append(shot)
            evidence.append({"sentence_id": sentence["sentence_id"], "beat_id": beat["id"], "item_id": item,
                             "text": sentence["text"], "at": at, "estimated_seconds": round(need, 2),
                             "footage_seconds": round(total, 2), "borrowed_from": borrowed,
                             "shots": [{k: s[k] for k in ("start", "end", "anchor", "shows")} for s in shots
                                       if s["sentence_index"] == index]})
        beat["shots"] = shots
        placed_beats.append(beat)
    return placed_beats, evidence


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
    from .analysis import ANALYSIS_VERSION, analyze_footage
    from .pipeline import Run, _key
    from .sync import duck_source_dialogue, timing_report

    run = Run(request, source, source_duration, workdir, outdir, scenes=scenes, speech=speech,
              transport=transport, synthesize=synthesize, log=log, sleep=sleep)
    plan, pool_map = load_discovery(Path(request["discovery_dir"]))
    log("Gemini: checking API key and model availability")
    models = run.client.resolve_models()
    _write_json(outdir / "gemini-settings.json", {**run.settings.to_json(), "path": "storyline"})
    source_key = _key(str(source), source_duration, [(s["id"], s["start"], s["end"]) for s in scenes])
    # 1. Moment log
    try:
        run.inventory = run.checkpoint(
            "A-event-inventory", [ANALYSIS_VERSION, source_key, run.settings.analysis, models["analysis"],
                                  request.get("story_notes", ""), run.speech.get("cache_key")],
            lambda: analyze_footage(run.client, source, workdir, scenes, run.speech, request.get("story_notes", ""),
                                    source_duration, log), "event-inventory.json")
    except (ValueError, GeminiOutputError) as exc:
        # The story plan and the pool map still say what each stretch shows.
        log(f"Gemini: footage analysis failed ({str(exc)[:160]}); using the discovery descriptions")
        run.record(stage="A", action="analysis", outcome="failed", error=str(exc)[:300])
        run.inventory = {"events": [], "coverage": {}}
    if not run.inventory.get("events"):
        run.inventory["events"] = [{"id": f"P{n:03d}", "start": p["pool_start"], "end": p["pool_end"],
                                    "peak": (p["pool_start"] + p["pool_end"]) / 2, "visible_action": p["shows"],
                                    "focus_x": 0.5} for n, p in enumerate(pool_map, 1)]
    if run.inventory.get("source_flags", {}).get("voice_over_narration"):
        run.speech_pool = {}
    ctx = Context(plan, pool_map, run.inventory, run.speech_pool, request, run.rate, run.target)
    run.editorial = {"hook_event_id": None, "selected": []}
    # 2-3. Story script and story editor
    inputs = [STORYLINE_VERSION, plan, pool_map, ctx.budget, models["writing"], models["planning"],
              request.get("brief"), request.get("user_story_notes"), run.rate]

    def compose():
        log("Gemini: writing the story script")
        draft = write_storyline(run.client, ctx)
        record = {"draft": draft, "critiques": []}
        beats = draft["beats"]
        try:
            review = critique(run.client, ctx, beats)
            record["critiques"].append({**_scores(review), "problems": review.problems, "notes": review.rewrite_notes})
            if min(_scores(review).values()) <= 3:
                log(f"Gemini: story editor scores {_scores(review)}; rewriting once")
                better = write_storyline(run.client, ctx, previous=beats, extra=(
                    "EDITOR NOTES ON THE CURRENT SCRIPT (fix all of them; keep what works):\n- "
                    + "\n- ".join(review.problems) + "\n" + review.rewrite_notes))
                record["rewrite"] = {"title": better["title"], "validation_retries": better["validation_retries"]}
                beats = better["beats"]
                draft = {**draft, "title": better["title"], "description": better["description"]}
                after = critique(run.client, ctx, beats)  # Reported, not acted on: one rewrite at most.
                record["critiques"].append({**_scores(after), "problems": after.problems,
                                            "notes": after.rewrite_notes, "after_rewrite": True})
        except (ValueError, GeminiOutputError) as exc:
            record["critique_error"] = str(exc)[:300]
        record.update(beats=beats, title=draft["title"], description=draft["description"])
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
        try:
            beats = rewrite_beat(run.client, ctx, beats, beat_id, "these lines contradict the footage they play over: "
                                 + "; ".join(f"{c['problem']} (on screen: {c['on_screen']})" for c in mine),
                                 min_words=max(4, math.floor(words * 0.85)), max_words=words + 3)
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
    # Quality record
    lines = [s["text"] for b in beats if b["kind"] == "narration" for s in b["sentences"]]
    critiques = script.get("critiques", [])
    if critiques and min(v for k, v in critiques[-1].items() if k in ("hook", "premise", "causality",
                                                                       "storytelling", "ending", "clarity")) <= 2:
        issues.append({"check": "story_quality", "severity": "major", "origin": "gemini_review",
                       "description": f"story editor scores {critiques[-1]}"})
    if script.get("draft", {}).get("accepted_with_issue"):
        issues.append({"check": "story_rules", "severity": "major", "origin": "deterministic",
                       "description": script["draft"]["accepted_with_issue"]})
    if play_by_play(lines):
        issues.append({"check": "pacing", "severity": "major", "description": "frame-by-frame narration remains",
                       "origin": "deterministic"})
    borrowed = [e["sentence_id"] for e in evidence if e["borrowed_from"]]
    total = sum(b["duration"] for b in beats)
    if total < run.target * 0.8:
        issues.append({"check": "duration", "severity": "major", "origin": "deterministic",
                       "description": f"the Short is {total:.1f}s against a {run.target:.0f}s target"})
    review_status = "passed_with_warnings" if issues else "passed_automated_review"
    _write_json(outdir / "quality-review.json", {
        "review_status": review_status, "path": "storyline", "story_critique": critiques, "issues": issues,
        "sentences_borrowing_neighbour_footage": borrowed, "duration_seconds": round(total, 2),
        "note": "Automated review does not replace human playback review."})
    _write_json(outdir / "evidence-map.json", {"sentences": evidence, "policy": (
        "Every narration sentence plays over continuous footage of its own story item (neighbouring items only "
        "when that item's footage is spent); no sentence is removed for lack of footage.")})
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


def _fit_length(run, ctx: Context, beats: list[dict]) -> list[dict]:
    """Synthesize, then bring the measured length inside [0.85, 1.0] x target by whole-beat rewrites."""
    beats = run.synthesize(beats)
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
        beats = run.synthesize(beats)
    total = sum(b["duration"] for b in beats)
    if total > run.target + 0.05:
        raise ValueError(f"Measured narration is {total:.1f}s; the maximum is {run.target:.0f}s after "
                         f"{run.settings.max_timing_repairs} shortening pass(es). See correction-history.json.")
    return beats
