"""Shot-level editing for discovered footage: a labelled shot catalog and fast, cue-timed cuts.

Run 20 played each sentence over one continuous 4-8 s window of its story item, so filler
(walking, reactions, empty frames) went on screen, one stretch ran 17 s without a cut, and
only 10 described moments covered 5 minutes of footage, so lines drifted from the picture.
The reference recaps cut about every 1.5-2 s, each cut showing what the words say.

1. Catalog: the pool's detected shots (sources.index_source) are split at story-item
   boundaries, and Gemini labels every shot of each item: what is visible, how intense it
   is (1-5), whether it is filler, its peak second and the subject's horizontal position.
2. The writer cites, for each sentence, the shots that show it and the words each one
   illustrates (`cue`).
3. Placement: each cited shot cuts in on the first word of its cue and stays on screen at
   most MAX_CUT seconds. Longer stretches are filled with the next best unused shots of the
   same moment (closest in time, most intense). Filler never plays unless nothing else is left.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .client import text_part, video_part

MAX_CUT = 2.2        # Seconds on screen for one shot.
FAST_CUT = 2.6       # Allowed for a fast, high-intensity shot that keeps moving.
HARD_MAX = 3.0       # No cut is ever held longer, even to avoid a sliver.
MIN_CUT = 0.6        # Shorter pieces would flash; the previous piece is held instead.
MIN_SHOT = 0.5       # Catalog shots shorter than this are dropped.
LABEL_BATCH = 45     # Shots per labelling request.


class ShotLabel(BaseModel):
    shot_id: str
    visible: str = Field(description="Concretely what is on screen: who does what to whom, with what")
    intensity: int = Field(ge=1, le=5, description="5 = peak action or shock, 1 = static or nothing happening")
    filler: bool = Field(description="True for shots a recap should never show: title cards, logos, black or blurred "
                                     "frames, empty establishing shots, a character idle with nothing happening")
    peak: float = Field(description="Pool second inside the shot where its action is clearest")
    motion: str = Field("normal", description="static, normal or fast")
    focus_x: float = Field(0.5, ge=0, le=1, description="Horizontal position of the key subject, 0 left to 1 right")


class ShotLabels(BaseModel):
    shots: list[ShotLabel]


LABEL_SYSTEM = ("You log film footage shot by shot for a recap editor. Describe only what is visible. Times are "
                "pool seconds (the clips are cut from one long video). Return JSON matching the schema.")


def catalog_shots(scenes: list[dict], ranges: dict[str, list[tuple[float, float]]]) -> list[dict]:
    """Detected shots, split at story-item boundaries, each owned by one item."""
    shots = []
    for item, spans in ranges.items():
        for a, b in spans:
            for scene in scenes:
                s, e = max(a, float(scene["start"])), min(b, float(scene["end"]))
                if e - s >= MIN_SHOT:
                    shots.append({"shot_id": f"{item}-{len([x for x in shots if x['item_id'] == item]) + 1:03d}",
                                  "item_id": item, "start": round(s, 3), "end": round(e, 3)})
    return sorted(shots, key=lambda x: x["start"])


def label_shots(client, file, shots: list[dict], items: dict[str, dict], log=print) -> dict[str, dict]:
    """Gemini labels every shot, one batch of one item's shots per request. Returns shot_id -> shot."""
    by_item: dict[str, list[dict]] = {}
    for shot in shots:
        by_item.setdefault(shot["item_id"], []).append(shot)
    labelled: dict[str, dict] = {}
    for item_id, mine in by_item.items():
        for first in range(0, len(mine), LABEL_BATCH):
            batch = mine[first:first + LABEL_BATCH]
            spans = _merge([(s["start"], s["end"]) for s in batch])
            item = items.get(item_id, {})
            log(f"Gemini: logging {len(batch)} shot(s) of {item.get('title', item_id)}")
            listing = [{"shot_id": s["shot_id"], "start": s["start"], "end": s["end"]} for s in batch]
            result = client.generate(
                "analysis:shots", ShotLabels,
                [video_part(file, a, b, 2.0) for a, b in spans]
                + [text_part(f"These clips show: {item.get('title', '')} - {item.get('what_happens', '')}. "
                             "They cover these pool ranges in order: " + str([[round(a, 2), round(b, 2)] for a, b in spans])
                             + ". Label every shot below (start/end are pool seconds).\n" + str(listing))],
                system=LABEL_SYSTEM, temperature=0.1, max_output_tokens=16384, resolution="low")
            answers = {label.shot_id: label for label in result.shots}
            for shot in batch:
                label = answers.get(shot["shot_id"])
                if label is None:
                    labelled[shot["shot_id"]] = {**shot, "visible": "", "intensity": 2, "filler": False,
                                                 "peak": (shot["start"] + shot["end"]) / 2, "motion": "normal",
                                                 "focus_x": 0.5, "unlabelled": True}
                    continue
                peak = label.peak if shot["start"] <= label.peak <= shot["end"] else (shot["start"] + shot["end"]) / 2
                labelled[shot["shot_id"]] = {**shot, "visible": label.visible.strip(), "intensity": label.intensity,
                                             "filler": label.filler, "peak": round(peak, 3), "motion": label.motion,
                                             "focus_x": label.focus_x}
    return labelled


def _merge(spans: list[tuple[float, float]], gap: float = 0.05) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


# ------------------------------------------------------------------ placement
class Footage:
    """Free (not yet shown) parts of each catalog shot."""

    def __init__(self, shots: dict[str, dict], order: list[str]):
        self.shots, self.order = shots, order
        self.free = {sid: [(s["start"], s["end"])] for sid, s in shots.items()}

    def reserve(self, a: float, b: float) -> None:
        for sid, parts in self.free.items():
            out = []
            for pa, pb in parts:
                if b <= pa or a >= pb:
                    out.append((pa, pb))
                    continue
                if a - pa >= MIN_CUT:
                    out.append((pa, a))
                if pb - b >= MIN_CUT:
                    out.append((b, pb))
            self.free[sid] = out

    def take(self, sid: str, length: float) -> tuple[float, float] | None:
        """A window of `length` (or the longest free part) of this shot, centred on its peak.
        Times stay unrounded until output: run 21 rounded a sliver to zero length and looped."""
        parts = [p for p in self.free.get(sid, []) if p[1] - p[0] >= MIN_CUT]
        if not parts or length <= 0:
            return None
        peak = self.shots[sid]["peak"]
        a, b = min(parts, key=lambda p: (0 if p[0] <= peak <= p[1] else 1, -(p[1] - p[0])))
        length = min(length, b - a)
        start = min(max(peak - length / 2, a), b - length)
        window = (start, start + length)
        self.reserve(*window)
        return window

    def grow(self, entry: list, extra: float) -> float:
        """Lengthen a placed window [sid, a, b] into free footage right next to it, inside its shot."""
        sid, a, b = entry
        extra = min(extra, max(0.0, HARD_MAX - (b - a)))
        if extra <= 0:
            return 0.0
        for pa, pb in self.free.get(sid, []):
            if abs(pa - b) < 1e-6:
                gain = min(extra, pb - pa)
                self.reserve(b, b + gain)
                entry[2] = b + gain
                return gain
            if abs(pb - a) < 1e-6:
                gain = min(extra, pb - pa)
                self.reserve(a - gain, a)
                entry[1] = a - gain
                return gain
        return 0.0

    def candidates(self, item: str, near: float, *, fill: bool) -> list[str]:
        """Shots to fill with: this item's unused shots, closest in time and most intense first."""
        position = self.order.index(item) if item in self.order else 0
        ranked = []
        for sid, shot in self.shots.items():
            if not any(p[1] - p[0] >= MIN_CUT for p in self.free[sid]):
                continue
            if shot["filler"] and not fill:
                continue
            distance_items = abs(self.order.index(shot["item_id"]) - position) if shot["item_id"] in self.order else 99
            untouched = self.free[sid] == [(shot["start"], shot["end"])]
            ranked.append(((distance_items, shot["filler"], not untouched, -shot["intensity"],
                            abs((shot["start"] + shot["end"]) / 2 - near)), sid))
        return [sid for _, sid in sorted(ranked)]


def cue_times(sentence: dict, words: list[dict], span: tuple[float, float], beat_start_words: int) -> list[float]:
    """Seconds from the sentence start at which each cited shot should cut in."""
    from ..voice import _normal

    tokens = [_normal(t) for t in sentence["text"].split() if _normal(t)]
    mine = words[beat_start_words:beat_start_words + len(tokens)]
    times = []
    for index, cue in enumerate(sentence.get("shots", [])):
        if index == 0:
            times.append(0.0)
            continue
        wanted = [_normal(t) for t in cue.get("cue", "").split() if _normal(t)]
        hit = None
        if wanted:
            for i in range(len(tokens) - len(wanted) + 1):
                if tokens[i:i + len(wanted)] == wanted and i < len(mine):
                    hit = float(mine[i]["start"]) - span[0]
                    break
        if hit is None:  # Unknown cue: spread the remaining shots evenly.
            hit = (span[1] - span[0]) * index / len(sentence["shots"])
        times.append(max(times[-1] + MIN_CUT, hit))
    return times


def place_sentence(footage: Footage, item: str, cited: list[str], cuts: list[float], duration: float,
                   sentence_index: int, sentence_id: str) -> tuple[list[dict], list[str]]:
    """Shots for one sentence: each cited shot from its cue, gaps filled so no cut exceeds MAX_CUT.

    Never fails: when the pool is spent, the last shot is held longer (the renderer may slow it to
    0.85x) and, as a last resort, a shot is shown again; both are noted.
    """
    plan: list[list] = []  # [sid, start, end]
    bounds = [min(max(t, 0.0), duration) for t in cuts] + [duration]
    notes: list[str] = []
    residue = 0.0
    for index, sid in enumerate(cited):
        segment = bounds[index + 1] - bounds[index] + residue
        residue = 0.0
        if segment < MIN_CUT and plan:
            residue = segment - footage.grow(plan[-1], segment)
            continue
        current, steps = sid, 0
        while segment > 0.04 and steps < 60:
            steps += 1
            shot = footage.shots[current]
            cap = FAST_CUT if shot["motion"] == "fast" and shot["intensity"] >= 4 else MAX_CUT
            piece = min(segment, cap)
            if segment - piece < MIN_CUT:  # Never leave a sliver: hold a little longer, or split evenly.
                piece = segment if segment <= min(cap + 0.4, HARD_MAX) else segment / 2
            window = footage.take(current, piece)
            if window is not None:
                plan.append([current, window[0], window[1]])
                segment -= window[1] - window[0]
            if segment <= 0.04:
                break
            nxt = [c for c in footage.candidates(item, shot["peak"], fill=False) if c != current]
            if not nxt:
                nxt = [c for c in footage.candidates(item, shot["peak"], fill=True) if c != current]
                if nxt:
                    notes.append("filler shown: the moment ran out of action shots")
            if nxt:
                current = nxt[0]
            elif window is None:
                break  # Nothing left anywhere.
        residue = max(0.0, segment)
    if residue > 0.04 and plan:
        residue -= footage.grow(plan[-1], residue)
    if residue > 0.04 and plan:
        # The pool is spent: show the strongest of this sentence's shots again, peak-centred.
        notes.append(f"repeated footage for {residue:.2f}s: the pool ran out")
        turn = 0
        while residue > 0.04:
            shot = footage.shots[cited[turn % len(cited)]]
            turn += 1
            piece = min(residue, MAX_CUT, shot["end"] - shot["start"])
            start = min(max(shot["peak"] - piece / 2, shot["start"]), shot["end"] - piece)
            plan.append([shot["shot_id"], start, start + piece])
            residue -= piece
    if not plan:  # Not even the cited shot was free: show it anyway.
        shot = footage.shots[cited[0]]
        piece = min(duration, shot["end"] - shot["start"])
        plan.append([cited[0], shot["start"], shot["start"] + piece])
        notes.append("cited shot shown again: no free footage was left")
    placed = []
    for sid, a, b in plan:
        shot = footage.shots[sid]
        placed.append({"start": round(a, 3), "end": round(b, 3), "anchor": round(min(max(shot["peak"], a), b), 3),
                       "focus_x": float(shot.get("focus_x", 0.5)), "sentence_index": sentence_index,
                       "sentence_id": sentence_id, "shot_id": sid, "event_id": "", "shows": shot.get("visible", ""),
                       "reason": "cited by the narration" if sid in cited else
                       ("same story moment" if shot["item_id"] == item else "neighbouring story moment"),
                       "relevance": 1.0 if sid in cited else (0.8 if shot["item_id"] == item else 0.6),
                       "shot_bounds": [shot["start"], shot["end"]], "trim_mode": "start"})
    return placed, notes


SLOP_PHRASES = ("little does he know", "little did he know", "in a world where", "all hell breaks loose",
                "stakes have never been higher", "deadly game", "cat and mouse", "collision course",
                "nothing will ever be the same", "the clock is ticking", "force to be reckoned with",
                "chaos ensues", "a deadly dance", "edge of", "embarks on", "journey", "testament to",
                "tapestry", "delve", "unleash", "unbeknownst", "sends shivers", "a storm of", "a hail of",
                "in a twist of fate", "fate has other plans", "one thing is certain", "only time will tell",
                "showdown of", "against all odds", "all-out war", "takes matters into his own hands",
                "total war", "high-stakes", "high stakes", "only option", "no other choice", "no choice but",
                "survival is", "survival requires", "absolutely", "every single", "at all costs")
SLOP_ADJECTIVES = ("deadly", "ruthless", "brutal", "relentless", "lethal", "fierce", "epic", "intense",
                   "terrifying", "desperate", "heart-pounding", "explosive", "savage", "vicious")
# A generic reason tacked on to sound causal (run 22: 'because survival is his only option', 'because escape is
# blocked', 'because lethal force is absolutely required'). A real reason is a story fact, and it is rare.
HOLLOW_REASON = re.compile(r"\bbecause (?:survival|escape|it is|it's|there is|there's|he must|he has to|"
                           r"lethal|force|danger|time|everyone|nothing|failure)\b")


def slop(lines: list[str]) -> list[str]:
    """Stock phrases and piled-up intensity adjectives that make narration sound generated."""
    text = " ".join(lines).casefold()
    found = [p for p in SLOP_PHRASES if p in text]
    adjectives = [w for w in re.findall(r"[a-z'-]+", text) if w in SLOP_ADJECTIVES]
    if len(adjectives) > max(2, len(lines) // 6):
        found.append(f"{len(adjectives)} stock intensity adjectives ({', '.join(sorted(set(adjectives)))})")
    hollow = HOLLOW_REASON.findall(text)
    if hollow:
        found.append(f"generic tacked-on reasons ({'; '.join(sorted(set(hollow)))}...): give the real story reason "
                     "(who wants what) or drop the clause")
    because = sum(1 for line in lines if re.search(r"\bbecause\b", line.casefold()))
    if because > max(2, len(lines) // 5):
        found.append(f"'because' in {because} of {len(lines)} sentences: most of those reasons are filler; keep "
                     "only the real ones and link the rest with so, but, until, when or a plain consequence")
    return found
