"""Structured outputs requested from Gemini, validated with Pydantic.

Schemas avoid unions/oneOf so they convert cleanly to Gemini response schemas.
Every model-produced value is still re-checked against source evidence by the
stage that consumes it: schema validity is not factual validity.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EventKind = Literal["action", "reaction", "dialogue", "reveal", "threat", "decision",
                    "consequence", "establishing", "transition", "other"]
NameBasis = Literal["none", "on_screen_text", "dialogue", "story_notes"]
Delivery = Literal["normal", "calm", "dramatic", "hype"]
Severity = Literal["critical", "major", "minor"]
ReviewCheck = Literal[
    "unsupported_statement", "shot_mismatch", "invalid_timestamp", "action_too_brief",
    "incoherent_story", "missing_event", "unsupported_identity_or_dialogue", "confusing_transition",
    "weak_opening", "weak_ending", "narration_footage_fit", "dialogue_clip_inaccurate",
    "repetition", "pacing", "other"]
FixType = Literal["rematch_shot", "rewrite_beat", "revise_opening", "restructure", "retime",
                  "remove_claim", "adjust_boundaries", "none"]


class Entity(BaseModel):
    label: str = Field(description="Visual descriptor, e.g. 'man in grey suit'.")
    name: str = Field("", description="Name only if established by the basis; otherwise empty.")
    name_basis: NameBasis = "none"


# ---------------------------------------------------------------- Stage A
class ObservedEvent(BaseModel):
    event_id: str
    start: float = Field(description="Source seconds, read from the burned-in timecode.")
    end: float
    peak: float = Field(description="Source second where the key action is most visible.")
    visible_action: str = Field(description="Only what is seen/heard. No speculation.")
    entities: list[Entity] = []
    objects: list[str] = []
    location: str = ""
    expression_or_reaction: str = ""
    dialogue_heard: str = Field("", description="Exact clearly audible words, else empty.")
    sound: str = ""
    kind: EventKind = "action"
    importance: int = Field(3, ge=1, le=5)
    confidence: float = Field(0.5, ge=0, le=1)
    motion: Literal["static", "normal", "fast"] = "normal"
    focus_x: float = Field(0.5, ge=0, le=1, description="Horizontal position (0 left..1 right) of the key subject.")
    uncertainty: str = ""


class WindowAnalysis(BaseModel):
    window_summary: str
    events: list[ObservedEvent]
    burned_in_captions: bool = Field(False, description="True if added subtitles/captions are burned into the picture.")
    voice_over_narration: bool = Field(False, description="True if a narrator (not a character on screen) talks over the footage.")


class EventCheck(BaseModel):
    event_id: str
    verdict: Literal["confirmed", "revised", "not_observed"]
    start: float
    end: float
    peak: float
    visible_action: str
    focus_x: float = Field(0.5, ge=0, le=1)
    notes: str = ""


class VerificationResult(BaseModel):
    checks: list[EventCheck]


# ---------------------------------------------------------------- Stage B
class Character(BaseModel):
    character_id: str
    label: str
    name: str = ""
    name_basis: NameBasis = "none"
    event_ids: list[str] = []


class InferredClaim(BaseModel):
    claim: str
    basis: Literal["story_notes", "dialogue", "visual_sequence"]
    supporting_event_ids: list[str]
    confidence: float = Field(0.5, ge=0, le=1)


class StoryEvent(BaseModel):
    event_id: str
    summary: str = Field(description="Observed content restated, no new facts.")
    character_ids: list[str] = []
    preceding_event_ids: list[str] = []
    following_event_ids: list[str] = []
    causes_event_ids: list[str] = []
    story_function: Literal["setup", "inciting", "escalation", "decision", "reversal",
                            "consequence", "climax", "resolution", "texture"] = "texture"
    importance: int = Field(3, ge=1, le=5)
    inferred_context: list[InferredClaim] = []
    needs_review: bool = False
    review_note: str = ""


class StoryMap(BaseModel):
    logline: str
    characters: list[Character]
    events: list[StoryEvent]
    open_questions: list[str] = []


# ---------------------------------------------------------------- Stage C
class SelectedEvent(BaseModel):
    event_id: str
    role: Literal["hook", "setup", "context", "escalation", "decision", "reversal",
                  "consequence", "climax", "payoff", "ending"]
    reason: str


class DroppedEvent(BaseModel):
    event_id: str
    reason: str


class EditorialPlan(BaseModel):
    title: str
    description: str
    hook_event_id: str
    cold_open: bool = Field(False, description="True only if the hook is shown before earlier events.")
    ending_mode: Literal["renewed_threat", "reveal", "reflection", "thematic_payoff", "resolved_payoff"]
    selected: list[SelectedEvent]
    dialogue_speech_ids: list[str] = []
    dropped: list[DroppedEvent] = []
    structure_notes: str = ""


# ---------------------------------------------------------------- Stage D
class ScriptSentence(BaseModel):
    text: str
    event_ids: list[str]
    visual_requirement: str = Field(description="What must be visible while this sentence plays.")
    claim_basis: Literal["observed", "story_notes", "dialogue"] = "observed"


class ScriptBeat(BaseModel):
    beat_id: str
    kind: Literal["narration", "dialogue"]
    delivery: Delivery = "normal"
    sentences: list[ScriptSentence] = []
    speech_id: str = ""


class ScriptDraft(BaseModel):
    title: str
    description: str
    beats: list[ScriptBeat]


class ScriptRevision(BaseModel):
    problems_found: list[str]
    changes_made: list[str]
    beats: list[ScriptBeat]


class BeatRewrite(BaseModel):
    beat_id: str
    sentences: list[ScriptSentence]
    rationale: str


# ---------------------------------------------------------------- Stage E
class ShotChoice(BaseModel):
    shot_id: str
    start: float
    end: float
    anchor: float = Field(description="Source second where the narrated action is clearest.")
    focus_x: float = Field(0.5, ge=0, le=1)
    shows: str = Field(description="What this footage actually shows.")
    reason: str = Field(description="Why it supports the sentence.")
    relevance: float = Field(ge=0, le=1)


class SentenceMatch(BaseModel):
    sentence_id: str
    verdict: Literal["supported", "partial", "unsupported"]
    shots: list[ShotChoice]
    problem: str = ""


class MatchingResult(BaseModel):
    matches: list[SentenceMatch]


# ---------------------------------------------------------------- Stage H/J
class ReviewIssue(BaseModel):
    check: ReviewCheck
    severity: Severity
    beat_id: str = ""
    sentence_id: str = ""
    source_start: float | None = None
    source_end: float | None = None
    description: str
    fix: FixType = "none"


class QualityReview(BaseModel):
    verdict: Literal["pass", "revise", "fail"]
    issues: list[ReviewIssue]
    strengths: list[str] = []
    opening_score: int = Field(3, ge=1, le=5)
    coherence_score: int = Field(3, ge=1, le=5)
    ending_score: int = Field(3, ge=1, le=5)


# ---------------------------------------------------------------- Hybrid: picture-first
class ClipChoice(BaseModel):
    shot_id: str
    start: float
    end: float
    anchor: float = Field(description="Source second where the event's key action is clearest.")
    focus_x: float = Field(0.5, ge=0, le=1)
    shows: str = Field(description="Concrete description of exactly what this clip shows.")
    role: Literal["action", "reaction", "detail", "establishing"] = "action"
    quality: float = Field(ge=0, le=1, description="How clearly and strikingly the clip shows its content.")


class EventClips(BaseModel):
    event_id: str
    clips: list[ClipChoice]


class ClipPlan(BaseModel):
    events: list[EventClips]


class PictureLine(BaseModel):
    clip_ids: list[str] = Field(description="1-3 consecutive clip IDs this line plays over.")
    text: str
    key_word: str = Field(description="The word naming the action/object the anchor frame shows.")


class PictureScript(BaseModel):
    lines: list[PictureLine]


class FinalLine(BaseModel):
    kind: Literal["action", "context"]
    text: str
    source_line_id: str = Field("", description="For action lines: the footage-first line ID it is based on.")
    clip_ids: list[str] = Field(description="Action: exactly the source line's clips. Context: unused cover clips.")
    key_word: str = ""
    event_ids: list[str] = []
    basis: Literal["observed", "story_notes", "inferred_context", "dialogue"] = "observed"


class ReconciledBeat(BaseModel):
    beat_id: str
    kind: Literal["narration", "dialogue"]
    delivery: Delivery = "normal"
    speech_id: str = ""
    lines: list[FinalLine] = []


class ReconciledScript(BaseModel):
    title: str
    description: str
    beats: list[ReconciledBeat]
    merge_notes: list[str] = []
