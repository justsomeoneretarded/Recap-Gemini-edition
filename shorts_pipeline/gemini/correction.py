"""Stage I: bounded, targeted corrections driven by the Stage H review.

Only the component named by an issue is changed: a sentence is re-matched, a
single beat is rewritten, or shot boundaries are adjusted. Changed beats are
re-synthesized (unchanged narration is reused from the voice cache), placement
is recomputed, deterministic checks rerun on everything, and the Gemini
reviewer re-examines only the changed beats. Each issue key gets at most
`max_rewrite_attempts` fixes; passes are capped by `review_passes`.
Unresolved critical problems are reported, never hidden.
"""
from __future__ import annotations

import math

from .script import count_words

ACTIONABLE = ("critical", "major")
MIN_KEEP = 0.8  # A review-driven rewrite keeps at least this share of the beat's words.


def _issue_key(issue: dict) -> tuple:
    return (issue["check"], issue.get("beat_id", ""), issue.get("sentence_id", ""))


def _cover_beat(run, beats: list[dict], beat: dict) -> tuple[list[dict], str | None, str]:
    """A beat with too little footage first gets its own verified event shots; it is shortened
    only if those are already in use, and then by at most a quarter."""
    from .pipeline import _event_footage

    added = []
    for sentence in beat["sentences"]:
        sid = sentence["sentence_id"]
        match = run.matches.get(sid, {"sentence_id": sid, "verdict": "unsupported", "shots": []})
        known = {(sh["start"], sh["end"]) for sh in match.get("shots", [])}
        extra = [sh for sh in _event_footage(run, sentence) if (sh["start"], sh["end"]) not in known]
        if extra:
            run.matches[sid] = {**match, "shots": list(match.get("shots", [])) + extra}
            added += [sh["shot_id"] for sh in extra]
    if added:
        return beats, beat["id"], f"added verified event footage to {beat['id']}: {', '.join(added)}"
    words = count_words(beat["text"])
    limit = max(4, math.ceil(words * 0.75))
    if limit >= words:
        return beats, None, f"{beat['id']} has no more verified footage and is too short to cut; left for final placement"
    beats = run.rewrite(beats, beat["id"], "there is not enough verified footage to cover this narration; say a "
                        "little less while keeping its key action", max_words=limit,
                        min_words=max(4, math.ceil(words * 0.6)))
    return beats, beat["id"], f"shortened {beat['id']} to {limit} words: no more verified footage"


def _fix(run, beats: list[dict], issue: dict) -> tuple[list[dict], str | None, str]:
    """Apply one fix. Returns (beats, changed_beat_id or None, description)."""
    fix, beat_id, sid = issue["fix"], issue.get("beat_id", ""), issue.get("sentence_id", "")
    beat = next((b for b in beats if b["id"] == beat_id), None)
    if fix == "retime" and issue["check"] == "pacing":
        narration = [b for b in beats if b["kind"] == "narration"]
        if not narration:
            return beats, None, "no narration to shorten"
        longest = max(narration, key=lambda b: b.get("duration", 0))
        words = count_words(longest["text"])
        beats = run.rewrite(beats, longest["id"], issue["description"] + "; shorten this beat",
                            max_words=max(4, math.floor(words * 0.8)))
        return beats, longest["id"], f"shortened {longest['id']} to fit the duration"
    if beat is not None and beat["kind"] == "dialogue" and issue["check"] in (
            "dialogue_clip_inaccurate", "unsupported_identity_or_dialogue", "incoherent_story", "other"):
        # A retained source clip that is not clean, relevant film dialogue is simply dropped.
        return [b for b in beats if b["id"] != beat_id], beat_id, f"removed dialogue clip {beat_id}: {issue['description'][:120]}"
    if beat is None or beat["kind"] != "narration":
        return beats, None, "issue does not point to a narration beat; left for human review"
    if issue["check"] == "narration_footage_fit" and fix == "retime" and not sid:
        return _cover_beat(run, beats, beat)
    if fix in ("rematch_shot", "adjust_boundaries", "retime") and sid:
        current = [s["shot_id"] for s in beat.get("shots", []) if s.get("sentence_id") == sid]
        if fix == "adjust_boundaries" and len(run.matches.get(sid, {}).get("shots", [])) > 1:
            shots = sorted(run.matches[sid]["shots"], key=lambda c: -c["relevance"])
            dropped = shots.pop()
            run.matches[sid] = {**run.matches[sid], "shots": shots}
            return beats, beat_id, f"dropped weakest shot {dropped['shot_id']} so the key action gets more screen time"
        exclude = current if issue["check"] in ("shot_mismatch", "repetition") else []
        run.rematch(beats, {sid}, wide=True, exclude={sid: exclude})
        return beats, beat_id, f"re-matched {sid} excluding {exclude or 'nothing'}"
    if fix in ("rewrite_beat", "remove_claim", "revise_opening", "restructure", "rematch_shot", "retime"):
        reason = issue["description"]
        if fix == "remove_claim":
            reason = "remove or correct this unsupported claim: " + reason
        if fix == "revise_opening":
            reason = (f"strengthen the opening using the hook event {run.editorial.get('hook_event_id')}: " + reason)
        words = count_words(beat["text"])
        # A targeted fix changes what the beat says, not how much: it must not gut the story.
        beats = run.rewrite(beats, beat_id, reason + "; keep the beat about as long as it is now",
                            max_words=words + 6, min_words=max(4, math.ceil(words * (0.6 if fix == "remove_claim" else MIN_KEEP))))
        return beats, beat_id, f"rewrote {beat_id}"
    return beats, None, f"no automatic fix for {fix}; left for human review"


def correction_loop(run, beats: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    settings = run.settings
    use_gemini = settings.review_passes > 0
    issues, summaries = run.review(beats, gemini=use_gemini)
    run.record(stage="H", action="review", pass_number=0, counts={s: sum(1 for i in issues if i["severity"] == s)
                                                                  for s in ("critical", "major", "minor")})
    attempts: dict[tuple, int] = {}
    for number in range(1, settings.review_passes + 1):
        actionable = [i for i in issues if i["severity"] in ACTIONABLE and i["fix"] != "none"]
        if not actionable:
            break
        changed: set[str] = set()
        handled: set[tuple] = set()
        for issue in sorted(actionable, key=lambda i: i["severity"] != "critical"):
            key = _issue_key(issue)
            target = (issue.get("beat_id"), issue.get("sentence_id"), issue["fix"])
            if key in handled or target in handled or attempts.get(key, 0) >= settings.max_rewrite_attempts:
                continue
            if issue.get("beat_id") in changed and issue["fix"] in ("rewrite_beat", "remove_claim", "revise_opening", "restructure"):
                continue  # The beat was already rewritten in this pass; recheck first.
            attempts[key] = attempts.get(key, 0) + 1
            handled |= {key, target}
            try:
                beats, beat_id, action = _fix(run, beats, issue)
                outcome = "applied" if beat_id else "not_applicable"
                if beat_id:
                    changed.add(beat_id)
            except (ValueError, RuntimeError) as exc:
                action, outcome = f"fix failed: {str(exc)[:300]}", "failed"
            run.record(stage="I", pass_number=number, issue=issue, action=action, outcome=outcome)
        if not changed:
            break
        from .pipeline import _finalize_placement
        beats, _, _ = _finalize_placement(run, beats)
        changed &= {b["id"] for b in beats}
        previous = issues
        issues, more = run.review(beats, gemini=use_gemini and bool(changed), only=changed, previous=previous)
        summaries += more
        run.record(stage="H", action="re-review", pass_number=number, rechecked_beats=sorted(changed),
                   counts={s: sum(1 for i in issues if i["severity"] == s) for s in ("critical", "major", "minor")})
    return beats, issues, summaries
