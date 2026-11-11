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


def _issue_key(issue: dict) -> tuple:
    return (issue["check"], issue.get("beat_id", ""), issue.get("sentence_id", ""))


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
    if beat is None or beat["kind"] != "narration":
        return beats, None, "issue does not point to a narration beat; left for human review"
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
        beats = run.rewrite(beats, beat_id, reason, max_words=words + 6)
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
