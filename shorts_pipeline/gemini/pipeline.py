"""Orchestrates Stages A–I and returns a synthesized, review-annotated storyboard.

Each expensive stage is checkpointed under work/gemini/runs/<key>/ with an
input fingerprint, so an interrupted run resumes without repeating finished
stages, and changing one input only reruns the stages that depend on it.
Gemini responses and uploads are separately content-cached by GeminiClient.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable

from .analysis import analyze_footage
from .client import BudgetExceeded, CostLedger, GeminiClient, _write_json
from .matching import match_sentences, plan_shots
from .review import deterministic_checks, gemini_review, summarize
from .script import count_words, rewrite_beat, write_script
from .settings import ANALYSIS_QUALITY, FRAME_TOKENS, GeminiSettings, planning_rate, price_for, settings_from_request
from .story import build_story_map, plan_editorial, usable_footage
from .sync import duck_source_dialogue, measured_sentence_seconds, timing_report

PIPELINE_VERSION = 1
TARGET_FILL = 0.93  # Aim slightly under the maximum; the measured voice decides the rest.


def _key(*values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def estimate_run_cost(settings: GeminiSettings, duration: float, target: float, models: dict) -> dict:
    """Conservative pre-run estimate (list prices; upper-bound token assumptions)."""
    q = ANALYSIS_QUALITY[settings.analysis_quality]

    def cost(stage, tokens_in, tokens_out):
        price = price_for(models[stage])
        return (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6

    windows = max(1, math.ceil(duration / q["window"]))
    analysis_in = duration * (q["fps"] * FRAME_TOKENS[q["resolution"]] + 32) + windows * 3000
    zoom_in = q["zoom_checks"] * 5 * (q["zoom_fps"] * FRAME_TOKENS[q["zoom_resolution"]] + 32)
    events = max(10, duration / 4)
    evidence = events * 160
    rows = {
        "analysis": cost("analysis", analysis_in + zoom_in, windows * 9000 + q["zoom_checks"] * 400),
        "planning": cost("planning", evidence * 3 + 12000, 24000),
        "writing": cost("writing", evidence * 2 + 10000, 16000 + 4000 * settings.review_passes),
        "matching": cost("matching", target * 2.5 * (q["match_fps"] * FRAME_TOKENS["low"] + 32) * 1.5, 12000),
        "review": cost("review", (target * (q["review_fps"] * FRAME_TOKENS["low"] + 32) + 15000) * (1 + settings.review_passes), 10000 * (1 + settings.review_passes)),
    }
    return {"by_stage_usd": {k: round(v, 4) for k, v in rows.items()}, "total_usd": round(sum(rows.values()), 4),
            "models": models, "basis": "upper-bound token assumptions x price table; actual usage is recorded per call"}


class Run:
    """Shared state for one Gemini production run."""

    def __init__(self, request: dict, source: Path, source_duration: float, workdir: Path, outdir: Path, *,
                 scenes: list[dict], speech: dict | None, transport=None, synthesize: Callable | None = None,
                 log=print, sleep=None):
        from ..planner import _source_speech_pool

        self.request, self.source, self.duration = request, Path(source), float(source_duration)
        self.workdir, self.outdir, self.log = Path(workdir), Path(outdir), log
        self.settings = settings_from_request(request)
        self.ledger = CostLedger(self.settings.max_budget_usd, self.outdir / "cost-report.json")
        self.client = GeminiClient(self.settings, self.ledger, self.workdir / "gemini" / "cache", transport, sleep=sleep)
        self.scenes = scenes
        self.speech = speech or {}
        self.segments = [s for s in self.speech.get("segments", []) if s.get("start") is not None and s.get("end") is not None]
        self.transcript = " ".join(str(s.get("text", "")) for s in self.segments)
        self.speech_pool = _source_speech_pool(self.speech.get("candidates") or [], self.duration)
        self.target = float(request["target_seconds"])
        self.fps = int(request.get("fps", 60))
        self.rate = planning_rate(request["narrator"], request["tts_model"])
        if synthesize is None:
            from ..voice import synthesize_plan
            synthesize = synthesize_plan
        self._synthesize = synthesize
        self.history: list[dict] = []
        self.matches: dict = {}
        self.file = None
        self.run_dir = self.workdir / "gemini" / "runs" / _key(PIPELINE_VERSION, str(self.source),
                                                                 request.get("title"), request.get("brief"))[:20]

    # ---- checkpoints
    def checkpoint(self, name: str, inputs, compute: Callable[[], object], artifact: str | None = None):
        path = self.run_dir / f"{name}.json"
        key = _key(PIPELINE_VERSION, name, inputs)
        started = time.time()
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("key") == key:
                self.log(f"Gemini: resumed {name} from checkpoint")
                self.ledger.note(kind="checkpoint_resumed", stage=name)
                data = saved["data"]
                if artifact:
                    _write_json(self.outdir / artifact, data)
                return data
        data = compute()
        _write_json(path, {"key": key, "data": data})
        if artifact:
            _write_json(self.outdir / artifact, data)
        self.ledger.time_stage(name, time.time() - started)
        return data

    def ensure_file(self):
        if self.file is None:
            from .analysis import make_proxy
            proxy, _ = make_proxy(self.source, self.workdir, 480 if self.settings.analysis["resolution"] != "low" else 360)
            self.file = self.client.upload(proxy)
        return self.file

    # ---- synthesis and placement
    def synthesize(self, beats: list[dict]) -> list[dict]:
        started = time.time()
        plan = {"beats": copy.deepcopy(beats), "source_path": str(self.source)}
        result = self._synthesize(plan, self.request["narrator"], self.workdir, model=self.request["tts_model"])
        beats = result["beats"]
        mode = self.request.get("forced_alignment", "off")
        if mode != "off":
            from .. import alignment
            if alignment.available():
                for beat in beats:
                    if beat["kind"] == "narration" and beat.get("audio_path"):
                        try:
                            beat["words"], beat["timing_provenance"] = alignment.refine_words(
                                beat["audio_path"], beat["words"], beat["duration"])
                        except Exception as exc:  # noqa: BLE001 - keep ASR estimate
                            self.ledger.note(kind="forced_alignment_failed", beat=beat["id"], error=str(exc)[:200])
            else:
                self.ledger.note(kind="forced_alignment_unavailable", detail="torchaudio MMS_FA not importable")
        self.ledger.time_stage("G_narration_sync", time.time() - started)
        return beats

    def place(self, beats: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        measured = measured_sentence_seconds(beats) if all(
            b.get("words") for b in beats if b["kind"] == "narration") else None
        return plan_shots(beats, self.matches, self.rate, measured=measured)

    def rematch(self, beats: list[dict], sentence_ids: set[str], *, wide: bool = False, exclude: dict | None = None):
        self.matches.update(match_sentences(self.client, self.ensure_file(), beats, self.inventory, self.scenes,
                                            only=sentence_ids, wide=wide, exclude=exclude, log=self.log))

    def rewrite(self, beats: list[dict], beat_id: str, reason: str, *, max_words: int | None = None,
                min_words: int | None = None) -> list[dict]:
        index = next(i for i, b in enumerate(beats) if b["id"] == beat_id)
        beat = beats[index]
        neighbours = [{"beat_id": b["id"], "text": b["text"]} for b in beats[max(0, index - 1):index + 2] if b is not beat]
        allowed = sorted({s["event_id"] for s in self.editorial["selected"]}
                         | {i for s in beat["sentences"] for i in s["event_ids"]})
        new = rewrite_beat(self.client, beat, reason, self.inventory, self.speech_pool, self.request,
                           max_words=max_words, min_words=min_words, allowed_event_ids=allowed, context=neighbours)
        for key in ("audio_path", "duration", "words", "timing_provenance"):
            new.pop(key, None)
        beats = beats[:index] + [new] + beats[index + 1:]
        for sentence in beat["sentences"]:
            self.matches.pop(sentence["sentence_id"], None)
        self.rematch(beats, {s["sentence_id"] for s in new["sentences"]})
        return beats

    def record(self, **entry):
        entry["at_epoch"] = round(time.time(), 2)
        self.history.append(entry)
        _write_json(self.outdir / "correction-history.json", self.history)

    def review(self, beats: list[dict], *, gemini: bool, only: set[str] | None = None, previous: list[dict] | None = None):
        issues = deterministic_checks(beats, inventory=self.inventory, speech_pool=self.speech_pool,
                                      source_duration=self.duration, fps=self.fps, target=self.target,
                                      notes=self.request.get("story_notes", ""), transcript=self.transcript,
                                      hook_event_id=self.editorial.get("hook_event_id"))
        summaries = []
        if gemini:
            found, summaries = gemini_review(self.client, self.ensure_file(), beats, inventory=self.inventory,
                                             story_map=self.story_map, editorial=self.editorial, only_beats=only,
                                             log=self.log)
            issues += found
            if only is not None and previous:
                issues += [i for i in previous if i.get("origin") == "gemini_review" and i.get("beat_id") not in only]
        return issues, summaries


def _timing_loop(run: Run, beats: list[dict]) -> list[dict]:
    """Synthesize, then fix duration overflow and sentence footage shortfalls (bounded)."""
    widened: set[str] = set()
    rewritten: set[str] = set()
    for attempt in range(run.settings.max_timing_repairs + 1):
        beats = run.synthesize(beats)
        total = sum(b["duration"] for b in beats)
        if total > run.target + 0.05:
            if attempt == run.settings.max_timing_repairs:
                raise ValueError(f"Measured narration is {total:.1f}s after {attempt} shortening pass(es); the maximum is "
                                 f"{run.target:.0f}s. See correction-history.json.")
            excess = total - run.target * 0.97
            for beat in sorted((b for b in beats if b["kind"] == "narration"), key=lambda b: -b["duration"]):
                if excess <= 0:
                    break
                words = count_words(beat["text"])
                cut = min(math.ceil(words * 0.3), math.ceil(excess * words / beat["duration"] * 1.15))
                try:
                    beats = run.rewrite(beats, beat["id"], f"the narration is {total:.1f}s but must fit {run.target:.0f}s; "
                                        "shorten this beat while keeping its key action", max_words=max(4, words - cut))
                    excess -= cut * beat["duration"] / words
                    run.record(stage="G", action="shorten_beat", beat_id=beat["id"], reason="duration overflow",
                               measured_total=round(total, 2), outcome="rewritten")
                except ValueError as exc:
                    run.record(stage="G", action="shorten_beat", beat_id=beat["id"], outcome="failed", error=str(exc)[:300])
            continue
        beats, evidence, shortfalls = run.place(beats)
        if not shortfalls:
            break
        changed = False
        wide = {s["sentence_id"] for s in shortfalls} - widened
        if wide:
            # Footage already placed under other sentences is excluded so the search finds new shots.
            exclude = {sid: [sh["shot_id"] for b in beats for sh in b.get("shots", [])
                             if sh.get("sentence_id") not in (None, sid) and sh.get("shot_id")] for sid in wide}
            run.rematch(beats, wide, wide=True, exclude=exclude)
            widened |= wide
            run.record(stage="G", action="widen_footage_search", sentence_ids=sorted(wide), outcome="rematched")
            beats, evidence, shortfalls = run.place(beats)
            changed = True
        for beat_id in sorted({s["beat_id"] for s in shortfalls} - rewritten):
            beat = next(b for b in beats if b["id"] == beat_id)
            mine = [s for s in shortfalls if s["beat_id"] == beat_id]
            spoken = beat.get("duration") or 0.1
            available = sum(e["footage_seconds"] for e in evidence if e["beat_id"] == beat_id)
            words = count_words(beat["text"])
            limit = max(4, math.floor(words * min(1.0, available / max(spoken, 0.1)) * 0.92))
            try:
                beats = run.rewrite(beats, beat_id, "there is not enough verified footage to cover this narration "
                                    f"({', '.join(s['sentence_id'] for s in mine)}); say less", max_words=limit)
                rewritten.add(beat_id)
                changed = True
                run.record(stage="G", action="shorten_for_footage", beat_id=beat_id, max_words=limit, outcome="rewritten")
            except ValueError as exc:
                run.record(stage="G", action="shorten_for_footage", beat_id=beat_id, outcome="failed", error=str(exc)[:300])
        if not changed:
            break
    if any(b["kind"] == "narration" and not b.get("words") for b in beats):
        beats = run.synthesize(beats)
    beats, _, _ = run.place(beats)
    return beats


def run_gemini_pipeline(request: dict, source: Path, source_duration: float, workdir: Path, outdir: Path, *,
                        scenes: list[dict], speech: dict | None, transport=None, synthesize=None,
                        log=print, sleep=None) -> tuple[dict, Run]:
    from .correction import correction_loop

    run = Run(request, source, source_duration, workdir, outdir, scenes=scenes, speech=speech,
              transport=transport, synthesize=synthesize, log=log, sleep=sleep)
    settings = run.settings
    if source_duration > settings.max_source_minutes * 60:
        raise ValueError(f"Source is {source_duration/60:.1f} minutes; the Gemini analysis limit is "
                         f"{settings.max_source_minutes:g} minutes. Use source_window for the relevant excerpt, "
                         "or deliberately raise max_source_minutes.")
    log("Gemini: checking API key and model availability")
    models = run.client.resolve_models()
    estimate = estimate_run_cost(settings, source_duration, run.target, models)
    _write_json(outdir / "gemini-settings.json", {**settings.to_json(), "preflight_estimate": estimate})
    if estimate["total_usd"] > settings.max_budget_usd:
        raise BudgetExceeded(f"Estimated Gemini cost ${estimate['total_usd']:.2f} exceeds the ${settings.max_budget_usd:.2f} "
                             f"budget before any upload ({estimate['by_stage_usd']}). Raise max_budget_usd, choose a "
                             "cheaper preset/analysis quality, or shorten the source window.")
    source_key = _key(str(source), source_duration, [(s["id"], s["start"], s["end"]) for s in scenes])

    # A — footage understanding
    run.inventory = run.checkpoint(
        "A-event-inventory", [source_key, settings.analysis, models["analysis"], request.get("story_notes", ""),
                              run.speech.get("cache_key")],
        lambda: analyze_footage(run.client, source, workdir, scenes, run.speech, request.get("story_notes", ""),
                                source_duration, log), "event-inventory.json")
    if not run.inventory["events"]:
        raise ValueError("Gemini found no verifiable story events in the footage. Check the source window.")
    # B — story map
    run.story_map = run.checkpoint(
        "B-story-map", [run.inventory, models["planning"], request.get("story_notes"), request.get("brief")],
        lambda: build_story_map(run.client, run.inventory, run.speech_pool, request, run.transcript), "story-map.json")
    # C — editorial selection
    narration_seconds = run.target * TARGET_FILL
    run.editorial = run.checkpoint(
        "C-editorial", [run.story_map, run.target, models["planning"]],
        lambda: plan_editorial(run.client, run.inventory, run.story_map, run.speech_pool, request, scenes,
                               narration_seconds), "editorial-plan.json")
    dialogue = run.editorial["dialogue_seconds"]
    footage = usable_footage([s["event_id"] for s in run.editorial["selected"]], run.inventory, scenes)
    seconds = max(5.0, min(run.target * TARGET_FILL - dialogue, footage * 0.95))
    budget_words = (math.floor(seconds * run.rate * 0.82), math.floor(seconds * run.rate))
    # D — script and script review
    script = run.checkpoint(
        "D-script", [run.editorial, budget_words, models["writing"], models["planning"]],
        lambda: write_script(run.client, run.inventory, run.story_map, run.editorial, run.speech_pool, request,
                             budget_words), "script-draft.json")
    beats = copy.deepcopy(script["beats"])
    # E — sentence-to-shot matching
    script_key = _key(beats)
    run.matches = run.checkpoint(
        "E-matches", [script_key, models["matching"], settings.analysis["match_fps"]],
        lambda: match_sentences(run.client, run.ensure_file(), beats, run.inventory, scenes, log=log))
    # Unsupported sentences: one wide search, then a rewrite, then removal.
    unsupported = {sid for sid, m in run.matches.items() if not m["shots"]}
    if unsupported:
        run.rematch(beats, unsupported, wide=True)
        run.record(stage="E", action="widen_footage_search", sentence_ids=sorted(unsupported), outcome="rematched")
    for beat in list(beats):
        missing = [s for s in beat.get("sentences", []) if not run.matches.get(s["sentence_id"], {}).get("shots")]
        if not missing:
            continue
        try:
            beats = run.rewrite(beats, beat["id"], "no footage visibly shows: " + " | ".join(s["text"] for s in missing)
                                + ". Narrate only actions that the cited events visibly show")
            run.record(stage="E", action="rewrite_unsupported", beat_id=beat["id"], outcome="rewritten")
        except ValueError as exc:
            run.record(stage="E", action="rewrite_unsupported", beat_id=beat["id"], outcome="failed", error=str(exc)[:300])
    beats = _drop_unsupported(run, beats)
    # F — shot plan (estimated timing; replaced after measurement)
    beats, evidence, _ = plan_shots(beats, run.matches, run.rate)
    _write_json(outdir / "storyboard-draft.json", {"beats": beats})
    # G — measured narration and synchronization
    log("Gemini: narration synthesis and synchronization")
    beats = _timing_loop(run, beats)
    # H + I — independent review and bounded targeted corrections
    beats, issues, summaries = correction_loop(run, beats)
    beats, evidence, shortfalls = _finalize_placement(run, beats)
    ducking = duck_source_dialogue(beats, run.segments)
    unresolved = [i for i in issues if i["severity"] == "critical"]
    review_status = "needs_human_review" if unresolved else (
        "passed_with_warnings" if any(i["severity"] == "major" for i in issues) else "passed_automated_review")
    quality = {"review_status": review_status, "review_passes": settings.review_passes,
               "gemini_review_summaries": summaries, **summarize(issues),
               "note": "Automated review does not replace human playback review."}
    _write_json(outdir / "quality-review.json", quality)
    _write_json(outdir / "evidence-map.json", {"sentences": evidence, "shortfalls": shortfalls,
                                               "policy": "Every narration sentence is placed only on footage chosen "
                                                         "for it with a recorded reason; unsupported sentences are "
                                                         "rewritten or removed, never padded with unrelated shots."})
    _write_json(outdir / "narration-timing.json", {**timing_report(beats), "source_ducking": ducking})
    if unresolved and settings.strict_review:
        raise RuntimeError(f"{len(unresolved)} critical review issue(s) remain after {settings.review_passes} correction "
                           "pass(es); see quality-review.json and correction-history.json.")
    plan = {"title": request["title"], "description": script.get("description") or run.editorial["description"],
            "beats": beats, "planning_mode": "gemini-multistage",
            "planning": {"version": PIPELINE_VERSION, "preset": settings.preset, "models": models,
                         "review_status": review_status, "planning_words_per_second": round(run.rate, 3),
                         "coverage": run.inventory["coverage"]}}
    run.ledger.save()
    return plan, run


def _finalize_placement(run: Run, beats: list[dict]):
    """Place footage; any sentence left without footage is removed and the beat re-voiced."""
    for _ in range(3):
        beats = _drop_unsupported(run, beats)
        if any(b["kind"] == "narration" and not b.get("words") for b in beats):
            beats = run.synthesize(beats)
        beats, evidence, shortfalls = run.place(beats)
        empty = {e["sentence_id"] for e in evidence if not e["shots"]}
        if not empty:
            return beats, evidence, shortfalls
        for sid in empty:
            run.matches[sid] = {**run.matches.get(sid, {}), "sentence_id": sid, "verdict": "unsupported", "shots": [],
                                "problem": "all matched footage was already used by other sentences"}
    raise ValueError("Footage placement did not converge: sentences kept losing their only supporting shots.")


def _drop_unsupported(run: Run, beats: list[dict]) -> list[dict]:
    """Last resort: remove sentences that still have no supporting footage."""
    result = []
    for beat in beats:
        if beat["kind"] != "narration":
            result.append(beat)
            continue
        keep = [s for s in beat["sentences"] if run.matches.get(s["sentence_id"], {}).get("shots")]
        removed = [s for s in beat["sentences"] if s not in keep]
        if removed:
            run.record(stage="E", action="remove_unsupported_sentence", beat_id=beat["id"],
                       sentences=[s["text"] for s in removed], outcome="removed")
        if not keep:
            continue
        if removed:
            beat = {**beat, "sentences": keep, "text": " ".join(s["text"] for s in keep)}
            for key in ("audio_path", "duration", "words", "timing_provenance"):
                beat.pop(key, None)
        result.append(beat)
    if not any(b["kind"] == "narration" for b in result):
        raise ValueError("No narration sentence could be matched to supporting footage; nothing honest to render.")
    return result
