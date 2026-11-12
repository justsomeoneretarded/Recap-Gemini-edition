# Gemini production pipeline: architecture

**Central rule:** narration is built only from verified movie events, and the footage under each sentence must show what that sentence says. Each stage either produces evidence for that relationship or checks it.

Code: `shorts_pipeline/gemini/`. The original local pipeline (Gemma writer, Qwen3-VL frame descriptions) is unchanged. You can still select it with `planner: local` (workflow preset `local-gemma4`), or use it as an automatic fallback with `fallback_writer`.

```
source ─► Whisper ASR (existing) ─┐
       ─► PySceneDetect shots (existing) ─┐
       ─► analysis proxy (360/480p, burned-in source timecode) ─► Files API (uploaded once, reused by hash)
                                          │
A  analysis.py    windowed native video understanding ─► code checks ─► zoom re-inspection of uncertain/fast/important events
B  story.py       story map: observed vs inferred (cited), relations, characters with name basis
C  story.py       editorial selection: hook, causal chain, ending mode, dialogue clips, footage sufficiency
D  script.py      script draft (sentences cite events + visual requirement) ─► separate script-review pass
E  matching.py    Gemini watches each sentence's candidate clips ─► shot, in/out, anchor, focus, reason, verdict
F  matching.py    deterministic placement: relevance, no reuse, dialogue reserved, headroom, in-shot extension
G  pipeline/sync  Chatterbox (existing) ─► measured durations/words ─► overflow & shortfall repairs ─► per-sentence frames
H  review.py      deterministic checks + independent Gemini reviewer watching the placed shots
I  correction.py  targeted fixes (rematch / rewrite one beat / boundaries / retime), re-check, bounded passes
   captions.py, render.py (existing FFmpeg path; sentence-segmented allocation + action-anchor trimming added)
J  verify.py      measurements of the final MP4 (+ optional Gemini watch of the render)
```

## Stage details

| Stage | Model role | What code verifies independently | Artifact |
|---|---|---|---|
| A Analysis | `analysis` model; `VideoMetadata` start/end offsets, configurable fps and `media_resolution`; audio included | Timestamps fall inside the analysed window. Clip-relative times are detected and shifted. Each event overlaps a detected shot. Quoted dialogue must match nearby Whisper words (≥60%) or it is removed. Names are kept only with a valid basis. A zoom pass checks uncertain, fast or important events at up to 10 fps; `not_observed` events are rejected | `event-inventory.json` |
| B Story map | `planning` | Unknown event IDs are dropped; inferences must cite events or story notes; unsupported character names are removed; observed and inferred fields are kept separate | `story-map.json` |
| C Editorial | `planning` (high thinking) | IDs exist; chronological order (one declared cold open allowed); dialogue ≤25% of target; enough distinct footage for the narration | `editorial-plan.json` |
| D Script | `writing` draft, `planning` review | Every sentence cites an event. Quoted words must match verified dialogue. No channel filler, dangling endings or repetitive openings. The word budget comes from the measured voice rate | `script-draft.json` |
| E Matching | `matching`; one video part per candidate range (≤10 per call) | Chosen shots must be candidates. In/out points must lie within shot bounds. Choices with relevance <0.45 are rejected. Anchors are clamped. `unsupported` leads to a wider search, then a rewrite, then removal of the sentence | `evidence-map.json` |
| F Shot plan | none | No source range is reused across sentences. Retained dialogue ranges are reserved. Footage gets 20% headroom over measured speech. Extension stays inside the same shot. Shots play chronologically within a sentence | `storyboard.json` |
| G Sync | none (Chatterbox + Whisper) | Measured duration drives everything. Overflow triggers a targeted shortening rewrite. Footage shortfall triggers a rematch that excludes used shots, then shortening. Frames are allocated per sentence. Source speech is ducked under narration. Optional MMS forced alignment, rejected if it disagrees with ASR | `narration-timing.json` |
| H Review | `review` (separate prompt; watches placed clips with audio) | Deterministic checks: evidence, timestamps, shot bounds, reuse, anchors kept after trimming and ≥1 s on screen, names, duration, dialogue confidence, hook | `quality-review.json` |
| I Correction | as the fix requires | Each issue gets at most `max_rewrite_attempts` fixes and passes are capped by `review_passes`. A beat short of footage first gets its own verified event shots; it is shortened only when none remain, and then by at most a quarter. A content rewrite keeps at least 80% of the beat's words (60% when removing an unsupported claim). Footage placement is finalized before the first review. Only changed beats are re-voiced and re-reviewed; deterministic checks rerun on everything. Unresolved critical issues become `needs_human_review`, or a failure with `strict_review` | `correction-history.json` |
| J Verify | optional `review` watch of the MP4 (on issues in balanced, always in quality) | Duration, dimensions, fps, frame count, A/V end alignment, EBU R128 loudness and true peak, silence during timed words, tail, black/frozen video, SRT order, repeated or out-of-range source use | `final-verification.json` |

`production_ready` is always `false`. Automated checks are triage, not a substitute for watching the video.

## Hybrid script (default `script_mode: hybrid`)

Story-first writing, followed by a search for footage, can only *approximate* the match between words and picture. The hybrid mode makes that match hold by construction, and still lets the narration carry context the camera never shows.

1. **Clip plan** (`matching:clips`). For each selected event, Gemini watches the candidate shots and cuts 1–3 clips. Each clip has an in/out point, an action anchor, the subject's horizontal position, a role (action/reaction/detail/establishing) and a description of exactly what it shows. Code rejects clips outside their shot, clips under 0.6 s and overlaps. When nothing valid comes back, the verified event timestamps bound a fallback clip, which is flagged for review.
2. **Script A, picture-first** (`writing:picture`). The narration is written line by line *for* 1–3 consecutive clips. Each line must fit the clips' word capacity at the measured voice rate and name its key word. Code enforces clip order, no reuse, capacity, and that the key word is present.
3. **Script B, story-first** (`writing:draft` + `planning:script_review`). This is the evidence-referenced script described above, carrying motives, stakes and context.
4. **Merge** (`writing:reconcile`). A is the backbone. *Action* lines keep A's exact clips, though their wording may be improved and short context folded in. *Context* lines from B play over **unused cover clips**, and each must cite story notes or a story-map inference. Code enforces:
   - at least 60% of A's lines kept;
   - context under 35% of the words;
   - each cover clip used once;
   - playback order;
   - every existing script rule (citations, quotes, filler, dangling endings, word budget).

   An invalid merge gets two corrective retries, then the run falls back to Script B with ordinary sentence-to-shot matching, and the fallback is recorded.
5. **Key-word alignment.** After narration is measured, each action line's key clip gets the time of its key word. The renderer slides that clip's window *within its own shot* so the anchor frame is on screen when the word is spoken. A shot is only marked `key_word_aligned` when its range permits this to within half a frame; otherwise the residual offset is recorded.

The reviewer is told each line's kind: action lines must show their key word's action, and context lines need only related, non-contradicting cover footage.

## Static vs. agentic video processing

The SDK and models expose two relevant mechanisms: Files API video with per-part `VideoMetadata` clipping and fps, and per-request `media_resolution`. Rather than a single static pass over the whole film, Stage A runs an **agent-style loop under code control**:

1. A cheap coarse pass over shot-aligned windows.
2. Code decides which events are uncertain (confidence <0.7), fast, or important.
3. Only those get short, high-fps, higher-resolution clips.

Stage E does the same for matching: it shows only the candidate clips for each sentence. This keeps cost proportional to the uncertainty rather than to the length of the film. The loop's decisions are deterministic and recorded (`zoom_checks` in the inventory). Model-driven tool use, such as code execution for zooming into images, was not adopted: it would make cost and behaviour harder to bound and could not be verified offline.

## Models, cost and caching

- **Per-stage model preferences.** Each preset (`quality`, `balanced`, `economy`) lists ordered model candidates per stage: `analysis`, `planning`, `writing`, `matching` and `review`. At startup `models.list()` is called and the first available model is chosen. Any substitution is recorded. `gemini_models` overrides any stage. Model IDs come from the installed SDK's model catalogue, and none are assumed to exist.
- **Thinking.** Gemini 2.x models get `thinking_budget`; newer models get `thinking_level`.
- **Caching.**
  - Responses are cached by content hash, so a rerun of an unchanged stage costs nothing.
  - Uploads are reused until shortly before the 48-hour expiry.
  - Stages A–E are checkpointed (`work/gemini/runs/<key>/`) to support resumable runs.
  - Text stages share a byte-identical evidence prefix (helps implicit caching), and an explicit context cache is created when that prefix is large enough.
  - The workflow persists `work/production/gemini` with `actions/cache`.
- **Budget and cost reporting.**
  - A conservative **pre-flight estimate** must fit `max_budget_usd` before any upload.
  - Every call reserves its worst-case cost before it is sent.
  - `cost-report.json` records model, input/cached/output/thinking tokens, estimated USD, latency, retries with reasons, and stage durations.
  - Prices come from `settings.py`; newer models without a known price use conservative tier placeholders. You can supply exact prices with `SHORTS_GEMINI_PRICES` (the `GEMINI_PRICES_JSON` repository variable in Actions). The API does not report billed amounts; AI Studio billing is authoritative.
- **Retries.**
  - 429, 5xx and timeouts back off exponentially, up to 4 retries.
  - Auth and missing-model errors fail immediately with instructions.
  - Malformed JSON gets one repair request; the raw output is then saved to diagnostics.
  - Evidence-validation failures get one corrective retry.

## Free tier

`gemini_preset: free` (the default) changes how the run uses the API:

- **Models:** Flash candidates for every stage, then Flash-Lite. A model that answers with `limit: 0` (no free quota) is replaced by the next available candidate for the rest of the run, and the switch is recorded.
- **Pacing and caps:** requests are paced to `requests_per_minute` (default 8). Each run is capped at `max_requests` (default 150), and the preflight compares that cap with its request estimate.
- **Retries:** 429s wait for the server's suggested retry delay. A daily-quota 429 stops immediately with resume instructions.
- **Overload:** a 503 or 504 is retried at most `overload_attempts` (3) times on one model, then the next candidate takes over. The last remaining candidate gets the full `max_retries` budget.
- **Lighter workload:** explicit context caching is off (it needs billing). The run uses at most 6 zoom checks and 1 review pass, and no Gemini review of the rendered file.
- **Budget:** the dollar budget is not enforced because nothing is billed. The cost report shows the paid-price equivalent.

## Timestamps

All Gemini-facing times are seconds in the *supplied* source. When a `source_window` is used, that is the excerpt, exactly as the storyboard always was. The burned-in timecode on the analysis proxy shows the same clock. `source-attribution.json` still records the original-film window offset.
