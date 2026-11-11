# Validation record

Verified6November2026. The [complete automatic GitHub workflow passed](https://github.com/justsomeoneretarded/Recap-Short-form/actions/runs/42165849180), using the current Gemma4 writing baseline and original Chatterbox/Warguy narration. The code commit tested was55794516af3bc0876cde8a45475272b51e82884b. This is an engineering validation, not a claim of reference-identical or channel-ready production quality.

## Complete automatic workflow

- Run42165849180 completed in11minutes2seconds. It restored nine representative scene descriptions and the canonical source excerpt from run42163298499, then generated a fresh Gemma4 storyboard, original-Warguy speech, captions and video. No supplied storyboard or imported narration bypassed these stages in the successful workflow.
- The request used a15-second maximum and original-source seconds24.24–43.8 of V1. The output is **14.000seconds,1080×1080,60fps**, with37model-written words. The measured speech files total13.92seconds; the decoded video contains836frames over13.933333seconds. Stereo48kHz AAC ends at14.000seconds and covers the last voice endpoint.
- Independent inspection confirmed all nine selected visual anchors remain in order, with positive frame allocations, bounded source ranges and no repeated/looped footage. All37script words match the timeline and SRT; ASS preserves them after capitalization. Word intervals are finite, positive, ordered and inside the corresponding audio.
- Pipeline QC passed its eight checks. Independent final-AAC measurement found **−14.13LUFS /−1.49dBTP**. These are measured results for this file, rather than a general loudness certification.
- Retained evidence: [video](validation-samples/automatic-warguy-pilot.mp4), [original generated storyboard](validation-samples/automatic-warguy-original-plan.json), [audit/QC](validation-samples/automatic-warguy-verification.json), and [contact sheet](validation-samples/automatic-warguy-contact-sheet.jpg). The downloadable workflow artifact also contains the script, description, SRT/ASS, measured timeline, caption-timing provenance, source attribution and three cover candidates.
- The source is an already edited reference Short. Its original captions visibly overlap the generated captions and its quiet old narration remains in the source bed. This engineering sample must not be presented as a new channel-ready recap. Script wording such as “active for his head” also needs editorial polish.

## Completed component checks

- **148 tests passed locally** in clean Python3.11; the [GitHub suite passed148 in10.98seconds](https://github.com/justsomeoneretarded/Recap-Short-form/actions/runs/42165712772) on Python3.11.18 and Ubuntu24.04/FFmpeg6.1.1. The earlier119-test revision also passed in Python3.12. Tests include actual synthetic A/V renders, dialogue/VO separation, audible voice-tail preservation with repeated music, vertical crops, caption reveal/fade pixels, CLI generation, source and planner failures, native-quote preservation, cache restoration, model-default changes and bounded narration repair.
- All eight original reference release assets were retrieved and SHA-256 checked against GitHub's asset digests, about485MiB total. Exports are1080-square H.264 at reported60fps,135.69–179.53seconds. Full word/segment transcripts and scene-change/loudness measurements are retained. Visual examination used sampled frames and contact sheets, not exhaustive frame-by-frame viewing or subjective audio audition.
- Graphify0.9.102 with video dependencies was installed separately and its graph/query/path workflow verified. The merged graph preserves timestamped reference evidence alongside code relationships. Original media/transcripts remain the primary evidence.
- Actual CPU inference and ASR script matching passed for all three voices on the same13-word sentence. Original Chatterbox: **Warguy4.72s, KitLaser4.80s, Female4.44s**. Nano: **Warguy4.76s, KitLaser4.44s, Female4.88s**. [WAV samples](validation-samples/README.md) are retained; numerical checks do not establish perceived voice likeness or sustained delivery quality.
- A real known-English V1 source-speech run with the local small.en override recovered44 bounded candidates, including22.36–23.78 “I wish you good luck on your path,”121.38–122.38 “Good luck, Mr.Wick,” and126.04–126.56 “Thanks, Doc.” Its pool also contains reference narration. Source-speech analysis does not identify speakers or distinguish pre-narrated Shorts from clean film audio.
- A10fps V1 filmstrip at1.8–4.6seconds confirmed progressive caption reveal/fade in that passage. The custom ASS implementation approximates white word alpha and stable phrase layout while preserving SRT text/times; the creator's exact font/mask algorithm is unknown.
- Real failures drove fixes: short source-bed audio could truncate narration; excess-footage trimming could drop the final visual anchor; FFmpeg6 music/sidechain EOF could lose the voice tail. Regressions now verify full audible endpoints and all four synthetic visual anchors. Silent/music branches are finite and bounded. The original FFmpeg6 stall's precise cause was not reproduced independently.
- Prepared excerpts now persist inside the restored source cache. A real-media test proves byte-preserving reuse after restoring only that cache and invalidation for changed footage/windows. Runner setup uses the official Ubuntu archive after repeated Azure-mirror download stalls, with separate bounded setup and test-step budgets.
- USD-aware ASR normalization handles “$14 million” versus written “fourteen million dollar,” preserving script text and estimated timing. Missing currency words without a currency symbol still reject. Rejected speech attempts preserve bounded waveform/transcript/error evidence; successful ASR comparison remains an estimate, not acoustic certification of every phoneme.

## Model and repair findings

The former compact Qwen3 writer repeatedly failed timing and produced generic narration. The [Qwen3.5 4B probe](https://github.com/justsomeoneretarded/Recap-Short-form/actions/runs/42161702969) invented a suit price and generated glued words under lexical masking; it was not promoted. The [Ministral3 probe](https://github.com/justsomeoneretarded/Recap-Short-form/actions/runs/42162747992) copied incomplete captions and missed budgets. These are case-specific results, not general rankings.

Gemma4 produced coherent suit→broken-rule→bounty narration and subsequently passed the full workflow. Generation uses ordinary structured JSON without lexical masks. Runtime checks enforce IDs, source ranges, complete-sentence shape and word budgets. Repair targets2.0words/second and uses the same existing2.5-word/second admission ceiling as initial planning; its global2.2-word/second limit remains. A remaining repair attempt may restore omitted supported facts/names after undershooting the minimum. No code truncates text, invents punctuation, loops footage or cuts speech to force success.

## Scope and remaining review

- One15-second-maximum request was validated. A cold-source150-second production, broad movie coverage, writer-quality ranking and Turbo speech inference remain unverified.
- Frame planning observes one midpoint per indexed span, not continuous motion. Automatic source-quote selection uses English ASR estimates and preserves recognized words/times by ID; it is not speaker identification or acoustic certification. Detailed reframing, story logic and character identity need review.
- Pacing is not calibrated to the references. The planner targets120words/minute and admits at most150words/minute against selected footage. The references measure206–231mixed-speech words/minute, including apparent movie dialogue; that is not an isolated narrator rate. Matching passages on clean source footage should guide future calibration.
- Final voice likeness/delivery, wording, action correspondence, captions, covers and mix need human playback review. Cover candidates are sampled timeline frames. Package/runtime versions and model tags can change; the workflow retains resolved-environment/model diagnostics and fails rather than substituting fake media.
- A channel-ready recap requires clean movie/scene footage. This project produces reviewable files and does not publish to YouTube or promise retention, monetization, or identical results to a reference channel.

Earlier checks remain distinct: the [11.901second reviewed pilot](validation-samples/original-warguy-engineering-pilot.mp4) bypassed writing. The [11.201second stage-resumed automatic-plan8 video](https://github.com/justsomeoneretarded/Recap-Short-form/blob/a77189a/docs/validation-samples/automatic-warguy-pilot.mp4) used a model-generated storyboard after cancellation; its [original plan](validation-samples/staged-plan8-original-plan.json) and [audit](validation-samples/staged-plan8-verification.json) are archived. Neither earlier check was proof of one complete automatic workflow.

## Gemini edition (11 November 2026)

**Verified here (offline):**

- **190 tests pass** (2 opt-in live tests skipped) on Python 3.11 with FFmpeg and google-genai 3.1.0. That is the 148 original tests plus 42 new ones.
- Gemini behaviour comes from a scripted fake transport derived from the prompt evidence. Synthetic media (testsrc2 and tones) and a fake synthesizer stand in for Chatterbox.
- The new tests cover:
  - a complete end-to-end run to a rendered, verified MP4 with every artifact
  - resume after an interrupted run with no repeated stages or upload
  - pre-flight and per-call budget stops
  - narration overflow repaired by a targeted rewrite, and an unfixable overflow failing clearly
  - failed corrections reported as `needs_human_review`, and `strict_review` failing instead
  - a review-driven shot rematch
  - local fallback when models are unavailable
  - missing or invalid key, unsupported models, rate limits, timeouts, malformed output, key redaction, response and upload caching
  - rejection of out-of-window timestamps, unheard dialogue, unsupported names, unobserved events, unknown event/shot IDs, chronology errors, invented quotes, channel filler, dangling sentences, footage reuse and actions trimmed away
  - sentence-segmented frame allocation
  - Stage J detection of wrong geometry, repeated footage and out-of-range sources in a real encoded file
  - workflow input parsing and secret scoping
- `tests/test_gemini_sdk_contract.py` builds real SDK request objects without network access:
  - clipped `VideoMetadata` parts, `media_resolution`, JSON response config, and both `thinking_budget` and `thinking_level` styles
  - conversion of all nine response schemas
  - classification of `ClientError` / `ServerError` codes
- Adding the `gemini` extra to the voice/analysis/source dependency set resolves with no change to existing pins (`uv pip compile`, PyPI metadata).

**Not verified, and needs a live key and footage:**

- No real Gemini call has been made from this environment: the docs site and API were unreachable here. Model availability, real response quality, token counts, timestamp accuracy and actual costs are therefore unmeasured.
- Model IDs come from the installed SDK's catalogue and are resolved at runtime. Prices for models newer than Gemini 3 preview are conservative placeholders.
- No 135–180-second Short has been produced by the Gemini path. There is no local-vs-Gemini comparison on the same clean footage, so **no quality improvement is claimed yet**.
- Chatterbox, Whisper and MMS forced alignment were not executed in these tests. MMS alignment is implemented as an option with an agreement guard, but it is unexercised on real audio.
- The Generate Shorts workflow change has not been run on GitHub Actions.

Recommended first live check: run **Gemini Live Check**, then Generate Shorts with `gemini-economy` on a 2–3-minute clean excerpt. Then run `gemini-balanced` and `local-gemma4` on the same excerpt, and compare factual accuracy, sentence-to-shot relevance (`evidence-map.json`), sync, review findings and `cost-report.json`.
